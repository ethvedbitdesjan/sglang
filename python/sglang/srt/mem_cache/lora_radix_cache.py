import heapq
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.lora.utils import AdapterInfo
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MHATokenToKVPoolHost,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

logger = logging.getLogger(__name__)


class RootNode:
    def __init__(self):
        # adapter_name -> adapter_node
        self.children: Dict[str, AdapterNode] = {}

        # adapters in the gpu memory
        self.active_adapters: Dict[str, AdapterInfo] = {}


class AdapterNode:
    def __init__(self, adapter_name: str):
        self.adapter_name = adapter_name
        self.parent = None
        self.children = defaultdict(TreeNode)
        self.value: AdapterInfo = None
        self.lock_ref = 0
        self.last_access_time = time.time()

        self.hit_count = 0
        # indicating the node is loading adapter from host
        self.loading = False
        # store the host indices of adapters
        self.host_value = None

    @property
    def evicted(self):
        return self.value is None

    # @property
    # def backuped(self):
    #     return self.host_value is not None


class BaseEvictionPolicy(ABC):
    """Eviction policy in lora radix tree."""

    @abstractmethod
    def get_adapter_recover_utility(self, **kwargs) -> float:
        pass

    @abstractmethod
    def get_token_recover_utility(self, **kwargs) -> float:
        pass


class ComputeOnlyEvictionPolicy(BaseEvictionPolicy):

    def get_adapter_recover_utility(self, adapter_node: AdapterNode) -> float:
        return adapter_node.last_access_time

    def get_token_recover_utility(self, node: TreeNode) -> float:
        return node.last_access_time


class LoraRadixCache(RadixCache):

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        page_size: int,
    ):
        self.eviction_policy: BaseEvictionPolicy = ComputeOnlyEvictionPolicy()
        super().__init__(
            req_to_token_pool, token_to_kv_pool_allocator, page_size, disable=False
        )

    ##### Public API #####

    def reset(self):
        super().reset()
        self.root_node = RootNode()

    def match_prefix(
        self, adapter_name: str, key: List[int], **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """Find the matching prefix from the radix tree.
        Args:
            key: A list of token IDs to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        if self.disable:
            return [], self.root_node

        if adapter_name not in self.root_node.children:
            self.root_node.children[adapter_name] = AdapterNode(adapter_name)
        adapter_node = self.root_node.children[adapter_name]

        value, last_node = self._match_prefix_helper(adapter_node, key)
        if value:
            value = torch.concat(value)
        else:
            value = torch.tensor([], dtype=torch.int32)
        return value, last_node

    def insert(self, adapter_name: str, key: List, value=None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]

        if adapter_name not in self.root_node.children:
            self.root_node.children[adapter_name] = AdapterNode(adapter_name)

        adapter_node = self.root_node.children[adapter_name]

        return self._insert_helper(adapter_node, key, value)

    def cache_adapters(
        self, lora_paths_in_batch: List[str], adapter_infos: Dict[str, AdapterInfo]
    ):
        """Cache adapters when they are invoked by forward_batch.
        Args:
            lora_paths_in_batch: lora paths for every req in the forward_batch
            adapter_infos:
        """
        for adapter_name in lora_paths_in_batch:
            if adapter_name is None:
                continue
            if adapter_name not in self.root_node.children:
                self.root_node.children[adapter_name] = AdapterNode(adapter_name)

            adapter_node = self.root_node.children[adapter_name]
            if adapter_node.value == None:
                self._fill_adapter_node(
                    adapter_info=adapter_infos[adapter_name], adapter_node=adapter_node
                )

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it finishes."""
        if self.disable:
            if token_ids is None:
                token_ids_len = len(req.origin_input_ids) + len(req.output_ids) - 1
            else:
                token_ids_len = len(token_ids)

            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :token_ids_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        if token_ids is None:
            token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(req.lora_path, token_ids, kv_indices.clone())
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

        # print(" ")
        # print("cache_finished_req")
        # self.pretty_print()

    def cache_unfinished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        if token_ids is None:
            token_ids = req.fill_ids

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(req.lora_path, token_ids, kv_indices.clone())
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node = self.match_prefix(req.lora_path, token_ids)
        assert len(new_indices) == len(token_ids)
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        req.prefix_indices = new_indices
        req.last_node = new_last_node

        # print(" ")
        # print("cache_unfinished_req")
        # self.pretty_print()

    def evict(self, num_cells: int):
        print("evict num_token", num_cells)

        def get_utility(node):
            if isinstance(node, AdapterNode):
                return self.eviction_policy.get_adapter_recover_utility(node)
            elif isinstance(node, TreeNode):
                return self.eviction_policy.get_token_recover_utility(node)

        if self.disable:
            return

        evictable_nodes = [
            (get_utility(node), node) for node in self._collect_evictable_nodes()
        ]

        heapq.heapify(evictable_nodes)

        num_evicted = 0
        while num_evicted < num_cells and len(evictable_nodes):
            _, x = heapq.heappop(evictable_nodes)

            assert x.lock_ref <= 0

            if isinstance(x, AdapterNode):
                self.token_to_kv_pool_allocator.free(x.value.loc)
                num_evicted += x.value.size
                self._evict_adapter_node(x)

            elif isinstance(x, TreeNode):
                self.token_to_kv_pool_allocator.free(x.value)
                num_evicted += len(x.value)
                self._delete_leaf(x)

                if len(x.parent.children) == 0 and not isinstance(
                    x.parent, AdapterNode
                ):
                    heapq.heappush(evictable_nodes, (get_utility(x.parent), x.parent))
            else:
                raise ValueError("unknown error here")

    def inc_lock_ref(self, node: Union[AdapterNode, TreeNode]):
        if self.disable:
            return 0

        delta = 0
        while node.parent != None:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent

        # for adapter node
        if node.lock_ref == 0:
            if node.value != None:
                self.evictable_size_ -= node.value.size
                self.protected_size_ += node.value.size
                delta -= node.value.size
        node.lock_ref += 1

        return delta

    def dec_lock_ref(self, node: Union[AdapterNode, TreeNode]):
        if self.disable:
            return 0

        delta = 0
        while node.parent != None:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent

        # for adapter node
        if node.lock_ref == 1:
            if node.value != None:
                self.evictable_size_ += node.value.size
                self.protected_size_ -= node.value.size
                delta += node.value.size
        node.lock_ref -= 1

        return delta

    ##### Internal Helper Functions #####

    def _print_helper(self, root_node: RootNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        for _, adapter_node in root_node.children.items():
            stack = [(adapter_node, indent)]
            while stack:
                current_node, current_indent = stack.pop()
                if isinstance(current_node, AdapterNode):
                    if current_node.value != None:
                        print(
                            " " * current_indent,
                            f"{current_node.adapter_name}",
                            "yes",
                            f"r={current_node.lock_ref}",
                        )
                    else:
                        print(
                            " " * current_indent,
                            f"{current_node.adapter_name}",
                            "no",
                            f"r={current_node.lock_ref}",
                        )
                else:
                    print(
                        " " * current_indent,
                        len(current_node.key),
                        current_node.key[:10],
                        f"r={current_node.lock_ref}",
                    )

                for _, child in current_node.children.items():
                    stack.append((child, current_indent + 3))

        print("protected_size", self.protected_size_)
        print("evictable_size", self.evictable_size_)
        print("available_size", self.token_to_kv_pool_allocator.available_size())

    def _total_size_helper(self):
        total_size = 0
        for _, adapter_node in self.root_node.children.items():
            stack = [adapter_node]
            while stack:
                current_node = stack.pop()
                if isinstance(current_node, AdapterNode):
                    if current_node.value != None:
                        total_size += current_node.value.size
                else:
                    total_size += len(current_node.value)
                for child in current_node.children.values():
                    if child.evicted:
                        continue
                    stack.append(child)
        return total_size

    def _fill_adapter_node(self, adapter_info: AdapterInfo, adapter_node: AdapterNode):
        assert adapter_node.lock_ref > 0
        adapter_node.value = adapter_info
        self.protected_size_ += adapter_node.value.size
        self.root_node.active_adapters[adapter_node.adapter_name] = adapter_node.value

    def _evict_adapter_node(self, adapter_node: AdapterNode):
        self.evictable_size_ -= adapter_node.value.size
        del self.root_node.active_adapters[adapter_node.adapter_name]
        adapter_node.value = None

    def _collect_evictable_nodes(self):
        ret_list = []
        for _, adapter_node in self.root_node.children.items():
            if adapter_node.value != None and adapter_node.lock_ref <= 0:
                ret_list.append(adapter_node)
            if len(adapter_node.children) != 0:
                stack = list(adapter_node.children.values())
                while stack:
                    cur_node = stack.pop()
                    if len(cur_node.children) == 0:
                        if cur_node.lock_ref <= 0:
                            ret_list.append(cur_node)
                    else:
                        stack.extend(cur_node.children.values())
        return ret_list


if __name__ == "__main__":
    tree = LoraRadixCache(None, None)
    adapter_name = "lora1"

    tree.insert(adapter_name, "Hello")
    tree.insert(adapter_name, "Hello")
    tree.insert(adapter_name, "Hello_L.A.!")

    adapter_name = "lora2"
    tree.insert(adapter_name, "Hello_world! Happy")
    tree.insert(adapter_name, "I love you!")
    tree.pretty_print()

    # print(tree.match_prefix('lora1',"Hello_L.A.! aha"))
    # print(tree.match_prefix('lora1',"www"))

    def evict_callback(x):
        print("evict", x)
        return len(x)

    tree.evict(5, evict_callback)
    tree.evict(10, evict_callback)
    tree.pretty_print()
