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
from sglang.srt.mem_cache.lora_radix_cache_utils import AdapterNode, RootNode
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MHATokenToKVPoolHost,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.mem_cache.tree_sieve import TreeSieve

logger = logging.getLogger(__name__)


class BaseEvictionPolicy(ABC):
    """Eviction policy in lora radix tree."""

    @abstractmethod
    def get_adapter_recover_utility(self, **kwargs) -> float:
        pass

    @abstractmethod
    def get_token_recover_utility(self, **kwargs) -> float:
        pass


class LRUEvictionPolicy(BaseEvictionPolicy):
    def get_adapter_recover_utility(self, adapter_node: AdapterNode) -> float:
        return adapter_node.last_access_time

    def get_token_recover_utility(self, node: TreeNode) -> float:
        return node.last_access_time


class LFUEvictionPolicy(BaseEvictionPolicy):
    def get_adapter_recover_utility(self, adapter_node: AdapterNode) -> float:
        return adapter_node.hit_count

    def get_token_recover_utility(self, node: TreeNode) -> float:
        return node.hit_count


class CachePolicyType:
    LRU = "lru"  # Multi-Head Attention
    LFU = "lfu"
    TreeSieve = "tree_sieve"  # Multi-head Latent Attention


class LoraRadixCache(RadixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        page_size: int,
    ):
        super().__init__(
            req_to_token_pool, token_to_kv_pool_allocator, page_size, disable=False
        )
        self.root_node = RootNode()
        self.evictable_size_ = 0
        self.protected_size_ = 0

        # cache policy option
        self.cache_policy_option = CachePolicyType.LFU
        self.disable = False
        self.tree_sieve_cache_policy: TreeSieve = TreeSieve(
            adapter_based_radix_tree=self.root_node,
            radix_tree_evict_callback=self._evict_callback,
            get_evictable_size_callback=self._get_evictable_size,
        )

        if self.cache_policy_option == CachePolicyType.LRU:
            self.unified_cache_policy: BaseEvictionPolicy = LRUEvictionPolicy()
        elif self.cache_policy_option == CachePolicyType.LFU:
            self.unified_cache_policy: BaseEvictionPolicy = LFUEvictionPolicy()

    ##### Public API #####
    def reset(self):
        super().reset()
        self.root_node = RootNode()

    def _get_evictable_size(self):
        return self.evictable_size_

    def match_prefix(
        self, adapter_name: str, key: List[int], is_sieve_get=True, **kwargs
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

        if adapter_name not in self.root_node.children:
            self.root_node.children[adapter_name] = AdapterNode(adapter_name)
        adapter_node = self.root_node.children[adapter_name]

        if self.disable:
            return [], adapter_node

        value, last_node = self._match_prefix_helper(adapter_node, key, is_sieve_get)
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
        self,
        lora_paths_in_batch: List[str],
        adapter_infos: Dict[str, AdapterInfo],
        is_prefill: bool,
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

            if adapter_node.is_empty:
                self._fill_adapter_node(
                    adapter_info=adapter_infos[adapter_name], adapter_node=adapter_node
                )
                if is_prefill:
                    import json

                    path = "/u/cjia/sglang-common/sglang/cache_lora_benchmark/acc.txt"
                    record = {"new": adapter_node.value.size, "cached": 0}
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
            else:
                if is_prefill:
                    import json

                    path = "/u/cjia/sglang-common/sglang/cache_lora_benchmark/acc.txt"
                    record = {
                        "new": adapter_node.value.size,
                        "cached": adapter_node.value.size,
                    }
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")

        # self.check_memory_leak()

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

            # TODO please check it @ Chaobo Jia
            self.dec_lock_ref(req.last_node)
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
        # self.tree_sieve_cache_policy.print_list()
        # self.check_memory_leak()

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
        new_indices, new_last_node = self.match_prefix(
            req.lora_path, token_ids, is_sieve_get=False
        )
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
        # self.tree_sieve_cache_policy.print_list()
        # if self.tree_sieve_cache_policy.hand is not None:
        #     print('hand:' ,self.tree_sieve_cache_policy.hand.key)
        # self.check_memory_leak()

    def _evict_callback(self, node):
        evict_size = 0
        if isinstance(node, AdapterNode):
            self.token_to_kv_pool_allocator.free(node.value.loc)

            # import json
            # path = "/u/cjia/sglang-common/sglang/cache_lora_benchmark/trace/acc.txt"
            # record = "evict_adapter_node adapter_name: " + x.adapter_name
            # with open(path, "a", encoding="utf-8") as f:
            #     f.write(json.dumps(record) + "\n")
            evict_size += node.value.size
            self._evict_adapter_node(node)
        elif isinstance(node, TreeNode):
            self.token_to_kv_pool_allocator.free(node.value)
            evict_size += len(node.value)
            self._delete_leaf(node)
        else:
            raise ValueError("error!!!!!")

        return evict_size

    def force_evict(self, num_cells):
        evictable_nodes = [
            (node.hit_count, node.id, node) for node in self._collect_evictable_nodes()
        ]
        heapq.heapify(evictable_nodes)

        num_evicted = 0
        while num_evicted < num_cells and len(evictable_nodes):
            _, _, x = heapq.heappop(evictable_nodes)

            assert x.lock_ref <= 0

            if isinstance(x, AdapterNode):
                self.token_to_kv_pool_allocator.free(x.value.loc)
                num_evicted += x.value.size

                import json

                path = "/u/cjia/sglang-common/sglang/cache_lora_benchmark/acc.txt"
                record = "evict_adapter_node adapter_name: " + x.adapter_name
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                # print('evict_adapter_node adapter_name:',x.adapter_name)
                self._evict_adapter_node(x)

            elif isinstance(x, TreeNode):
                self.token_to_kv_pool_allocator.free(x.value)
                num_evicted += len(x.value)
                self._delete_leaf(x)

                if (
                    len(x.parent.children) == 0
                    and not isinstance(x.parent, AdapterNode)
                    and x.parent.lock_ref <= 0
                ):
                    heapq.heappush(
                        evictable_nodes, (x.parent.hit_count, x.parent.id, x.parent)
                    )
            else:
                raise ValueError("unknown error here")

    def evict(self, num_cells: int):
        print("evict num_token", num_cells)
        if self.cache_policy_option == CachePolicyType.TreeSieve:
            self.tree_sieve_cache_policy.evict(num_cells)
        elif self.cache_policy_option == CachePolicyType.LFU:
            evictable_nodes = [
                (node.hit_count, 1 / node.id, node)
                for node in self._collect_evictable_nodes()
            ]

            heapq.heapify(evictable_nodes)

            num_evicted = 0
            while num_evicted < num_cells and len(evictable_nodes):
                _, _, x = heapq.heappop(evictable_nodes)

                assert x.lock_ref <= 0

                if isinstance(x, AdapterNode):
                    self.token_to_kv_pool_allocator.free(x.value.loc)
                    num_evicted += x.value.size

                    import json

                    path = "/u/cjia/sglang-common/sglang/cache_lora_benchmark/acc.txt"
                    record = "evict_adapter_node adapter_name: " + x.adapter_name
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")

                    # print('evict_adapter_node adapter_name:',x.adapter_name)
                    self._evict_adapter_node(x)

                elif isinstance(x, TreeNode):
                    self.token_to_kv_pool_allocator.free(x.value)
                    num_evicted += len(x.value)
                    self._delete_leaf(x)

                    if (
                        len(x.parent.children) == 0
                        and not isinstance(x.parent, AdapterNode)
                        and x.parent.lock_ref <= 0
                    ):
                        heapq.heappush(
                            evictable_nodes, (x.parent.hit_count, x.parent.id, x.parent)
                        )
                else:
                    raise ValueError("unknown error here")
        else:

            def get_utility(node):
                if isinstance(node, AdapterNode):
                    return self.unified_cache_policy.get_adapter_recover_utility(node)
                elif isinstance(node, TreeNode):
                    return self.unified_cache_policy.get_token_recover_utility(node)

            evictable_nodes = [
                (get_utility(node), node) for node in self._collect_evictable_nodes()
            ]

            # print("evictable_nodes", evictable_nodes)

            heapq.heapify(evictable_nodes)

            num_evicted = 0
            while num_evicted < num_cells and len(evictable_nodes):
                _, x = heapq.heappop(evictable_nodes)

                assert x.lock_ref <= 0

                if isinstance(x, AdapterNode):
                    self.token_to_kv_pool_allocator.free(x.value.loc)
                    num_evicted += x.value.size

                    import json

                    path = "/u/cjia/sglang-common/sglang/cache_lora_benchmark/acc.txt"
                    record = "evict_adapter_node adapter_name: " + x.adapter_name
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")

                    # print('evict_adapter_node adapter_name:',x.adapter_name)
                    self._evict_adapter_node(x)

                elif isinstance(x, TreeNode):
                    self.token_to_kv_pool_allocator.free(x.value)
                    num_evicted += len(x.value)
                    self._delete_leaf(x)

                    if (
                        len(x.parent.children) == 0
                        and not isinstance(x.parent, AdapterNode)
                        and x.parent.lock_ref <= 0
                    ):
                        heapq.heappush(
                            evictable_nodes, (get_utility(x.parent), x.parent)
                        )
                else:
                    raise ValueError("unknown error here")

        # self.check_memory_leak()

    def inc_lock_ref(self, node: Union[AdapterNode, TreeNode]):
        if self.disable and isinstance(node, TreeNode):
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
        if self.disable and isinstance(node, TreeNode):
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

    def _inc_hit_count(self, node: AdapterNode | TreeNode):
        node.hit_count += 1

    def _match_prefix_helper(self, node: AdapterNode, key: List, is_sieve_get: bool):
        node.last_access_time = time.time()

        if is_sieve_get and not node.is_empty:
            self._tree_sieve_get(node.id)
        self._inc_hit_count(node)

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.time()
            prefix_len = self.key_match_fn(child.key, key)

            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node

                if is_sieve_get:
                    self._tree_sieve_get(new_node.id)
                self._inc_hit_count(new_node)

                break
            else:
                if is_sieve_get:
                    self._tree_sieve_get(child.id)
                self._inc_hit_count(child)

                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self._tree_sieve_split(new_node=new_node, id=child.id)

        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.time()
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)

            self._tree_sieve_insert(key=new_node.id, value=new_node)
        return total_prefix_length

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
        if self.token_to_kv_pool_allocator is not None:
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

        self._tree_sieve_insert(key=adapter_node.id, value=adapter_node)

    def _evict_adapter_node(self, adapter_node: AdapterNode):
        self.evictable_size_ -= adapter_node.value.size
        del self.root_node.active_adapters[adapter_node.adapter_name]
        adapter_node.value = None

    def _collect_evictable_nodes(self):
        ret_list = []
        for _, adapter_node in self.root_node.children.items():
            if not adapter_node.is_empty and adapter_node.lock_ref <= 0:
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

    def _tree_sieve_insert(self, key: int, value: AdapterNode | TreeNode):
        if self.cache_policy_option == CachePolicyType.TreeSieve:
            self.tree_sieve_cache_policy.insert(key=key, value=value)

    def _tree_sieve_get(self, key: int):
        if self.cache_policy_option == CachePolicyType.TreeSieve:
            self.tree_sieve_cache_policy.get(key=key)

    def _tree_sieve_split(self, new_node: AdapterNode | TreeNode, id: int):
        if self.cache_policy_option == CachePolicyType.TreeSieve:
            self.tree_sieve_cache_policy.split(
                new_node=new_node,
                id=id,
            )

    def check_memory_leak(self):
        available_size = (
            self.token_to_kv_pool_allocator.available_size() + self.evictable_size()
        )
        protected_size = self.protected_size()
        memory_leak = available_size != (
            self.token_to_kv_pool_allocator.size - protected_size
        )
        if memory_leak:
            msg = (
                "KV cache pool leak detected! "
                f"{available_size=}, {protected_size=}, {self.token_to_kv_pool_allocator.size=}\n"
                f"{self.token_to_kv_pool_allocator.available_size()=}\n"
                f"{self.evictable_size()=}\n"
            )
            raise ValueError(msg)


if __name__ == "__main__":
    tree = LoraRadixCache(None, None, 1)
    # tree.cache_adapters(lora_paths_in_batch = ['lora1'],
    #                     adapter_infos = {'lora1': AdapterInfo(rank = 0,
    #                                                             loc = None,
    #                                                             size = 100)})
    # tree.cache_adapters(lora_paths_in_batch = ['lora2'],
    #                     adapter_infos = {'lora2': AdapterInfo(rank = 0,
    #                                                             loc = None,
    #                                                             size = 120)})

    adapter_name = "lora1"

    # tree.insert(adapter_name, "Hello")
    tree.insert(adapter_name, "Hellomon")
    tree.insert(adapter_name, "Hello_L.A.!")

    adapter_name = "lora2"
    tree.insert(adapter_name, "Hello_world! Happy")
    tree.insert(adapter_name, "I love you!")
    # tree.tree_sieve_cache_policy.print_list()
    tree.pretty_print()
    print()
    print()
    print()
    print()

    # print(tree.match_prefix('lora1',"Hello_L.A.! aha"))
    # print(tree.match_prefix('lora1',"www"))

    # tree.tree_sieve_cache_policy.get(1)
    # tree.tree_sieve_cache_policy.get(2)

    # tree.evict(5)
    tree.evict(10)

    # tree.tree_sieve_cache_policy.print_list()
    tree.pretty_print()
