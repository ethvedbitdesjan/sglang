import heapq
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

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

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, key: int, value: Any):
        self.key: int = key
        self.value: Any = value
        self.prev: Node = None
        self.next: Node = None
        self.visited: bool = False

    def __repr__(self):
        return f"Node(" f"key={self.key}, " f"visited={self.visited}," f")"


class TreeSieve:
    def __init__(
        self,
        adapter_based_radix_tree: RootNode,
        radix_tree_evict_callback,
        get_evictable_size_callback,
    ):

        self.adapter_based_radix_tree: RootNode = adapter_based_radix_tree

        self.head: Node = None
        self.tail: Node = None
        self.size: int = 0

        self.hand: Node = None
        self.key_value_dict: Dict[int, Node] = {}

        self.radix_tree_evict_callback = radix_tree_evict_callback
        self.get_evictable_size_callback = get_evictable_size_callback

    def get(self, key: int):
        node = self.key_value_dict[key]
        if node != None:
            node.visited = True

    def insert(self, key: int, value: AdapterNode | TreeNode):
        new_sieve_node = Node(key, value)
        self._add_node_at_head(new_sieve_node)
        self._put_key_value_dict(key, new_sieve_node)

    def evict(self, num_cells: int):
        num_evicted = 0
        while num_evicted < num_cells:
            evict_size = self._evict_once(value=num_cells)
            num_evicted += evict_size

            if evict_size == 0:
                break

    def split(self, new_node: TreeNode, id: int):
        target_sieve_node = self.key_value_dict[id]
        new_sieve_node = Node(new_node.id, new_node)
        self._insert_after(target_node=target_sieve_node, new_node=new_sieve_node)
        self._put_key_value_dict(new_node.id, new_sieve_node)

    # evict callback
    def _prune_radix_tree(self, node: AdapterNode | TreeNode):
        # print(' _prune_radix_tree',node.id)
        evict_size = 0
        queue = deque()
        queue.append(node)
        while queue:
            current_node = queue.popleft()
            if len(current_node.children) != 0:
                for _, child in current_node.children.items():
                    queue.append(child)

            # delete node in the sieve
            # print('delete node: ',current_node.id)
            sieve_node = self.key_value_dict[current_node.id]
            self._delete_key_value_dict(sieve_node.key)
            self._remove_node(sieve_node)

            # delete node in the adapter-based radix tree
            evict_size += self.radix_tree_evict_callback(current_node)
        return evict_size

    def _put_key_value_dict(self, key: int, node: Node):
        self.key_value_dict[key] = node

    def _delete_key_value_dict(self, key: int):
        del self.key_value_dict[key]

    def _evict_once(self, value):
        n = self.hand

        self.evict_list = []

        if n == None:
            n = self.tail

        if self.tail == None:
            return

        # while ((n != self.head) and (n.value.lock_ref > 0 or n.visited)):
        #     if n.value.lock_ref <= 0:
        #         n.visited = False
        #     self.evict_list.append(("id="+str(n.key),n.value.lock_ref,n.visited))
        #     n = n.prev
        # if (n.value.lock_ref > 0 or n.visited):
        #     if n.value.lock_ref <= 0:
        #         n.visited = False
        #     n = self.tail
        #     while ((n != self.head) and (n.value.lock_ref > 0 or n.visited)):
        #         if n.value.lock_ref <= 0:
        #             n.visited = False
        #         self.evict_list.append(("id="+str(n.key),n.value.lock_ref,n.visited))
        #         n = n.prev
        #     if (n.value.lock_ref > 0 or n.visited):
        #         if n.value.lock_ref <= 0:
        #             n.visited = False
        #         n = None

        while n != None and (n.value.lock_ref > 0 or n.visited):
            if n.value.lock_ref <= 0:
                n.visited = False
            self.evict_list.append(("id=" + str(n.key), n.value.lock_ref, n.visited))
            n = n.prev
        if n == None:
            n = self.tail
        while n != None and (n.value.lock_ref > 0 or n.visited):
            if n.value.lock_ref <= 0:
                n.visited = False
            self.evict_list.append(("id=" + str(n.key), n.value.lock_ref, n.visited))
            n = n.prev
        if n == None:
            n = self.tail
        while n != None and (n.value.lock_ref > 0 or n.visited):
            if n.value.lock_ref <= 0:
                n.visited = False
            self.evict_list.append(("id=" + str(n.key), n.value.lock_ref, n.visited))
            n = n.prev

        evict_size = 0
        if n != None:
            self.hand = n
            evict_size = self._prune_radix_tree(n.value)
        else:
            # print('evict_num now',value)
            # print('get_evictable_size_callback',self.get_evictable_size_callback())
            # self.print_list()
            # self._print_helper(self.adapter_based_radix_tree ,0)
            # print('hand',self.hand)
            # print('evict_list',self.evict_list)
            # raise ValueError('n == None')
            1

        return evict_size

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

    def _add_node_at_head(self, node):
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
        self.size += 1

    def _insert_after(self, target_node, new_node):
        new_node.prev = target_node
        new_node.next = target_node.next

        if target_node.next is not None:
            target_node.next.prev = new_node
        target_node.next = new_node

        if target_node == self.tail:
            self.tail = new_node

        self.size += 1

    def _remove_node(self, node):
        if self.size == 0:
            return

        if self.size == 1:
            self.head = None
            self.tail = None
            self.hand = None
        else:
            if node == self.head:
                self.head = self.head.next
                self.head.prev = None
            if node == self.tail:
                self.tail = self.tail.prev
                self.tail.next = None

            if self.hand == node:
                self.hand = self.hand.prev

            if node.prev != None:
                node.prev.next = node.next
            if node.next != None:
                node.next.prev = node.prev

        node.prev = None
        node.next = None
        self.size -= 1

    def get_size(self):
        return self.size

    def is_empty(self):
        return self.size == 0

    def print_list(self):
        current = self.head
        values = []
        while current:
            values.append(str(current.value) + "visited:" + str(current.visited))
            current = current.next
        print(" <-> ".join(values) if values else "Empty List")


def evict_callback(node):
    # print('evict node:',node.id)
    if isinstance(node, AdapterNode):
        return node.value.size
    elif isinstance(node, TreeNode):
        return len(node.value)


def evictable_callback():
    1


if __name__ == "__main__":
    root_node = RootNode()
    root_node.children["adapter_name0"] = AdapterNode("adapter_name0")
    root_node.children["adapter_name1"] = AdapterNode("adapter_name1")
    root_node.children["adapter_name2"] = AdapterNode("adapter_name2")
    root_node.children["adapter_name3"] = AdapterNode("adapter_name3")

    root_node.children["adapter_name0"].value = AdapterInfo(rank=0, loc=None, size=1)
    root_node.children["adapter_name1"].value = AdapterInfo(rank=0, loc=None, size=1)
    root_node.children["adapter_name2"].value = AdapterInfo(rank=0, loc=None, size=1)
    root_node.children["adapter_name3"].value = AdapterInfo(rank=0, loc=None, size=1)

    # father = root_node.children['adapter_name2']
    # new_node_1 = TreeNode()
    # new_node_1.parent = father
    # new_node_1.value = [1]
    # father.children[0] = new_node_1

    # new_node_2 = TreeNode()
    # new_node_2.parent = father
    # new_node_2.value = [1]
    # father.children[1] = new_node_2

    # new_node_3 = TreeNode()
    # new_node_3.parent = father
    # new_node_3.value = [1]
    # father.children[2] = new_node_3

    # root_node.children['adapter_name1'].lock_ref = 1

    tree_sieve = TreeSieve(
        adapter_based_radix_tree=root_node,
        radix_tree_evict_callback=evict_callback,
        get_evictable_size_callback=evictable_callback,
    )

    tree_sieve.insert(0, root_node.children["adapter_name0"])
    tree_sieve.insert(1, root_node.children["adapter_name1"])
    tree_sieve.insert(2, root_node.children["adapter_name2"])
    tree_sieve.insert(3, root_node.children["adapter_name3"])

    # tree_sieve.insert(3, new_node_1)
    # tree_sieve.insert(4, new_node_2)

    tree_sieve.get(0)

    tree_sieve.evict(1)

    root_node.children["adapter_name2"].lock_ref = 1
    root_node.children["adapter_name3"].lock_ref = 1

    # tree_sieve.get(2)
    # tree_sieve.get(3)

    tree_sieve.evict(1)

    # tree_sieve.split(new_node_3,3)

    print(tree_sieve.hand)
    tree_sieve.print_list()
    print(tree_sieve.key_value_dict)
