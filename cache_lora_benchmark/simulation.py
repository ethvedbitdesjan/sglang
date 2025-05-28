import collections

import numpy as np

#!/usr/bin/env python3
"""
example usage
for i in 0.2 0.4 0.6 0.8 1 1.2 1.4 1.6; do
    python3 dataGen.py -m 1000000 -n 100000000 --alpha $i > /disk/data/zipf_${i}_1_100 &
done

for i in 0.2 0.4 0.6 0.8 1 1.2 1.4 1.6; do
    python3 dataGen.py -m 10000000 -n 100000000 --alpha $i > /disk/data/zipf_${i}_10_100 &
done


"""


def get_adapter_node_key(adapter_id) -> int:
    return adapter_id * 20000000


def get_system_prompt_node_key(adapter_id, system_prompt_id) -> int:
    return adapter_id * 20000000 + system_prompt_id * 200000 + 1


def get_multi_turn_node_key(
    adapter_id, system_prompt_id, conversation_id, turn_id
) -> int:
    return (
        adapter_id * 20000000
        + system_prompt_id * 200000
        + conversation_id * 200
        + turn_id
        + 2
    )


from typing import Any, Dict


class TreeSieveNode:
    def __init__(self, key: int, value: Any):
        self.key: int = key
        self.value: Any = value
        self.prev: TreeSieveNode = None
        self.next: TreeSieveNode = None
        self.visited: bool = False

    def __repr__(self):
        return f"Node(" f"key={self.key}, " f"visited={self.visited}," f")"


class TreeSieve:
    def __init__(self, capacity: int):

        self.head: TreeSieveNode = None
        self.tail: TreeSieveNode = None
        self.size: int = 0
        self.total_num: int = 0
        self.capacity: int = capacity

        self.hand: TreeSieveNode = None
        self.key_value_dict: Dict[int, TreeSieveNode] = {}

    def get(self, key: int):
        if key in self.key_value_dict and self.key_value_dict[key] != None:
            self.key_value_dict[key].visited = True
            return 1
        else:
            return -1

    def insert(self, key: int, value: int):
        estimated_size = self.size + value
        if estimated_size > self.capacity:
            self.evict(estimated_size - self.capacity)

        new_sieve_node = TreeSieveNode(key, value)
        self._add_node_at_head(new_sieve_node)
        self._put_key_value_dict(key, new_sieve_node)

    def evict(self, num_cells: int):
        num_evicted = 0
        while num_evicted < num_cells:
            evict_size = self._evict_once()
            num_evicted += evict_size

            if evict_size == 0:
                break

    # evict callback
    def _prune_radix_tree(self, sieve_node: TreeSieveNode):
        # delete node in the sieve
        self._delete_key_value_dict(sieve_node.key)
        self._remove_node(sieve_node)
        # delete node in the adapter-based radix tree
        return sieve_node.value

    def _put_key_value_dict(self, key: int, node: TreeSieveNode):
        self.key_value_dict[key] = node

    def _delete_key_value_dict(self, key: int):
        del self.key_value_dict[key]

    def _evict_once(self):
        n = self.hand

        self.evict_list = []

        if n == None:
            n = self.tail

        if self.tail == None:
            return

        while n != None and n.visited:
            n.visited = False
            n = n.prev
        if n == None:
            n = self.tail
        while n != None and n.visited:
            n.visited = False
            n = n.prev
        if n == None:
            n = self.tail
        while n != None and n.visited:
            n.visited = False
            n = n.prev

        evict_size = 0
        if n != None:
            self.hand = n
            evict_size = self._prune_radix_tree(n)
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

    def _add_node_at_head(self, node):
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
        self.size += node.value

    def _insert_after(self, target_node, new_node):
        new_node.prev = target_node
        new_node.next = target_node.next

        if target_node.next is not None:
            target_node.next.prev = new_node
        target_node.next = new_node

        if target_node == self.tail:
            self.tail = new_node

        self.size += new_node.value

    def _remove_node(self, node):
        if self.size == 0:
            return

        # if self.size == 1:
        #     self.head = None
        #     self.tail = None
        #     self.hand = None
        # else:
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
        self.size -= node.value

    def get_size(self):
        return self.size

    def is_empty(self):
        return self.size == 0

    def print_list(self):
        current = self.head
        values = []
        while current:
            values.append(
                "(key:" + str(current.key) + ", visited:" + str(current.visited) + ")"
            )
            current = current.next
        print(" <-> ".join(values) if values else "Empty List")


import bisect
import math
import random
import struct
from functools import *

import numpy as np


class ZipfGenerator:

    def __init__(self, m, alpha):
        # Calculate Zeta values from 1 to n:
        tmp = [1.0 / (math.pow(float(i), alpha)) for i in range(1, m + 1)]
        zeta = reduce(lambda sums, x: sums + [sums[-1] + x], tmp, [0])

        # Store the translation map:
        self.distMap = [x / zeta[-1] for x in zeta]

    def next(self):
        # Take a uniform 0-1 pseudo-random value:
        u = random.random()

        # Translate the Zipf variable:
        return bisect.bisect(self.distMap, u) - 1


def gen_zipf(m: int, alpha: float, n: int, start: int = 0) -> np.ndarray:
    """generate zipf distributed workload

    Args:
        m (int): the number of objects
        alpha (float): the skewness
        n (int): the number of requests
        start (int, optional): start obj_id. Defaults to 0.

    Returns:
        requests that are zipf distributed
    """

    np_tmp = np.power(np.arange(1, m + 1), -alpha)
    np_zeta = np.cumsum(np_tmp)
    dist_map = np_zeta / np_zeta[-1]
    r = np.random.uniform(0, 1, n)
    return np.searchsorted(dist_map, r) + start


def gen_uniform(m: int, n: int, start: int = 0) -> np.ndarray:
    """generate uniform distributed workload

    Args:
        m (int): the number of objects
        n (int): the number of requests
        start (int, optional): start obj_id. Defaults to 0.

    Returns:
        requests that are uniform distributed
    """

    return np.random.uniform(0, m, n).astype(int) + start


lru_hit_num = 0
lru_total_num = 0

lfu_hit_num = 0
lfu_total_num = 0

sieve_hit_num = 0
sieve_total_num = 0


class Node:
    def __init__(self, key, value, freq):
        self.key = key
        self.value = value
        self.freq = freq

    def __repr__(self):
        return f"Node({self.key}, {self.value}, {self.freq})"


class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self._data = {}
        self.queue = {}
        self.min = 0

    def get(self, key: int) -> int:
        if key not in self._data:
            return -1

        node = self._data[key]
        self.queue[node.freq].remove(node)
        node.freq += 1
        if not self.queue.get(node.freq):
            self.queue[node.freq] = []
        self.queue[node.freq] = [node] + self.queue[node.freq]

        self._data[key] = node
        if (node.freq - 1 == self.min) and not len(self.queue[node.freq - 1]):
            self.min += 1

        return node.value

    def evict(self, num_cells: int):
        num_evicted = 0
        while num_evicted < num_cells:
            evict_size = self._evict_once()
            num_evicted += evict_size

            if evict_size == 0:
                break

    def _evict_once(self):
        node_list = self.queue[self.min]
        invalid_node = node_list.pop()
        self._data.pop(invalid_node.key)
        self.size -= invalid_node.value

        while not len(self.queue[self.min]):
            self.min += 1

        return invalid_node.value

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return

        node = self._data.get(key)
        if node:
            node_list = self.queue[node.freq]
            node_list.remove(node)

            node.freq = 1

            if self.queue.get(node.freq) is None:
                self.queue[node.freq] = []
            self.queue[node.freq] = [node] + self.queue[node.freq]

            self.min = 1
            return

        estimated_size = self.size + value
        if estimated_size > self.capacity:
            self.evict(estimated_size - self.capacity)

        self.min = 1
        new_node = Node(key, value, 1)
        if self.queue.get(new_node.freq) is None:
            self.queue[new_node.freq] = []
        self.queue[new_node.freq] = [new_node] + self.queue[new_node.freq]

        self._data[key] = new_node
        self.size += value


# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


class LRUCache(collections.OrderedDict):

    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
        self.size = 0

    def get(self, key: int) -> int:
        if key not in self:
            return -1

        self.move_to_end(key)
        return self[key]

    def evict(self, num_cells: int):
        num_evicted = 0
        while num_evicted < num_cells:
            evict_size = self._evict_once()
            num_evicted += evict_size

            if evict_size == 0:
                break

    def _evict_once(self):
        _, value = self.popitem(last=False)
        self.size -= value
        return value

    def put(self, key: int, value: int) -> None:
        if key in self:
            self.move_to_end(key)
            self[key] = value
            return

        estimated_size = self.size + value
        if estimated_size > self.capacity:
            self.evict(estimated_size - self.capacity)

        self[key] = value
        self.size += value


def lru_test():
    global lru_hit_num, lru_total_num
    cache = LRUCache(80)

    cache.put(0, 20)
    cache.put(1, 20)
    cache.put(2, 20)
    cache.put(3, 20)
    cache.get(0)
    cache.put(4, 50)
    print(cache)
    print(cache.size)


def lfu_test():
    global lfu_hit_num, lfu_total_num
    cache = LFUCache(80)

    cache.put(0, 20)
    cache.get(0)
    cache.get(0)
    cache.put(0, 20)
    print(cache.queue)
    print(cache.size)
    print(cache.min)

    # cache.put(1, 20)
    # cache.put(2, 20)
    # cache.put(3, 20)
    # cache.get(1)
    # cache.put(4, 50)
    # print(cache.queue)
    # print(cache.size)


from lru import LRU


def sieve_test():
    global sieve_total_num, sieve_hit_num
    cache = TreeSieve(80)

    cache.insert(0, 20)
    cache.insert(1, 20)
    cache.insert(2, 20)
    cache.insert(3, 20)
    cache.get(1)
    cache.insert(4, 50)
    cache.print_list()


def generate_system_prompts_trace(
    adapter_num,
    alpha,
    system_prompt_num,
):
    n = 100000
    obj_size = 1
    time_span = 100000
    batch_size = 100000

    trace = []
    i = 0

    m = adapter_num * system_prompt_num
    for n_batch in range((n - 1) // batch_size + 1):
        for obj in gen_zipf(m, alpha, batch_size):
            i += 1
            ts = i * time_span // n
            adapter_id = obj // system_prompt_num
            prompt_id = obj % system_prompt_num
            trace.append((ts, adapter_id, prompt_id))
    return trace


def generate_multi_turn_trace(
    adapter_num, alpha, system_prompt_num, multi_turn_num, think_time, req_num=100000
):
    n = req_num
    obj_size = 1
    time_span = req_num
    batch_size = req_num

    trace = []
    output_trace = []
    i = 0

    m = adapter_num * system_prompt_num
    for n_batch in range((n - 1) // batch_size + 1):
        for obj in gen_zipf(m, alpha, batch_size):
            i += 1
            ts = i * time_span // n
            adapter_id = obj // system_prompt_num
            prompt_id = obj % system_prompt_num
            trace.append((ts, adapter_id, prompt_id))

    conversation_id_dict = {}

    for info in trace:
        cur_ts = info[0]
        adapter_id = info[1]
        prompt_id = info[2]
        if (adapter_id, prompt_id) not in conversation_id_dict:
            conversation_id_dict[(adapter_id, prompt_id)] = 0

        conversation_id = conversation_id_dict[(adapter_id, prompt_id)]
        conversation_id_dict[(adapter_id, prompt_id)] += 1

        for turn_id in range(multi_turn_num):
            output_trace.append(
                (cur_ts, adapter_id, prompt_id, conversation_id, turn_id)
            )
            cur_ts += think_time

    output_trace = sorted(output_trace)

    return output_trace


class CacheType:
    system_prompt = "system_prompt"
    lora_adapter = "lora_adapter"
    multi_turn_q = "multi_turn_q"
    multi_turn_a = "multi_turn_a"


# size_dict = {CacheType.lora_adapter : 193,
#              CacheType.system_prompt : 200,

#             #  CacheType.multi_turn_q : 0,
#             #  CacheType.multi_turn_a : 0}

#              CacheType.multi_turn_q : 16,
#              CacheType.multi_turn_a : 64}

size_dict = {
    CacheType.lora_adapter: 193,
    CacheType.system_prompt: 200,
    #  CacheType.multi_turn_q : 0,
    #  CacheType.multi_turn_a : 0}
    CacheType.multi_turn_q: 16,
    CacheType.multi_turn_a: 64,
}


def system_prompts_lru_test(trace, cache_size):
    lru_hit_num = 0
    lru_total_num = 0

    lru_adapter_hit_num = 0
    lru_adapter_total_num = 0

    lru_prefix_hit_num = 0
    lru_prefix_total_num = 0

    cache = LRUCache(cache_size)
    for info in trace:
        adapter_id = info[1]
        prompt_id = info[2]

        adapter_key = get_adapter_node_key(adapter_id)
        prompt_key = get_system_prompt_node_key(
            adapter_id=adapter_id, system_prompt_id=prompt_id
        )
        adapter_size = size_dict[CacheType.lora_adapter]
        system_prompt_size = size_dict[CacheType.system_prompt]

        lru_total_num += adapter_size + system_prompt_size
        lru_adapter_total_num += adapter_size
        res = cache.get(adapter_key)
        if res == -1:
            cache.put(adapter_key, adapter_size)
        else:
            lru_hit_num += adapter_size
            lru_adapter_hit_num += adapter_size

        lru_prefix_total_num += system_prompt_size

        res = cache.get(prompt_key)
        if res == -1:
            cache.put(prompt_key, system_prompt_size)
        else:
            lru_hit_num += system_prompt_size
            lru_prefix_hit_num += system_prompt_size

    return (
        lru_hit_num / lru_total_num,
        lru_adapter_hit_num / lru_adapter_total_num,
        lru_prefix_hit_num / lru_prefix_total_num,
    )


def system_prompts_lfu_test(trace, cache_size):
    lfu_hit_num = 0
    lfu_total_num = 0

    lfu_adapter_hit_num = 0
    lfu_adapter_total_num = 0

    lfu_prefix_hit_num = 0
    lfu_prefix_total_num = 0

    cache = LFUCache(cache_size)
    for info in trace:
        adapter_id = info[1]
        prompt_id = info[2]

        adapter_key = get_adapter_node_key(adapter_id)
        prompt_key = get_system_prompt_node_key(
            adapter_id=adapter_id, system_prompt_id=prompt_id
        )
        adapter_size = size_dict[CacheType.lora_adapter]
        system_prompt_size = size_dict[CacheType.system_prompt]

        lfu_total_num += adapter_size + system_prompt_size
        lfu_adapter_total_num += adapter_size
        res = cache.get(adapter_key)
        if res == -1:
            cache.put(adapter_key, adapter_size)
        else:
            lfu_hit_num += adapter_size
            lfu_adapter_hit_num += adapter_size

        lfu_prefix_total_num += system_prompt_size

        res = cache.get(prompt_key)
        if res == -1:
            cache.put(prompt_key, system_prompt_size)
        else:
            lfu_hit_num += system_prompt_size
            lfu_prefix_hit_num += system_prompt_size

    return (
        lfu_hit_num / lfu_total_num,
        lfu_adapter_hit_num / lfu_adapter_total_num,
        lfu_prefix_hit_num / lfu_prefix_total_num,
    )


def system_prompts_sieve_test(trace, cache_size):
    sieve_hit_num = 0
    sieve_total_num = 0

    sieve_adapter_hit_num = 0
    sieve_adapter_total_num = 0

    sieve_prefix_hit_num = 0
    sieve_prefix_total_num = 0

    cache = TreeSieve(cache_size)
    for info in trace:
        adapter_id = info[1]
        prompt_id = info[2]

        adapter_key = get_adapter_node_key(adapter_id)
        prompt_key = get_system_prompt_node_key(
            adapter_id=adapter_id, system_prompt_id=prompt_id
        )
        adapter_size = size_dict[CacheType.lora_adapter]
        system_prompt_size = size_dict[CacheType.system_prompt]

        sieve_total_num += adapter_size + system_prompt_size
        sieve_adapter_total_num += adapter_size
        res = cache.get(adapter_key)
        if res == -1:
            cache.insert(adapter_key, adapter_size)
        else:
            sieve_hit_num += adapter_size
            sieve_adapter_hit_num += adapter_size

        sieve_prefix_total_num += system_prompt_size

        res = cache.get(prompt_key)
        if res == -1:
            cache.insert(prompt_key, system_prompt_size)
        else:
            sieve_hit_num += system_prompt_size
            sieve_prefix_hit_num += system_prompt_size

    return (
        sieve_hit_num / sieve_total_num,
        sieve_adapter_hit_num / sieve_adapter_total_num,
        sieve_prefix_hit_num / sieve_prefix_total_num,
    )


def system_prompts_benchmark():
    adapter_num = 64
    alpha = 1.0
    system_prompt_num = 4
    cache_size = 20000

    work_set_size = (
        adapter_num * size_dict[CacheType.lora_adapter]
        + system_prompt_num * adapter_num * size_dict[CacheType.system_prompt]
    )

    cache_ratio_list = [2, 5, 10, 20]
    for cache_ratio in cache_ratio_list:
        cache_size = int(work_set_size * cache_ratio / 100)

        print(work_set_size, cache_size)

        trace = generate_system_prompts_trace(adapter_num, alpha, system_prompt_num)

        lfu_hit, lfu_adapter, lfu_prefix = system_prompts_lfu_test(trace, cache_size)
        lru_hit, lru_adapter, lru_prefix = system_prompts_lru_test(trace, cache_size)
        sieve_hit, sieve_adapter, sieve_prefix = system_prompts_sieve_test(
            trace, cache_size
        )

        print("lru: ", lru_hit, lru_adapter, lru_prefix)
        print("lfu: ", lfu_hit, lfu_adapter, lfu_prefix)
        print("sieve: ", sieve_hit, sieve_adapter, sieve_prefix)
        print("------------------------")


def multi_turn_lru_test(trace, cache_size):
    lru_hit_num = 0
    lru_total_num = 0

    lru_adapter_hit_num = 0
    lru_adapter_total_num = 0

    lru_prefix_hit_num = 0
    lru_prefix_total_num = 0

    cache = LRUCache(cache_size)
    for info in trace:
        adapter_id = info[1]
        prompt_id = info[2]
        conversation_id = info[3]
        turn_id = info[4]

        adapter_key = get_adapter_node_key(adapter_id)
        prompt_key = get_system_prompt_node_key(
            adapter_id=adapter_id, system_prompt_id=prompt_id
        )
        adapter_size = size_dict[CacheType.lora_adapter]
        system_prompt_size = size_dict[CacheType.system_prompt]
        multi_turn_qa_size = (
            size_dict[CacheType.multi_turn_q] + size_dict[CacheType.multi_turn_a]
        )

        lru_total_num += adapter_size
        lru_adapter_total_num += adapter_size
        res = cache.get(adapter_key)
        if res == -1:
            cache.put(adapter_key, adapter_size)
        else:
            lru_hit_num += adapter_size
            lru_adapter_hit_num += adapter_size

        lru_total_num += system_prompt_size + multi_turn_qa_size * (turn_id + 1)
        lru_prefix_total_num += system_prompt_size + multi_turn_qa_size * (turn_id + 1)

        for i in range(turn_id, -1, -1):
            multi_turn_qa_key = get_multi_turn_node_key(
                adapter_id=adapter_id,
                system_prompt_id=prompt_id,
                conversation_id=conversation_id,
                turn_id=i,
            )
            res = cache.get(multi_turn_qa_key)
            if res != -1:
                lru_hit_num += multi_turn_qa_size
                lru_prefix_hit_num += multi_turn_qa_size
        res = cache.get(prompt_key)
        if res != -1:
            lru_hit_num += system_prompt_size
            lru_prefix_hit_num += system_prompt_size

        for i in range(turn_id, -1, -1):
            multi_turn_qa_key = get_multi_turn_node_key(
                adapter_id=adapter_id,
                system_prompt_id=prompt_id,
                conversation_id=conversation_id,
                turn_id=i,
            )
            cache.put(multi_turn_qa_key, multi_turn_qa_size)
        cache.put(prompt_key, system_prompt_size)

        # print(cache)
        # print(lru_hit_num,lru_total_num)

    return (
        lru_hit_num / lru_total_num,
        lru_adapter_hit_num / lru_adapter_total_num,
        lru_prefix_hit_num / lru_prefix_total_num,
    )


def multi_turn_lfu_test(trace, cache_size):
    lfu_hit_num = 0
    lfu_total_num = 0

    lfu_adapter_hit_num = 0
    lfu_adapter_total_num = 0

    lfu_prefix_hit_num = 0
    lfu_prefix_total_num = 0

    cache = LFUCache(cache_size)
    for info in trace:
        adapter_id = info[1]
        prompt_id = info[2]
        conversation_id = info[3]
        turn_id = info[4]

        adapter_key = get_adapter_node_key(adapter_id)
        prompt_key = get_system_prompt_node_key(
            adapter_id=adapter_id, system_prompt_id=prompt_id
        )
        adapter_size = size_dict[CacheType.lora_adapter]
        system_prompt_size = size_dict[CacheType.system_prompt]
        multi_turn_qa_size = (
            size_dict[CacheType.multi_turn_q] + size_dict[CacheType.multi_turn_a]
        )

        lfu_total_num += adapter_size
        lfu_adapter_total_num += adapter_size
        res = cache.get(adapter_key)
        if res == -1:
            cache.put(adapter_key, adapter_size)
        else:
            lfu_hit_num += adapter_size
            lfu_adapter_hit_num += adapter_size

        lfu_total_num += system_prompt_size + multi_turn_qa_size * (turn_id + 1)
        lfu_prefix_total_num += system_prompt_size + multi_turn_qa_size * (turn_id + 1)
        is_sys_prompt_exist = True
        insert_list = []

        for i in range(turn_id, -1, -1):
            multi_turn_qa_key = get_multi_turn_node_key(
                adapter_id=adapter_id,
                system_prompt_id=prompt_id,
                conversation_id=conversation_id,
                turn_id=i,
            )
            res = cache.get(multi_turn_qa_key)
            if res != -1:
                lfu_hit_num += multi_turn_qa_size
                lfu_prefix_hit_num += multi_turn_qa_size
            else:
                insert_list.append(multi_turn_qa_key)
        res = cache.get(prompt_key)
        if res != -1:
            lfu_hit_num += system_prompt_size
            lfu_prefix_hit_num += system_prompt_size
        else:
            is_sys_prompt_exist = False

        for key in insert_list:
            cache.put(key, multi_turn_qa_size)
        if not is_sys_prompt_exist:
            cache.put(prompt_key, system_prompt_size)

        # print(cache.queue)
        # print(lfu_hit_num,lfu_total_num)

    return (
        lfu_hit_num / lfu_total_num,
        lfu_adapter_hit_num / lfu_adapter_total_num,
        lfu_prefix_hit_num / lfu_prefix_total_num,
    )


def multi_turn_sieve_test(trace, cache_size):
    sieve_hit_num = 0
    sieve_total_num = 0

    sieve_adapter_hit_num = 0
    sieve_adapter_total_num = 0

    sieve_prefix_hit_num = 0
    sieve_prefix_total_num = 0

    cache = TreeSieve(cache_size)
    for info in trace:
        adapter_id = info[1]
        prompt_id = info[2]
        conversation_id = info[3]
        turn_id = info[4]

        adapter_key = get_adapter_node_key(adapter_id)
        prompt_key = get_system_prompt_node_key(
            adapter_id=adapter_id, system_prompt_id=prompt_id
        )
        adapter_size = size_dict[CacheType.lora_adapter]
        system_prompt_size = size_dict[CacheType.system_prompt]
        multi_turn_qa_size = (
            size_dict[CacheType.multi_turn_q] + size_dict[CacheType.multi_turn_a]
        )

        sieve_total_num += adapter_size
        sieve_adapter_total_num += adapter_size
        res = cache.get(adapter_key)
        if res == -1:
            cache.insert(adapter_key, adapter_size)
        else:
            sieve_hit_num += adapter_size
            sieve_adapter_hit_num += adapter_size

        sieve_total_num += system_prompt_size + multi_turn_qa_size * (turn_id + 1)
        sieve_prefix_total_num += system_prompt_size + multi_turn_qa_size * (
            turn_id + 1
        )
        is_sys_prompt_exist = True
        insert_list = []

        for i in range(turn_id, -1, -1):
            multi_turn_qa_key = get_multi_turn_node_key(
                adapter_id=adapter_id,
                system_prompt_id=prompt_id,
                conversation_id=conversation_id,
                turn_id=i,
            )
            res = cache.get(multi_turn_qa_key)
            if res != -1:
                sieve_hit_num += multi_turn_qa_size
                sieve_prefix_hit_num += multi_turn_qa_size
            else:
                insert_list.append(multi_turn_qa_key)
        res = cache.get(prompt_key)
        if res != -1:
            sieve_hit_num += system_prompt_size
            sieve_prefix_hit_num += system_prompt_size
        else:
            is_sys_prompt_exist = False

        for key in insert_list:
            cache.insert(key, multi_turn_qa_size)
        if not is_sys_prompt_exist:
            cache.insert(prompt_key, system_prompt_size)

        # print(cache.queue)
        # print(sieve_hit_num,sieve_total_num)

    return (
        sieve_hit_num / sieve_total_num,
        sieve_adapter_hit_num / sieve_adapter_total_num,
        sieve_prefix_hit_num / sieve_prefix_total_num,
    )


def multi_turn_benchmark():
    adapter_num = 64
    alpha = 1.0
    system_prompt_num = 8
    cache_size = 20000
    multi_turn_num = 8
    think_time = 0

    work_set_size = (
        adapter_num * size_dict[CacheType.lora_adapter]
        + system_prompt_num * adapter_num * size_dict[CacheType.system_prompt]
        + system_prompt_num
        * adapter_num
        * multi_turn_num
        * (size_dict[CacheType.multi_turn_q] + size_dict[CacheType.multi_turn_a])
    )
    think_time_list = [0, 5, 15, 30]
    for think_time in think_time_list:
        cache_ratio_list = [5, 10, 20, 30]
        for cache_ratio in cache_ratio_list:
            cache_size = int(work_set_size * cache_ratio / 100)

            print(work_set_size, cache_size)

            trace = generate_multi_turn_trace(
                adapter_num, alpha, system_prompt_num, multi_turn_num, think_time
            )

            lfu_hit, lfu_adapter, lfu_prefix = multi_turn_lfu_test(trace, cache_size)
            lru_hit, lru_adapter, lru_prefix = multi_turn_lru_test(trace, cache_size)
            sieve_hit, sieve_adapter, sieve_prefix = multi_turn_sieve_test(
                trace, cache_size
            )

            print("lru: ", lru_hit, lru_adapter, lru_prefix)
            print("lfu: ", lfu_hit, lfu_adapter, lfu_prefix)
            print("sieve: ", sieve_hit, sieve_adapter, sieve_prefix)
            print("------------------------")
        print("<<<<<<<<<<<<<<<<<<<<<<<<")


if __name__ == "__main__":
    # system_prompts_benchmark()
    multi_turn_benchmark()
    print("------------------------")
    # system_prompts_benchmark()
