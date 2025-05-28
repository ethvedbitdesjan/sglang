import time
from collections import defaultdict
from typing import Dict

from sglang.srt.lora.utils import AdapterInfo
from sglang.srt.mem_cache.radix_cache import TreeNode


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

        self.id = TreeNode.counter
        TreeNode.counter += 1

    @property
    def is_empty(self):
        return self.value is None

    def __repr__(self):
        return (
            f"AdapterNode("
            f"name={self.adapter_name!r}, "
            f"id={self.id}, "
            f"size={self.value.size}, "
            f"is_empty={self.is_empty}, "
            f"is_lock={self.lock_ref > 0}, "
            f"children={len(self.children)}"
            f")"
        )

    # @property
    # def evicted(self):
    #     return self.value is None

    # @property
    # def backuped(self):
    #     return self.host_value is not None
