import torch
import logging
import numpy as np
from typing import Dict, Tuple, Optional, Set
from dataclasses import dataclass
from sglang.srt.lora.utils import LoRAType, get_weight_name, get_stacked_multiply
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.layers.radix_attention import RadixAttention
from typing import List, Optional, Tuple, Union
from sglang.srt.utils import debug_timing, get_compiler_backend

from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter


logger = logging.getLogger(__name__)

############################################
# Data classes for block and adapter tracking
############################################

@dataclass
class AdapterInfo:
    """Tracks information about a loaded LoRA adapter."""
    rank: int
    
    loc : torch.Tensor  # location indices in the unified memory pool

    size: int        # Number of cells allocated (typically rank * 4)
    last_used: int   # For LRU tracking

@dataclass
class LoraMHAConfig:
    attn_head_num: int
    kv_head_num: int
    head_dim: int

############################################
# LoraMHATokenToKVPool Implementation
############################################

class LoraMHATokenToKVPool:
    def __init__(
        self,
        total_size: int,
        unified_k_buffer : List[torch.Tensor],
        unified_v_buffer : List[torch.Tensor],
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ):
        self.total_size = total_size
        self.dtype = dtype
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device

        self.unified_k_buffer = unified_k_buffer
        self.unified_v_buffer = unified_v_buffer

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB."
        )

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.unified_k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.unified_v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    # Todo: different memory layout
    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        flatten = torch.stack(
            [
                torch.stack([self.unified_k_buffer[i][indices] for i in range(self.layer_num)]),
                torch.stack([self.unified_v_buffer[i][indices] for i in range(self.layer_num)]),
            ]
        )
        return flatten

    @debug_timing
    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        for i in range(self.layer_num):
            self.unified_k_buffer[i][indices] = k_data[i]
            self.unified_v_buffer[i][indices] = v_data[i]

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.unified_k_buffer[layer_id].view(self.dtype)
        return self.unified_k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.unified_v_buffer[layer_id].view(self.dtype)
        return self.unified_v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.unified_k_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
            self.unified_v_buffer[layer_id][loc] = cache_v.view(self.store_dtype)
        else:
            self.unified_k_buffer[layer_id][loc] = cache_k
            self.unified_v_buffer[layer_id][loc] = cache_v


############################################
# LoraUnifiedMemoryPool Implementation
############################################

class LoraUnifiedMemoryPool:

    #TODO: please fix the description
    """
    A unified memory pool that implements SGLang's expected interfaces while
    providing unified paging for both KV cache and LoRA adapter weights.
    
    Rather than partitioning memory into fixed regions, this pool uses a block-based 
    allocator over the entire pool. All cells are managed dynamically and can be 
    allocated to either KV cache usage (ephemeral) or LoRA adapters (longer-lived).
    
    The pool exposes:
      - Generic allocation: alloc_unified_block, free_unified_block, clear, free_group_begin/end.
      - BaseTokenToKVPool interface: available_size_method(), alloc(need_size), get_key_buffer(), 
        get_value_buffer(), get_kv_buffer(), set_kv_buffer().
      - MHATokenToKVPool extras: get_kv_size_bytes(), get_flat_data(), transfer().
      - LoRAMemoryPool extras: alloc_lora_adapter(), free_lora_adapter(), init_buffers(), 
        prepare_lora_batch(), load_lora_weight_to_buffer(), get_tensor(), get_buffer_id(), 
        get_adapter_weights(), etc.
    """
    def __init__(
        self,
        total_size: int,
        dtype: torch.dtype,
        device: str,
        layer_num: int,


        # TODO: please add selection according to the type of attention(e.g., MHA, MLA, etc.), I just set MHA as an example.
        lora_mha_config : LoraMHAConfig,



        enable_memory_saver: bool,
    ):
        # Basic configuration
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.total_size = total_size
        self.dtype = dtype
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device


        with self.memory_saver_adapter.region():
            # [size, H] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            
            # Create unified key and value buffers for each layer.

            # TODO: please add selection according to the type of attention(e.g., MHA, MLA, etc.) too, I just set 
            self.unified_k_buffer = [
                torch.empty((total_size + 1, lora_mha_config.kv_head_num, lora_mha_config.head_dim), dtype=self.store_dtype, device=device)
                for _ in range(layer_num)
            ]
            self.unified_v_buffer = [
                torch.empty((total_size + 1, lora_mha_config.kv_head_num, lora_mha_config.head_dim), dtype=self.store_dtype, device=device)
                for _ in range(layer_num)
            ]

        # TODO: please add selection according to the type of attention(e.g., MHA, MLA, etc.), I just set MHA as an example.
        self.token_to_kv_pool = LoraMHATokenToKVPool(
            total_size=total_size,
            unified_k_buffer=self.unified_k_buffer,
            unified_v_buffer=self.unified_v_buffer,
            dtype=dtype,
            head_num=lora_mha_config.kv_head_num,
            head_dim=lora_mha_config.head_dim,
            layer_num=layer_num,
            device=device,
        )

        # For LoRA adapter tracking:

        # active_adapters: adapter_id -> AdapterInfo (for LRU tracking)
        self.active_adapters: Dict[str, AdapterInfo] = {}
        self.access_counter = 0
        # For LoRA manager: record weight names (set of tuples)
        self.lora_weight_names: Set[Tuple[str, ...]] = set()



        self.free_slots = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

    # ------------------ Core Unified Paging Memory Allocation ------------------

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int):
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index.to(self.device, non_blocking=True)

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            self.free_slots = torch.concat((self.free_slots, free_index.cpu()))
        else:
            self.free_group.append(free_index)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.concat(self.free_group))

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_slots = torch.arange(1, self.total_size + 1, dtype=torch.int32)
        self.is_in_free_group = False
        self.free_group = []

    # ---------------- BaseTokenToKVPool Interface (KV cache) ----------------

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.token_to_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.token_to_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.token_to_kv_pool.get_kv_buffer(layer_id)
    
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None
    ) -> None:
        self.token_to_kv_pool.set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale)
    # ---------------- Additional MHATokenToKVPool Methods ----------------

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return self.token_to_kv_pool.get_kv_size_bytes()

    def get_flat_data(self, indices: torch.Tensor) -> torch.Tensor:
        return self.token_to_kv_pool.get_flat_data(indices)

    def transfer(self, indices: torch.Tensor, flat_data: torch.Tensor):
        self.token_to_kv_pool.transfer(indices, flat_data)

    # ---------------- LoRA Adapter Methods ----------------

    def alloc_lora_adapter(self, adapter_id: str, rank: int) -> bool:
        """
        Allocate memory for a LoRA adapter with a given rank.
        Required size = rank * 4 cells.
        """

        # TODO: this is wrong, because for LLaMa-3 model, kv_head_num is not equal to attn_head_num,please refer to our doc.
        # So we need to multiply attn_head_num/kv_head_num to the required_size for MHA attention.
        # you also need to use 'if' to check the type of attention and do the corresponding calculation.
        # we can just support MHA attention for now.

        required_size = rank * 4
        adapter_loc = self.alloc(required_size)
        # TODO: add check for adapter_loc is None

        self.active_adapters[adapter_id] = AdapterInfo(
            rank=rank,
            loc=adapter_loc,
            size=required_size,
            last_used=self.access_counter
        )
        self.access_counter += 1
        return True

    def free_lora_adapter(self, adapter_id: str):
        if adapter_id in self.active_adapters:
            self.free(self.active_adapters[adapter_id].loc)
            del self.active_adapters[adapter_id]

    def init_buffers(self, lora_weight_names: Set[Tuple[str, ...]], base_model: torch.nn.Module):
        """
        Record the set of weight name tuples that are targeted for LoRA.
        (No new buffer allocation is needed.)
        """
        self.lora_weight_names = lora_weight_names

    def prepare_lora_batch(self, cur_uids: Set[Optional[str]], lora_adapters: Dict[str, LoRAAdapter]):
        """
        For each adapter UID in cur_uids (if not already loaded), allocate a block and load its weights.
        """
        for uid in cur_uids:
            if uid is None:
                continue
            if uid not in self.active_adapters:
                adapter = lora_adapters.get(uid, None)
                if adapter is None:
                    continue
                success = self.alloc_lora_adapter(uid, adapter.r)
                if not success:
                    raise ValueError(f"Cannot allocate memory for adapter {uid}")
                self.load_lora_weight_to_buffer(uid, lora_adapter=adapter)

    # TODO: please fix this function
    def load_lora_weight_to_buffer(self, uid: str, lora_adapter: Optional[LoRAAdapter] = None):
        """
        We assume each layer of the adapter has a 'weights' dict.
        For each key, use get_weight_name() and get_stacked_multiply() (imported from sglang.srt.lora.utils)
        to decide whether it is a LoRA_A weight or a LoRA_B weight.
        """
        if uid is None:
            return
        if lora_adapter is None:
            raise ValueError(f"lora_adapter must be provided for uid {uid}")
        if uid not in self.adapter_to_block:
            raise ValueError(f"Adapter {uid} not allocated")
        block_id = self.adapter_to_block[uid]
        info = self.allocated_blocks[block_id]
        start = info.start
        size = info.size
        half = size // 2  # first half for LoRA_A, second half for LoRA_B
        for layer_id in range(self.layer_num):
            # Get the weight dictionary for this layer.
            layer_weights = lora_adapter.layers[layer_id].weights
            # Initialize temporary tensors for this adapter's A and B weights.
            # We assume that the expected length for each side is given by half divided by (4/rank) ?
            # In the original design, the adapter block size is rank*4.
            # For simplicity, we assume that the adapter provides weights with shape (rank, ...) for each matrix.
            # Here we determine L from the weight shape of one of the matrices.
            L = None
            for name, weight in layer_weights.items():
                if "lora_A" in name:
                    L = weight.shape[0]
                    break
            if L is None:
                raise ValueError("No lora_A weights found for adapter " + uid)
            # Now load weights.
            for name, weight in layer_weights.items():
                if "lora_A" in name:
                    lora_weight_name = get_weight_name(name, self.lora_weight_names, LoRAType.LORA_A)
                    if lora_weight_name:
                        # Copy weight into key buffer starting at 'start'
                        self.key_buffer[layer_id][start: start + L].copy_(weight)
                else:
                    lora_weight_name = get_weight_name(name, self.lora_weight_names, LoRAType.LORA_B)
                    if lora_weight_name:
                        c = get_stacked_multiply(lora_weight_name)
                        # If there are stacked components, assume weight is indexable by [stack_id]
                        if c > 1:
                            for stacked_id in range(c):
                                self.value_buffer[layer_id][start + L * stacked_id: start + L * (stacked_id + 1)].copy_(weight[stacked_id])
                        else:
                            self.value_buffer[layer_id][start: start + L].copy_(weight)

    # TODO: please fix this function, we need to take weight_name(q,k,v,o) into account.
    # return parameter: adapter_loc, buffer.
    def get_adapter_memory_info(
        self, weight_name: str, layer_id: int, lora_type: LoRAType
    ) -> torch.Tensor:

        # if lora_type == LoRAType.LORA_A:
        #     return self.A_buffer[weight_name][layer_id]

        # return self.B_buffer[weight_name][layer_id]
        1

    # def get_tensor(self, adapter_id: str, layer_id: int, lora_type: int) -> torch.Tensor:
    #     """
    #     Given an adapter identifier, return the corresponding weight slice
    #     for the specified layer and LoRA type.
    #     """
    #     if adapter_id not in self.active_adapters:
    #         raise ValueError(f"Adapter {adapter_id} not loaded")
    #     A, B = self.get_adapter_weights(adapter_id, layer_id)
    #     return A if lora_type == LoRAType.LORA_A else B


    # def get_adapter_weights(self, adapter_id: str, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Return the weight slices for a LoRA adapter at a given layer.
    #     The allocated block (of size rank * 4) is split into two halves:
    #       - The first half for matrix A (lora_A)
    #       - The second half for matrix B (lora_B)
    #     """
    #     if adapter_id not in self.active_adapters:
    #         raise ValueError(f"Adapter {adapter_id} is not active")
    #     info = self.active_adapters[adapter_id]
    #     start = info.start_idx
    #     half = info.size // 2
    #     info.last_used = self.access_counter
    #     self.access_counter += 1
    #     A = self.key_buffer[layer_id][start: start + half]
    #     B = self.value_buffer[layer_id][start + half: start + info.size]
    #     return A, B

    # def get_tensor_from_adapter(self, adapter_id: str, layer_id: int, lora_type: int) -> torch.Tensor:
    #     """Alias for get_tensor(adapter_id, layer_id, lora_type)."""
    #     return self.get_tensor(adapter_id, layer_id, lora_type)