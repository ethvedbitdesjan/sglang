import torch
import logging
import numpy as np
import math
from typing import Dict, Tuple, Optional, Set, List, Union
from dataclasses import dataclass
from sglang.srt.lora.utils import LoRAType, get_weight_name, get_stacked_multiply, get_layer_id
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import debug_timing, get_compiler_backend
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

# Constants
GB = 1024 * 1024 * 1024

############################################
# Data classes for block and adapter tracking
############################################
def get_projection_index(weight_name: str) -> Optional[int]:
    mapping = {
        "q_proj": 0,
        "k_proj": 1,
        "v_proj": 2,
        "o_proj": 3,
        "qkv_proj": 0,
        "kv_proj": 1,
        "gate_up_proj": 4,
        "down_proj": 5,
    }
    return mapping.get(weight_name, None)

@dataclass
class AdapterInfo:
    """Tracks information about a loaded LoRA adapter."""
    rank: int
    loc: torch.Tensor  # location indices in the unified memory pool
    size: int          # Number of cells allocated
    last_used: int     # For LRU tracking

@dataclass
class LoraMHAConfig:
    """Configuration for Multi-Head Attention"""
    attn_head_num: int
    kv_head_num: int
    head_dim: int

@dataclass
class LoraMLAConfig:
    """Configuration for Multi-head Latent Attention"""
    kv_lora_rank: int
    qk_rope_head_dim: int
    head_dim: int
    num_hidden_layers: int

############################################
# AttentionType Enum
############################################

class AttentionType:
    MHA = "mha"  # Multi-Head Attention
    MLA = "mla"  # Multi-head Latent Attention

############################################
# LoraMHATokenToKVPool Implementation
############################################

class LoraMHATokenToKVPool:
    def __init__(
        self,
        total_size: int,
        unified_k_buffer: List[torch.Tensor],
        unified_v_buffer: List[torch.Tensor],
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ):
        self.total_size = total_size
        self.dtype = dtype
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
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
        k_size_bytes = sum(np.prod(k_cache.shape) * k_cache.dtype.itemsize for k_cache in self.unified_k_buffer)
        v_size_bytes = sum(np.prod(v_cache.shape) * v_cache.dtype.itemsize for v_cache in self.unified_v_buffer)
        return k_size_bytes, v_size_bytes

    def get_flat_data(self, indices):
        flatten = torch.stack([
            torch.stack([self.unified_k_buffer[i][indices] for i in range(self.layer_num)]),
            torch.stack([self.unified_v_buffer[i][indices] for i in range(self.layer_num)]),
        ])
        return flatten

    @debug_timing
    def transfer(self, indices, flat_data):
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
            self.unified_k_buffer[layer_id][loc].copy_(cache_k.view(self.store_dtype))
            self.unified_v_buffer[layer_id][loc].copy_(cache_v.view(self.store_dtype))
        else:
            self.unified_k_buffer[layer_id][loc].copy_(cache_k)
            self.unified_v_buffer[layer_id][loc].copy_(cache_v)

############################################
# LoraMLATokenToKVPool Implementation
############################################

class LoraMLATokenToKVPool:
    def __init__(
        self,
        total_size: int,
        unified_kv_buffer: List[torch.Tensor],
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
    ):
        self.total_size = total_size
        self.dtype = dtype
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device

        self.unified_kv_buffer = unified_kv_buffer
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.layer_num = layer_num

        kv_size = self.get_kv_size_bytes()
        logger.info(f"KV Cache is allocated. KV size: {kv_size / GB:.2f} GB.")

    def get_kv_size_bytes(self):
        return sum(np.prod(kv_cache.shape) * kv_cache.dtype.itemsize for kv_cache in self.unified_kv_buffer)

    def get_flat_data(self, indices):
        return torch.stack([self.unified_kv_buffer[i][indices] for i in range(self.layer_num)])

    @debug_timing
    def transfer(self, indices, flat_data):
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        for i in range(self.layer_num):
            self.unified_kv_buffer[i][indices] = flat_data[i]

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.unified_kv_buffer[layer_id].view(self.dtype)
        return self.unified_kv_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        buffer = self.get_key_buffer(layer_id)
        return buffer[..., :self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        buffer = self.get_key_buffer(layer_id)
        return buffer, buffer[..., :self.kv_lora_rank]

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
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.unified_kv_buffer[layer_id][loc].copy_(cache_k.view(self.store_dtype))
        else:
            self.unified_kv_buffer[layer_id][loc].copy_(cache_k)

############################################
# LoraUnifiedMemoryPool Implementation
############################################

class LoraUnifiedMemoryPool:
    """
    A unified memory pool
    """
    def __init__(
        self,
        total_size: int,
        dtype: torch.dtype,
        device: str,
        layer_num: int,
        attention_type: str,
        attention_config: Union[LoraMHAConfig, LoraMLAConfig],
        enable_memory_saver: bool,
    ):
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(enable=enable_memory_saver)
        self.total_size = total_size
        self.dtype = dtype
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device
        self.layer_num = layer_num
        self.attention_type = attention_type
        self.attention_config = attention_config

        with self.memory_saver_adapter.region():
            if self.attention_type == AttentionType.MHA:
                self.unified_k_buffer = [
                    torch.empty((total_size + 1, attention_config.kv_head_num, attention_config.head_dim),
                                dtype=self.store_dtype, device=device)
                    for _ in range(layer_num)
                ]
                self.unified_v_buffer = [
                    torch.empty((total_size + 1, attention_config.kv_head_num, attention_config.head_dim),
                                dtype=self.store_dtype, device=device)
                    for _ in range(layer_num)
                ]
                self.token_to_kv_pool = LoraMHATokenToKVPool(
                    total_size=total_size,
                    unified_k_buffer=self.unified_k_buffer,
                    unified_v_buffer=self.unified_v_buffer,
                    dtype=dtype,
                    head_num=attention_config.kv_head_num,
                    head_dim=attention_config.head_dim,
                    layer_num=layer_num,
                    device=device,
                )
            elif self.attention_type == AttentionType.MLA:
                self.unified_kv_buffer = [
                    torch.empty((total_size + 1, 1, attention_config.kv_lora_rank + attention_config.qk_rope_head_dim),
                                dtype=self.store_dtype, device=device)
                    for _ in range(layer_num)
                ]
                self.token_to_kv_pool = LoraMLATokenToKVPool(
                    total_size=total_size,
                    unified_kv_buffer=self.unified_kv_buffer,
                    dtype=dtype,
                    kv_lora_rank=attention_config.kv_lora_rank,
                    qk_rope_head_dim=attention_config.qk_rope_head_dim,
                    layer_num=layer_num,
                    device=device,
                )
            else:
                raise ValueError(f"Unsupported attention type: {self.attention_type}")

        self.active_adapters: Dict[str, AdapterInfo] = {}
        self.cur_adapters: Dict[str, AdapterInfo] = {}
        self.access_counter = 0
        self.lora_weight_names: Set[Tuple[str, ...]] = set()
        self.adapter_to_idx: Dict[str, int] = {}
        self.idx_to_adapter: Dict[int, str] = {}
        self.free_slots = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

    # --- Core Memory Allocation Methods ---
    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int):
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index.to(self.device, non_blocking=True)

    def alloc_contiguous(self, need_size: int):        
        if need_size > len(self.free_slots):
            return None
        
        # Sort the free slots to find contiguous blocks
        sorted_slots, _ = torch.sort(self.free_slots)
        
        if sorted_slots.numel() == 1:
            if need_size == 1:
                select_index = sorted_slots
                self.free_slots = torch.tensor([], dtype=torch.int32)
                return select_index.to(self.device)
            else:
                return None
        
        # Calculate the differences between consecutive elements
        diff = sorted_slots[1:] - sorted_slots[:-1]
        # Find positions where the difference is not 1 (indicating a break in contiguous blocks)
        break_positions = torch.where(diff != 1)[0]
        
        # Generate start and end indices for each contiguous segment
        starts = torch.cat([
            torch.tensor([0]),
            break_positions + 1  # Next segment starts after the break
        ])
        ends = torch.cat([
            break_positions + 1,  # End before the break
            torch.tensor([len(sorted_slots)])
        ])
        
        # Iterate through each segment to find the first one with sufficient length
        for i in range(len(starts)):
            start_idx = starts[i].item()
            end_idx = ends[i].item()
            segment_length = end_idx - start_idx
            if segment_length >= need_size:
                # Extract the contiguous block
                select_index = sorted_slots[start_idx : start_idx + need_size]
                # Update remaining free slots by removing the allocated block
                remaining = torch.cat([
                    sorted_slots[:start_idx],
                    sorted_slots[start_idx + need_size:]
                ])
                self.free_slots = remaining
                return select_index.to(device=self.device, non_blocking=True)
        
        # No contiguous block of sufficient size found
        return None

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
        self.free_slots = torch.arange(1, self.total_size + 1, dtype=torch.int32)
        self.is_in_free_group = False
        self.free_group = []
        self.active_adapters = {}
        self.access_counter = 0
        self.adapter_to_idx = {}
        self.idx_to_adapter = {}

    # --- KV Cache Interface ---
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

    def get_kv_size_bytes(self) -> Union[Tuple[int, int], int]:
        return self.token_to_kv_pool.get_kv_size_bytes()

    def get_flat_data(self, indices: torch.Tensor) -> torch.Tensor:
        return self.token_to_kv_pool.get_flat_data(indices)

    def transfer(self, indices: torch.Tensor, flat_data: torch.Tensor):
        self.token_to_kv_pool.transfer(indices, flat_data)

    # --- LoRA Adapter Methods ---
    def alloc_lora_adapter(self, adapter_id: str, rank: int) -> bool:
        if self.attention_type == AttentionType.MHA:
            head_ratio = self.attention_config.attn_head_num / self.attention_config.kv_head_num
            required_size = math.ceil(rank * (4+2) * head_ratio)
        elif self.attention_type == AttentionType.MLA:
            total_dims = self.attention_config.kv_lora_rank + self.attention_config.qk_rope_head_dim
            required_size = math.ceil(rank * (4+2) * (total_dims / self.attention_config.head_dim))
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")

        adapter_loc = self.alloc(required_size)
        if adapter_loc is None:
            #raise error, no need to evict
            success = self._evict_lru_adapters(required_size)
            if not success:
                return False
            adapter_loc = self.alloc(required_size)
            if adapter_loc is None:
                return False

        self.active_adapters[adapter_id] = AdapterInfo(
            rank=rank,
            loc=adapter_loc,
            size=required_size,
            last_used=self.access_counter
        )
        idx = len(self.adapter_to_idx)
        self.adapter_to_idx[adapter_id] = idx
        self.idx_to_adapter[idx] = adapter_id
        self.access_counter += 1
        return True

    def _evict_lru_adapters(self, required_size: int) -> bool:
        available = len(self.free_slots)
        # Compute how many additional slots are needed.
        need_to_free = max(0, required_size - available)
        if need_to_free == 0:
            return True

        if not self.active_adapters:
            return False

        # Sort active adapters by last_used (LRU order).
        sorted_adapters = sorted(self.active_adapters.items(), key=lambda x: x[1].last_used)
        freed_size = 0
        evicted_adapters = []
        for adapter_id, info in sorted_adapters:
            self.free(info.loc)
            freed_size += info.size
            evicted_adapters.append(adapter_id)
            if freed_size >= need_to_free:
                break

        for adapter_id in evicted_adapters:
            del self.active_adapters[adapter_id]
            idx = self.adapter_to_idx.pop(adapter_id, None)
            if idx is not None:
                del self.idx_to_adapter[idx]
        self._reindex_adapters()

        new_available = len(self.free_slots)
        return new_available >= required_size

    def _reindex_adapters(self):
        self.adapter_to_idx = {}
        self.idx_to_adapter = {}
        for i, adapter_id in enumerate(self.active_adapters.keys()):
            self.adapter_to_idx[adapter_id] = i
            self.idx_to_adapter[i] = adapter_id

    def free_lora_adapter(self, adapter_id: str):
        if adapter_id in self.active_adapters:
            self.free(self.active_adapters[adapter_id].loc)
            del self.active_adapters[adapter_id]
            idx = self.adapter_to_idx.pop(adapter_id, None)
            if idx is not None:
                del self.idx_to_adapter[idx]
            self._reindex_adapters()

    def init_buffers(self, lora_weight_names: Set[Tuple[str, ...]], base_model: torch.nn.Module):
        self.lora_weight_names = lora_weight_names

    def prepare_lora_batch(self, cur_uids: Set[Optional[str]], lora_adapters: Dict[str, LoRAAdapter]):
        self.cur_adapters = {}
        for uid in cur_uids:
            if uid is None:
                continue
            if uid in self.active_adapters:
                self.active_adapters[uid].last_used = self.access_counter
                self.cur_adapters[uid] = self.active_adapters[uid]
                self.access_counter += 1
                continue
            adapter = lora_adapters.get(uid, None)
            if adapter is None:
                continue
            success = self.alloc_lora_adapter(uid, adapter.r)
            if not success:
                raise ValueError(f"Cannot allocate memory for adapter {uid}")
            self.load_lora_weight_to_buffer(uid, lora_adapter=adapter)
            self.cur_adapters[uid] = self.active_adapters[uid]        

    def load_lora_weight_to_buffer(self, uid: str, lora_adapter: Optional[LoRAAdapter] = None):
        if uid is None or lora_adapter is None:
            return
    
        if uid not in self.active_adapters:
            raise ValueError(f"Adapter {uid} not allocated in the memory pool")
        info = self.active_adapters[uid]
        for layer_id in range(self.layer_num):
            layer_weights = lora_adapter.layers[layer_id].weights
        # Route to the appropriate implementation based on attention type
            if self.attention_type == AttentionType.MHA:
                self._load_weights_mha(layer_id, info.loc, info.rank, layer_weights)
            elif self.attention_type == AttentionType.MLA:
                raise NotImplementedError("MLA attention type not yet implemented")
            else:
                raise ValueError(f"Unsupported attention type: {self.attention_type}")
            

    def _load_weights_mha(self, layer_id: int, info_loc: torch.Tensor, rank: int, layer_weights: Dict[str, torch.Tensor]):
        # Process lora_A weights
        head_ratio = self.attention_config.attn_head_num / self.attention_config.kv_head_num
        
        for name, weights in layer_weights.items():
            if "lora_A" in name:
                weight_name = get_weight_name(name, self.lora_weight_names, LoRAType.LORA_A)
                if weight_name is None:
                    continue
                proj_index = get_projection_index(weight_name)
                if proj_index is None:
                    continue
                segment_length = math.ceil(head_ratio * rank)
                offset = proj_index * segment_length
                selected_indices = info_loc[offset : offset + segment_length]
                # If necessary, reshape weights if the inner dimension does not match
                if weights.shape[1] != self.attention_config.kv_head_num:
                    assert math.ceil(rank * weights.shape[1]/self.attention_config.kv_head_num) == segment_length
                    weights = weights.reshape(segment_length, self.attention_config.kv_head_num, self.attention_config.head_dim)
                for j in range(len(selected_indices)):
                    self.unified_k_buffer[layer_id][selected_indices[j]].copy_(weights[j])
            # elif 'kv' in name:
            #     #pass TODO: only fill up rank cells, not sure
            #     weight_name = get_weight_name(name, self.lora_weight_names, LoRAType.LORA_B)
            #     c = get_stacked_multiply(weight_name)
            #     segment_length = head_ratio * rank
                
            #     k_offset = 1 * head_ratio * rank
            #     k_indices = info_loc[k_offset : k_offset + rank]
            #     v_offset = 2 * head_ratio * rank
            #     v_indices = info_loc[v_offset : v_offset + rank]
                
            #     for j in range(k_indices):
            #         self.unified_v_buffer[layer_id][k_indices[j]].copy_(weights[0][j])
            #     for j in range(v_indices):
            #         self.unified_v_buffer[layer_id][v_indices[j]].copy_(weights[1][j])
                
            #     dummy_k_indices = info_loc[k_offset + rank : k_offset + segment_length]
            #     dummy_v_indices = info_loc[v_offset + rank : v_offset + segment_length]
                
            #     for j in range(len(dummy_k_indices)):
            #         self.unified_v_buffer[layer_id][dummy_k_indices[j]].zero_()

            #     for j in range(len(dummy_v_indices)):
            #         self.unified_v_buffer[layer_id][dummy_v_indices[j]].zero_()
                                    
            else:
                weight_name = get_weight_name(name, self.lora_weight_names, LoRAType.LORA_B)
                if weight_name is None:
                    continue
                proj_index = get_projection_index(weight_name)
                if proj_index is None:
                    continue
                c = get_stacked_multiply(weight_name)
                segment_length = math.ceil(head_ratio * rank)
                offset = proj_index * segment_length
                if c > 1:
                    if weights.shape[2] != self.attention_config.kv_head_num:
                        assert math.ceil(rank * weights.shape[2] / self.attention_config.kv_head_num) == segment_length
                        weights = weights.reshape(c, segment_length, self.attention_config.kv_head_num, self.attention_config.head_dim)
                    for stacked_id in range(c):
                        selected_indices = info_loc[offset:offset + segment_length]
                        for j in range(len(selected_indices)):
                            self.unified_v_buffer[layer_id][selected_indices[j]].copy_(weights[stacked_id][j])
                else:
                    if weights.shape[1] != self.attention_config.kv_head_num:
                        assert math.ceil(rank * weights.shape[1] / self.attention_config.kv_head_num) == segment_length
                        weights = weights.reshape(segment_length, self.attention_config.kv_head_num, 
                                            self.attention_config.head_dim)
                        
                    selected_indices = info_loc[offset : offset + segment_length]
                    for j in range(len(selected_indices)):
                        self.unified_v_buffer[layer_id][selected_indices[j]].copy_(weights[j])


    def _load_weights_mla(self, layer_id: int, loc: torch.Tensor, rank: int, layer_weights: Dict[str, torch.Tensor]):
        raise NotImplementedError("MLA attention type not yet implemented")


    def get_adapter_memory_info(self, adapter_cache_type: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.cur_adapters:
            # Return empty tensors when no adapters are present
            empty = torch.tensor([], dtype=torch.long, device=self.device)
            return empty, empty, empty

        if self.attention_type != AttentionType.MHA:
            raise NotImplementedError("Only MHA is currently supported")
        
        head_ratio = self.attention_config.attn_head_num / self.attention_config.kv_head_num
        locs = []
        starts = []
        lens = []
        if adapter_cache_type == "qkvo":
            for adapter_id, info in self.cur_adapters.items():
                rank = info.rank
                segment_length = int(math.ceil(head_ratio * rank))
                for i in range(4):
                    offset = i * segment_length
                    end_offset = offset + segment_length
                    
                    # Get location indices
                    if end_offset <= len(info.loc):
                        adapter_loc = info.loc[offset:end_offset]
                        
                        locs.append(adapter_loc)
                        starts.append(torch.zeros(1, dtype=torch.long, device=self.device))
                        
                        # Always return head_ratio * rank for consistency even though for lora_B we only use the first rank cells
                        lens.append(torch.tensor([segment_length], dtype=torch.long, device=self.device))
        elif adapter_cache_type == "gate_up":
            pass
        elif adapter_cache_type == "down":
            pass
        else:
            raise ValueError(f"Unsupported adapter cache type: {adapter_cache_type}")
        
        return (
            torch.cat(locs) if locs else torch.tensor([], dtype=torch.long, device=self.device),
            torch.cat(starts) if starts else torch.tensor([], dtype=torch.long, device=self.device),
            torch.cat(lens) if lens else torch.tensor([], dtype=torch.long, device=self.device)
        )

    # def get_buffer_id(self, adapter_id: Optional[str]) -> int:
    #     if adapter_id is None:
    #         return -1
    #     if adapter_id not in self.adapter_to_idx:
    #         raise ValueError(f"Adapter {adapter_id} not loaded")
    #     return self.adapter_to_idx[adapter_id]

    # def get_tensor(
    #     self, weight_name: str, layer_id: int, lora_type: LoRAType
    # ) -> torch.Tensor:
    #     loc = self.get_adapter_memory_info(weight_name, layer_id, lora_type)
    #     if lora_type == LoRAType.LORA_A:
    #         if self.attention_type == AttentionType.MHA:
    #             return self.unified_k_buffer[layer_id][loc]
    #         else:
    #             return self.unified_kv_buffer[layer_id][loc]
    #     else:
    #         if self.attention_type == AttentionType.MHA:
    #             return self.unified_v_buffer[layer_id][loc]
    #         else:
    #             return self.unified_kv_buffer[layer_id][loc]