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
            required_size = math.ceil(rank * 4 * head_ratio)
        elif self.attention_type == AttentionType.MLA:
            total_dims = self.attention_config.kv_lora_rank + self.attention_config.qk_rope_head_dim
            required_size = math.ceil(rank * 4 * (total_dims / self.attention_config.head_dim))
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")

        adapter_loc = self.alloc(required_size)
        if adapter_loc is None:
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
        for uid in cur_uids:
            if uid is None:
                continue
            if uid in self.active_adapters:
                self.active_adapters[uid].last_used = self.access_counter
                self.access_counter += 1
                continue
            adapter = lora_adapters.get(uid, None)
            if adapter is None:
                continue
            success = self.alloc_lora_adapter(uid, adapter.r)
            if not success:
                raise ValueError(f"Cannot allocate memory for adapter {uid}")
            self.load_lora_weight_to_buffer(uid, lora_adapter=adapter)

    def load_lora_weight_to_buffer(self, uid: str, lora_adapter: Optional[LoRAAdapter] = None):
        if uid is None:
            return
        if lora_adapter is None:
            raise ValueError(f"lora_adapter must be provided for uid {uid}")
        if uid not in self.active_adapters:
            raise ValueError(f"Adapter {uid} not allocated")
        info = self.active_adapters[uid]
        rank = lora_adapter.r
        for layer_id in range(self.layer_num):
            layer_weights = lora_adapter.layers[layer_id].weights
            logger.info(f"Loading weights for adapter {uid} layer {layer_id}")
            if self.attention_type == AttentionType.MHA:
                self._load_weights_mha(layer_id, info.loc, rank, layer_weights)
            elif self.attention_type == AttentionType.MLA:
                self._load_weights_mla(layer_id, info.loc, rank, layer_weights)
            else:
                raise ValueError(f"Unsupported attention type: {self.attention_type}")

    def _load_weights_mha(self, layer_id: int, loc: torch.Tensor, rank: int, layer_weights: Dict[str, torch.Tensor]):
        """
        Load MHA (Multi-Head Attention) LoRA weights into the unified buffer.
        
        Args:
            layer_id: The layer ID to load weights for
            loc: Location indices in the unified buffer
            rank: The LoRA rank
            layer_weights: Dictionary of layer weights
        """
        # Load lora_A weights
        keys_A = ['q_proj.lora_A.weight', 'k_proj.lora_A.weight', 
                'v_proj.lora_A.weight', 'o_proj.lora_A.weight']
        
        for i, key in enumerate(keys_A):
            if key in layer_weights:
                weight = layer_weights[key]
                # The weights are already in the correct shape from the test (r, attn_head, head_dim)
                # We need to scale from attn_head to kv_head if necessary
                if weight.shape[1] != self.attention_config.kv_head_num:
                    ratio = int(round(weight.shape[1] / self.attention_config.kv_head_num))
                    weight = weight.reshape(rank, self.attention_config.kv_head_num, ratio, self.attention_config.head_dim).mean(dim=2)
                
                # Select the appropriate range of indices to store this weight matrix
                offset = i * rank
                end_offset = offset + rank
                selected_indices = loc[offset:end_offset]
                
                # Make sure the weight is the right shape before copying
                # In the buffer, each index holds a (kv_head_num, head_dim) tensor
                for j in range(rank):
                    if j < len(selected_indices):
                        with open('intermediate.txt', 'a') as f:
                            f.write(f"{layer_id} {selected_indices[j]} {weight[j]}")
                        self.unified_k_buffer[layer_id][selected_indices[j]].copy_(weight[j])
        
        # Load lora_B weights
        keys_B = ['q_proj.lora_B.weight', 'k_proj.lora_B.weight', 
                'v_proj.lora_B.weight', 'o_proj.lora_B.weight']
        
        for i, key in enumerate(keys_B):
            if key in layer_weights:
                weight = layer_weights[key]
                # Same downsampling logic as above
                if weight.shape[1] != self.attention_config.kv_head_num:
                    ratio = int(round(weight.shape[1] / self.attention_config.kv_head_num))
                    weight = weight.reshape(rank, self.attention_config.kv_head_num, ratio, self.attention_config.head_dim).mean(dim=2)
                
                # Select the appropriate range of indices
                offset = i * rank
                end_offset = offset + rank
                selected_indices = loc[offset:end_offset]
                
                # Copy each row of the weight matrix to the corresponding index
                for j in range(rank):
                    if j < len(selected_indices):
                        self.unified_v_buffer[layer_id][selected_indices[j]].copy_(weight[j])

    def _load_weights_mla(self, layer_id: int, loc: torch.Tensor, rank: int, layer_weights: Dict[str, torch.Tensor]):
        offset = 0
        for key in layer_weights:
            if 'lora_A' in key:
                weight = layer_weights[key]
                weight_size = rank
                self.unified_kv_buffer[layer_id][loc[offset:offset + weight_size]].copy_(weight)
                offset += weight_size
        offset = 0
        for key in layer_weights:
            if 'lora_B' in key:
                weight = layer_weights[key]
                weight_size = rank
                half_point = len(loc) // 2
                self.unified_kv_buffer[layer_id][loc[half_point + offset:half_point + offset + weight_size]].copy_(weight)
                offset += weight_size

    def get_adapter_memory_info(
        self, weight_name: str, layer_id: int, lora_type: LoRAType
    ) -> torch.Tensor:
        if not self.active_adapters:
            return torch.tensor([], dtype=torch.long, device=self.device)
        matrix_idx = 0
        if 'q_proj' in weight_name:
            matrix_idx = 0
        elif 'k_proj' in weight_name:
            matrix_idx = 1
        elif 'v_proj' in weight_name:
            matrix_idx = 2
        elif 'o_proj' in weight_name:
            matrix_idx = 3
        adapter_locations = []
        for adapter_id, info in self.active_adapters.items():
            rank = info.rank
            if lora_type == LoRAType.LORA_A:
                offset = matrix_idx * rank
                adapter_loc = info.loc[offset:offset+rank]
            else:
                if self.attention_type == AttentionType.MHA:
                    offset = matrix_idx * rank
                    adapter_loc = info.loc[offset:offset+rank]
                else:
                    half_point = len(info.loc) // 2
                    offset = matrix_idx * rank
                    adapter_loc = info.loc[half_point + offset:half_point + offset + rank]
            adapter_locations.append(adapter_loc)
        return torch.cat(adapter_locations)

    def get_buffer_id(self, adapter_id: Optional[str]) -> int:
        if adapter_id is None:
            return -1
        if adapter_id not in self.adapter_to_idx:
            raise ValueError(f"Adapter {adapter_id} not loaded")
        return self.adapter_to_idx[adapter_id]

    def get_tensor(
        self, weight_name: str, layer_id: int, lora_type: LoRAType
    ) -> torch.Tensor:
        loc = self.get_adapter_memory_info(weight_name, layer_id, lora_type)
        if lora_type == LoRAType.LORA_A:
            if self.attention_type == AttentionType.MHA:
                return self.unified_k_buffer[layer_id][loc]
            else:
                return self.unified_kv_buffer[layer_id][loc]
        else:
            if self.attention_type == AttentionType.MHA:
                return self.unified_v_buffer[layer_id][loc]
            else:
                return self.unified_kv_buffer[layer_id][loc]
