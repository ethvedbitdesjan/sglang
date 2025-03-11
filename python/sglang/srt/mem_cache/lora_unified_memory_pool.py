import abc
import torch
import logging
import numpy as np
import math
from typing import Dict, Tuple, Optional, Set, List, Union
from dataclasses import dataclass
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.lora.utils import LoRAType, get_hidden_dim, get_weight_name, get_stacked_multiply, get_layer_id
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import debug_timing, get_compiler_backend
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.hf_transformers_utils import AutoConfig

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
        "gate_up_proj": 4
    }
    #calculate offset for down proj not implemented as it depends on intermediate size    
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

class AttentionType:
    MHA = "mha"  # Multi-Head Attention
    MLA = "mla"  # Multi-head Latent Attention

############################################
# LoraKVCache Implementation
############################################

class LoraMHATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        unified_k_buffer: List[torch.Tensor],
        unified_v_buffer: List[torch.Tensor],
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ):
        self.size = size
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
        assert hasattr(self, "unified_k_buffer")
        assert hasattr(self, "unified_v_buffer")
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

class LoraMLATokenToKVPool(KVCache):
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
        size: int,
        dtype: torch.dtype,
        device: str,
        layer_num: int,
        attention_type: str,
        attention_config: Union[LoraMHAConfig, LoraMLAConfig],
        enable_memory_saver: bool,
        base_hf_config: AutoConfig
    ):
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(enable=enable_memory_saver)
        self.size = size
        self.dtype = dtype
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device
        self.layer_num = layer_num
        self.attention_type = attention_type
        self.attention_config = attention_config
        self.base_hf_config = base_hf_config

        # memory allocated for lora adapters is contiguous or non-contiguous
        self.is_lora_contiguous = True

        with self.memory_saver_adapter.region():
            if self.attention_type == AttentionType.MHA:
                self.unified_k_buffer = [
                    torch.empty((size + 1, attention_config.kv_head_num, attention_config.head_dim),
                                dtype=self.store_dtype, device=device)
                    for _ in range(layer_num)
                ]
                self.unified_v_buffer = [
                    torch.empty((size + 1, attention_config.kv_head_num, attention_config.head_dim),
                                dtype=self.store_dtype, device=device)
                    for _ in range(layer_num)
                ]
                self.token_to_kv_pool = LoraMHATokenToKVPool(
                    size=size,
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
                    torch.empty((size + 1, 1, attention_config.kv_lora_rank + attention_config.qk_rope_head_dim),
                                dtype=self.store_dtype, device=device)
                    for _ in range(layer_num)
                ]
                self.token_to_kv_pool = LoraMLATokenToKVPool(
                    total_size=size,
                    unified_kv_buffer=self.unified_kv_buffer,
                    dtype=dtype,
                    kv_lora_rank=attention_config.kv_lora_rank,
                    qk_rope_head_dim=attention_config.qk_rope_head_dim,
                    layer_num=layer_num,
                    device=device,
                )
            else:
                raise ValueError(f"Unsupported attention type: {self.attention_type}")

        # total active adapters in gpu memory
        self.active_adapters: Dict[str, AdapterInfo] = {}

        # adapters in the current batch
        self.cur_adapters: Dict[str, AdapterInfo] = {}

        self.access_counter = 0
        self.lora_weight_names: Set[Tuple[str, ...]] = set()


        # Lora uid -> buffer idx in memory pool
        self.uid_to_buffer_id: Dict[Optional[str], int] = {}
        # Buffer idx -> lora uid in memory pool
        self.buffer_id_to_uid: List[Optional[str]] = []
        self.adapter_buffer: Dict[LoRAType,Dict[str, List[torch.Tensor]]] = {}

        self.free_slots = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

        print('init LoraUnifiedMemoryPool')

    # --- Core Memory Allocation Methods ---
    def available_size(self):
        return len(self.free_slots)
    
    def get_kvcache(self):
        return self.token_to_kv_pool

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
        
        raise ValueError(f"Cannot allocate contiguous memory for adapter")
        # No contiguous block of sufficient size found
        # return None

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
        self.free_slots = torch.arange(1, self.size + 1, dtype=torch.int32)
        self.is_in_free_group = False
        self.free_group = []
        self.active_adapters = {}
        self.access_counter = 0

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
    def alloc_lora_adapter(self, uid: str, rank: int) -> bool:
        if self.attention_type == AttentionType.MHA:
            head_ratio = self.attention_config.attn_head_num // self.attention_config.kv_head_num
            qkvo_size = int(rank * 4 * head_ratio)
            if self.base_hf_config is not None and hasattr(self.base_hf_config, "intermediate_size"):
                gate_up_size = math.ceil(rank * 2* head_ratio * (self.base_hf_config.intermediate_size / self.base_hf_config.hidden_size))
                down_size = math.ceil(rank * head_ratio * (self.base_hf_config.intermediate_size / self.base_hf_config.hidden_size))
                required_size = qkvo_size + gate_up_size + down_size
            else:
                required_size = qkvo_size
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        adapter_loc = self.alloc(required_size)
        if adapter_loc is None:
            raise ValueError(f"Cannot allocate memory for adapter {uid}")

        self.active_adapters[uid] = AdapterInfo(
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
        self.lora_weight_names = lora_weight_names
        self.base_model = base_model
        self.adapter_buffer[LoRAType.LORA_A] = {}
        self.adapter_buffer[LoRAType.LORA_B] = {}
        for module_A, module_B in lora_weight_names:
            if module_A not in self.adapter_buffer[LoRAType.LORA_A]:
                self.adapter_buffer[LoRAType.LORA_A][module_A] = [
                    None
                    for _ in range(self.layer_num)
                ]
            if module_B not in self.adapter_buffer[LoRAType.LORA_B]:
                self.adapter_buffer[LoRAType.LORA_B][module_B] = [
                    None
                    for _ in range(self.layer_num)
                ]

    def get_unified_memory_pool(self,layer_id: int)-> Tuple[torch.Tensor,torch.Tensor]:
        if self.attention_type == AttentionType.MHA:
            return self.unified_k_buffer[layer_id],self.unified_v_buffer[layer_id]
        else:
            raise ValueError('get_unified_memory_pool error')

    def prepare_lora_batch(self, cur_uids: Set[Optional[str]], lora_adapters: Dict[str, LoRAAdapter]):
        self.cur_adapters = {}
        self.uid_to_buffer_id = {}
        self.buffer_id_to_uid = []

        buffer_id = 0

        cur_uids = list(sorted(cur_uids))
        for buffer_id, uid in enumerate(cur_uids):
            if uid == None:
                break
            # assert(uid is not None)
            self.uid_to_buffer_id[uid] = buffer_id
            self.buffer_id_to_uid.append(uid)
            if uid in self.active_adapters:
                self.active_adapters[uid].last_used = self.access_counter
                self.cur_adapters[uid] = self.active_adapters[uid]
                self.access_counter += 1
                continue
            adapter = lora_adapters.get(uid, None)
            if adapter is None:
                raise ValueError("adapter is None, uid:",{uid})
            
            success = self.alloc_lora_adapter(uid, adapter.rank)
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
        head_ratio = self.attention_config.attn_head_num // self.attention_config.kv_head_num
        if self.base_hf_config is not None and hasattr(self.base_hf_config, "intermediate_size"):
            mlp_segment_length = math.ceil(rank * head_ratio * (self.base_hf_config.intermediate_size / self.base_hf_config.hidden_size))
        else:
            mlp_segment_length = 0
        for name, weights in layer_weights.items():
            segment_length = int(rank * head_ratio)
            if "lora_A" in name:
                weight_name = get_weight_name(name, self.lora_weight_names, LoRAType.LORA_A)
                if weight_name is None:
                    continue
                
                c = get_stacked_multiply(weight_name)
                
                
                proj_index = get_projection_index(weight_name)
                if proj_index is None:
                    if 'down_proj' not in weight_name:
                        continue
                    offset = segment_length * 4 + mlp_segment_length * 2
                    segment_length = mlp_segment_length
                else:
                    offset = proj_index * segment_length
                
                if 'up_proj' in weight_name:
                    segment_length = 2 * segment_length
                else:
                    segment_length = c * segment_length
                
                selected_indices = info_loc[offset : offset + segment_length]
                weights = weights.view(-1,
                                       self.attention_config.kv_head_num, 
                                       self.attention_config.head_dim).to(device=self.device,dtype=self.store_dtype)
                
                assert weights.shape[0] == segment_length, f"weights.shape[0] != segment_length: {weights.shape[0]} != {segment_length}, name: {name}, {weights.shape}"
                self.unified_k_buffer[layer_id][selected_indices] = weights
            else:
                weight_name = get_weight_name(name, self.lora_weight_names, LoRAType.LORA_B)
                if weight_name is None:
                    continue
                c = get_stacked_multiply(weight_name)
                proj_index = get_projection_index(weight_name)
                if proj_index is None:
                    if 'down_proj' not in weight_name:
                        continue
                    offset = segment_length * 4 + mlp_segment_length * 2
                    segment_length = segment_length
                else:
                    offset = proj_index * segment_length
                if 'up_proj' in weight_name:
                    segment_length = mlp_segment_length # will be stacked so no need to multiply by 2
                
                if c > 1:
                    weights = weights.view(c, -1, self.attention_config.kv_head_num, 
                                            self.attention_config.head_dim).to(device=self.device,dtype=self.store_dtype)
                    if weights.shape[1] == rank and ('k_proj' in weight_name or 'v_proj' in weight_name or 'kv_proj' in weight_name):
                        #k,v have reduced shapes
                        #zero add to the dim=1 of weights to make it equal segment_length
                        weights = torch.cat([weights, torch.zeros(c, segment_length - rank, self.attention_config.kv_head_num, 
                                                                  self.attention_config.head_dim, device=self.device,dtype=self.store_dtype)],
                                            dim=1)
                    assert weights.shape[1] == segment_length, f"weights.shape[1] != segment_length: {weights.shape[1]} != {segment_length}, name: {name}, {weights.shape}"
                    for stacked_id in range(c):
                        stacked_offset = offset + segment_length * stacked_id
                        selected_indices = info_loc[stacked_offset:stacked_offset + segment_length]
                        self.unified_v_buffer[layer_id][selected_indices] = weights[stacked_id]
                else:
                    weights = weights.view(-1, 
                                           self.attention_config.kv_head_num, 
                                           self.attention_config.head_dim).to(device=self.device,dtype=self.store_dtype)
                    assert weights.shape[0] == segment_length, f"weights.shape[0] != segment_length: {weights.shape[0]} != {segment_length}, name: {name}, {weights.shape}"
                    selected_indices = info_loc[offset : offset + segment_length]
                    self.unified_v_buffer[layer_id][selected_indices] = weights

    def get_buffer_id(self, lora_uid: str):
        return self.uid_to_buffer_id[lora_uid]

    def get_adapter_memory_info(self, proj_type: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.cur_adapters:
            raise ValueError("self.cur_adapters is empty")

        if self.attention_type != AttentionType.MHA:
            raise NotImplementedError("Only MHA is currently supported")

        head_ratio = self.attention_config.attn_head_num // self.attention_config.kv_head_num
        locs = []
        starts = []
        lens = []
        if proj_type == "qkvo":
            start_pos = 0
            for uid in self.buffer_id_to_uid:
                info = self.cur_adapters[uid]
                rank = info.rank
                segment_length = int(head_ratio * 4 * rank)
                offset = 0
                end_offset = segment_length
                if end_offset <= len(info.loc):
                    qkvo_loc = info.loc[offset:end_offset]
                    locs.append(qkvo_loc)
                    starts.append(torch.tensor([start_pos], dtype=torch.long, device=self.device))
                    lens.append(torch.tensor([segment_length], dtype=torch.long, device=self.device))
                else:
                    raise ValueError("end_offset > len(info.loc)")
                start_pos += segment_length
        elif proj_type == "gate_up" or proj_type == "down":
            start_pos = 0
            for uid in self.buffer_id_to_uid:
                info = self.cur_adapters[uid]
                rank = info.rank
                start_pos += int(head_ratio * 4 * rank)
                segment_length = int(head_ratio * rank * (self.base_hf_config.intermediate_size / self.base_hf_config.hidden_size))
                offset = start_pos
                if proj_type == "gate_up":
                    segment_length *= 2
                else:
                    offset += segment_length * 2
                    start_pos = offset
                    
                end_offset = offset + segment_length
                if end_offset <= len(info.loc):
                    gate_loc = info.loc[offset:end_offset]
                    locs.append(gate_loc)
                    starts.append(torch.tensor([start_pos], dtype=torch.long, device=self.device))
                    lens.append(torch.tensor([segment_length], dtype=torch.long, device=self.device))
                else:
                    raise ValueError("end_offset > len(info.loc)")
                start_pos += segment_length
                if proj_type == "gate_up":
                    start_pos += segment_length//2
        else:
            raise ValueError(f"Unsupported adapter cache type: {proj_type}")

        return (
            torch.cat(locs) if locs else torch.tensor([], dtype=torch.long, device=self.device),
            torch.cat(starts) if starts else torch.tensor([], dtype=torch.long, device=self.device),
            torch.cat(lens) if lens else torch.tensor([], dtype=torch.long, device=self.device)
        )


    # def prepare_lora_batch(
    #     self,
    #     cur_uids: Set[Optional[str]],
    #     lora_adapters: Dict[str, LoRAAdapter],
    # ):
    #     # clear current adapters for next batch
    #     self.cur_adapters = {}
    #     self.uid_to_buffer_id = {}
    #     self.buffer_id_to_uid = []

    #     buffer_id = 0
    #     total_rank = 0
    #     rank_offset = [0]
    #     for uid in cur_uids:
    #         if uid not in self.uid_to_buffer_id:
    #             self.uid_to_buffer_id[uid] = buffer_id
    #             self.buffer_id_to_uid.append(uid)
    #             buffer_id += 1

    #             total_rank += adapter.rank
    #             rank_offset.append(total_rank)

    #     for layer_id in range(self.layer_num):
    #         layer_weights = adapter.layers[layer_id].weights
    #         for name, weights in layer_weights.items():
    #             if "lora_A" in name:
    #                 lora_weight_name = get_weight_name(
    #                     name, self.lora_weight_names, LoRAType.LORA_A
    #                 )
    #                 c = get_stacked_multiply(lora_weight_name)

    #                 if self.attention_type == AttentionType.MHA:
    #                     head_ratio = self.attention_config.attn_head_num // self.attention_config.kv_head_num
    #                     required_size = math.ceil(total_rank * c * head_ratio)
    #                 else:
    #                     raise ValueError(f"Unsupported attention type: {self.attention_type}")

    #                 loc = self.alloc_contiguous(required_size)
    #                 tensor = self.unified_k_buffer[layer_id][loc]
    #                 assert(tensor.is_contiguous())
                    
    #                 input_dim = get_hidden_dim(lora_weight_name,self.base_hf_config,self.base_model)[0]

    #                 # self.adapter_buffer[LoRAType.LORA_A][lora_weight_name][layer_id] = tensor.view(-1,self.max_lora_dim * c,input_dim)
    #                 self.adapter_buffer[LoRAType.LORA_A][lora_weight_name][layer_id] = tensor.view(-1,input_dim)

    #                 rank_offset * c * head_ratio

    #                 for uid in cur_uids:
    #                     buffer_id = self.uid_to_buffer_id[uid]
    #                     adapter = lora_adapters[uid]
    #                     adapter_info = AdapterInfo(
    #                         rank=adapter.rank,
    #                         loc=loc[],
    #                         size=required_size,
    #                         last_used=self.access_counter
    #                     )



    #                 if lora_weight_name:
    #                     self.A_buffer[lora_weight_name][layer_id][buffer_id].copy_(
    #                         weights
    #                     )
    #             else:
    #                 lora_weight_name = get_weight_name(
    #                     name, self.lora_weight_names, LoRAType.LORA_B
    #                 )
    #                 if lora_weight_name:
    #                     c = get_stacked_multiply(lora_weight_name)
    #                     if c > 1:
    #                         for stacked_id in range(c):
    #                             self.B_buffer[lora_weight_name][layer_id][stacked_id][
    #                                 buffer_id
    #                             ].copy_(weights[stacked_id])
    #                     else:
    #                         self.B_buffer[lora_weight_name][layer_id][0][
    #                             buffer_id
    #                         ].copy_(weights)
            
    #             adapter = lora_adapters.get(uid, None)
    #             if adapter is None:
    #                 raise ValueError(f"adapter is None: {uid}")
                



    #             if self.attention_type == AttentionType.MHA:
    #                 head_ratio = self.attention_config.attn_head_num // self.attention_config.kv_head_num
    #                 required_size = math.ceil(adapter.rank * 4 * head_ratio)
    #             else:
    #                 raise ValueError(f"Unsupported attention type: {self.attention_type}")

    #             adapter_loc = self.alloc_contiguous(required_size)

    #             adapter_info = AdapterInfo(
    #                 rank=adapter.rank,
    #                 loc=adapter_loc,
    #                 size=required_size,
    #                 last_used=self.access_counter
    #             )
                

    #             self.active_adapters[uid] = AdapterInfo(
    #                 rank=rank,
    #                 loc=adapter_loc,
    #                 size=required_size,
    #                 last_used=self.access_counter
    #             )
    #             success = self.alloc_lora_adapter(uid, adapter.rank)
    #             if not success:
    #                 raise ValueError(f"Cannot allocate memory for adapter {uid}")
    #             self.load_lora_weight_to_buffer(uid, lora_adapter=adapter)

    # def load_lora_weight_to_buffer(
    #     self, uid: str, buffer_id: int, lora_adapter: LoRAAdapter = None
    # ):

    #     if uid is None:
    #         for i in range(self.num_layer):
    #             for k in self.A_buffer.keys():
    #                 self.A_buffer[k][i][buffer_id] *= 0
    #         return

    #     assert lora_adapter is not None
    #     for layer_id in range(self.num_layer):
    #         layer_weights = lora_adapter.layers[layer_id].weights
    #         for name, weights in layer_weights.items():
    #             if "lora_A" in name:
    #                 lora_weight_name = get_weight_name(
    #                     name, self.lora_weight_names, LoRAType.LORA_A
    #                 )
    #                 if lora_weight_name:
    #                     self.A_buffer[lora_weight_name][layer_id][buffer_id].copy_(
    #                         weights
    #                     )
    #             else:
    #                 lora_weight_name = get_weight_name(
    #                     name, self.lora_weight_names, LoRAType.LORA_B
    #                 )
    #                 if lora_weight_name:
    #                     c = get_stacked_multiply(lora_weight_name)
    #                     if c > 1:
    #                         for stacked_id in range(c):
    #                             self.B_buffer[lora_weight_name][layer_id][stacked_id][
    #                                 buffer_id
    #                             ].copy_(weights[stacked_id])
    #                     else:
    #                         self.B_buffer[lora_weight_name][layer_id][0][
    #                             buffer_id
    #                         ].copy_(weights)

    # def get_tensor(
    #     self, weight_name: str, layer_id: int, lora_type: LoRAType
    # ) -> torch.Tensor:

    #     if lora_type == LoRAType.LORA_A:
    #         return self.A_buffer[weight_name][layer_id]

    #     return self.B_buffer[weight_name][layer_id]

    # def get_buffer_id(self, lora_uid: str):
    #     return self.uid_to_buffer_id[lora_uid]
