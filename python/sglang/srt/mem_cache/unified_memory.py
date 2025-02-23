import torch
from typing import Dict, Tuple, Optional, Set
from dataclasses import dataclass
from sglang.srt.lora.utils import LoRAType, get_weight_name, get_stacked_multiply
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.layers.radix_attention import RadixAttention

############################################
# Data classes for block and adapter tracking
############################################

@dataclass
class AdapterInfo:
    """Tracks information about a loaded LoRA adapter."""
    rank: int
    start_idx: int   # Starting cell index in the pool
    size: int        # Number of cells allocated (typically rank * 4)
    last_used: int   # For LRU tracking

@dataclass
class BlockInfo:
    """Information for any allocated block in the unified pool."""
    start: int
    size: int
    block_type: str  # e.g. "kv_cache" or "lora"
    metadata: dict   # For adapters, store adapter_id and rank

############################################
# UnifiedMemoryPool Implementation
############################################

class UnifiedMemoryPool:
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
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str = "cuda"
    ):
        # Basic configuration
        self.total_size = total_size  # total number of cells
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.device = device

        # Unified memory state: a 1D Boolean tensor, True means free.
        self.mem_state = torch.ones(total_size, dtype=torch.bool, device=device)
        self._mem_cumsum = torch.empty(total_size, dtype=torch.int32, device=device)
        self.available_size = total_size

        # For FP8 support: if dtype is a float8 type, use an alternative storage type.
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype

        # Create unified key and value buffers for each layer.
        self.key_buffer = [
            torch.empty((total_size, head_num, head_dim), dtype=self.store_dtype, device=device)
            for _ in range(layer_num)
        ]
        self.value_buffer = [
            torch.empty((total_size, head_num, head_dim), dtype=self.store_dtype, device=device)
            for _ in range(layer_num)
        ]

        # Block allocation bookkeeping.
        # allocated_blocks: block_id -> BlockInfo
        self.allocated_blocks: Dict[int, BlockInfo] = {}
        self.block_counter = 0

        # For LoRA adapter tracking:
        # adapter_to_block: adapter_id -> block_id
        # active_adapters: adapter_id -> AdapterInfo (for LRU tracking)
        self.adapter_to_block: Dict[str, int] = {}
        self.active_adapters: Dict[str, AdapterInfo] = {}
        self.access_counter = 0

        # Free-group support (optional, not used by default)
        self.free_group = []
        self.is_not_in_free_group = True

        # For LoRA manager: record weight names (set of tuples)
        self.lora_weight_names: Set[Tuple[str, ...]] = set()

    # ------------------ Core Unified Block Allocation ------------------

    def alloc_unified_block(self, size: int, block_type: str, metadata: dict = None) -> Optional[int]:
        """
        Allocate a contiguous block of 'size' cells from the entire pool.
        Searches the entire mem_state using cumulative sum.
        If successful, marks the block as allocated and returns a block ID.
        """
        if size > self.available_size:
            return None

        torch.cumsum(self.mem_state.to(torch.int32), dim=0, out=self._mem_cumsum)
        for i in range(self.total_size - size + 1):
            if i == 0:
                window_sum = self._mem_cumsum[size - 1]
            else:
                window_sum = self._mem_cumsum[i + size - 1] - self._mem_cumsum[i - 1]
            if window_sum.item() == size:
                indices = torch.arange(i, i + size, device=self.device)
                self.mem_state[indices] = False
                self.available_size -= size
                block_id = self.block_counter
                self.block_counter += 1
                self.allocated_blocks[block_id] = BlockInfo(start=i, size=size, block_type=block_type, metadata=metadata or {})
                return block_id
        return None

    def free_unified_block(self, block_id: int):
        """Free the block corresponding to block_id and update bookkeeping."""
        if block_id not in self.allocated_blocks:
            return
        info = self.allocated_blocks[block_id]
        indices = torch.arange(info.start, info.start + info.size, device=self.device)
        self.mem_state[indices] = True
        self.available_size += info.size
        if info.block_type == 'lora':
            adapter_id = info.metadata.get('adapter_id')
            if adapter_id in self.adapter_to_block:
                del self.adapter_to_block[adapter_id]
        del self.allocated_blocks[block_id]

    def clear(self):
        """Reset the entire pool: mark all cells free and clear bookkeeping."""
        self.mem_state.fill_(True)
        self.available_size = self.total_size
        self.allocated_blocks.clear()
        self.adapter_to_block.clear()
        self.active_adapters.clear()

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    # ---------------- BaseTokenToKVPool Interface (KV cache) ----------------

    def available_size_method(self) -> int:
        """Return the number of free cells available."""
        return self.available_size

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """
        Allocate 'need_size' contiguous free cells for KV cache usage.
        This calls alloc_unified_block with block_type 'kv_cache'.
        Returns a tensor of indices if successful.
        """
        block_id = self.alloc_unified_block(need_size, block_type='kv_cache')
        if block_id is None:
            return None
        info = self.allocated_blocks[block_id]
        return torch.arange(info.start, info.start + info.size, device=self.device)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        if self.store_dtype != self.dtype:
            return self.key_buffer[layer_id].view(self.dtype)
        return self.key_buffer[layer_id]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        if self.store_dtype != self.dtype:
            return self.value_buffer[layer_id].view(self.dtype)
        return self.value_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None
    ) -> None:
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k = cache_k / k_scale
            cache_k = cache_k.to(self.dtype)
        if cache_v.dtype != self.dtype:
            if v_scale is not None:
                cache_v = cache_v / v_scale
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.key_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
            self.value_buffer[layer_id][loc] = cache_v.view(self.store_dtype)
        else:
            self.key_buffer[layer_id][loc] = cache_k
            self.value_buffer[layer_id][loc] = cache_v

    # ---------------- Additional MHATokenToKVPool Methods ----------------

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        k_bytes = sum(int(torch.prod(torch.tensor(buf.shape)).item() * buf.element_size())
                      for buf in self.key_buffer)
        v_bytes = sum(int(torch.prod(torch.tensor(buf.shape)).item() * buf.element_size())
                      for buf in self.value_buffer)
        return k_bytes, v_bytes

    def get_flat_data(self, indices: torch.Tensor) -> torch.Tensor:
        k_data = torch.stack([buf[indices] for buf in self.key_buffer])
        v_data = torch.stack([buf[indices] for buf in self.value_buffer])
        return torch.stack([k_data, v_data])

    def transfer(self, indices: torch.Tensor, flat_data: torch.Tensor):
        k_data = flat_data[0]
        v_data = flat_data[1]
        for i in range(self.layer_num):
            self.key_buffer[i][indices] = k_data[i]
            self.value_buffer[i][indices] = v_data[i]

    # ---------------- LoRA Adapter Methods ----------------

    def alloc_lora_adapter(self, adapter_id: str, rank: int) -> bool:
        """
        Allocate memory for a LoRA adapter with a given rank.
        Required size = rank * 4 cells.
        """
        required_size = rank * 4
        block_id = self.alloc_unified_block(required_size, block_type='lora',
                                            metadata={'adapter_id': adapter_id, 'rank': rank})
        if block_id is None:
            return False
        self.adapter_to_block[adapter_id] = block_id
        info = self.allocated_blocks[block_id]
        self.active_adapters[adapter_id] = AdapterInfo(
            rank=rank,
            start_idx=info.start,
            size=required_size,
            last_used=self.access_counter
        )
        self.access_counter += 1
        return True

    def free_lora_adapter(self, adapter_id: str):
        if adapter_id not in self.adapter_to_block:
            return
        block_id = self.adapter_to_block[adapter_id]
        self.free_unified_block(block_id)
        if adapter_id in self.active_adapters:
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
            if uid not in self.adapter_to_block:
                adapter = lora_adapters.get(uid, None)
                if adapter is None:
                    continue
                success = self.alloc_lora_adapter(uid, adapter.r)
                if not success:
                    raise ValueError(f"Cannot allocate memory for adapter {uid}")
                self.load_lora_weight_to_buffer(uid, lora_adapter=adapter)

    def load_lora_weight_to_buffer(self, uid: str, buffer_id: Optional[int] = None, lora_adapter: Optional[LoRAAdapter] = None):
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

    def get_tensor(self, adapter_id: str, layer_id: int, lora_type: int) -> torch.Tensor:
        """
        Given an adapter identifier, return the corresponding weight slice
        for the specified layer and LoRA type.
        """
        if adapter_id not in self.active_adapters:
            raise ValueError(f"Adapter {adapter_id} not loaded")
        A, B = self.get_adapter_weights(adapter_id, layer_id)
        return A if lora_type == LoRAType.LORA_A else B

    def get_buffer_id(self, adapter_id: str) -> int:
        """
        Return the starting index (buffer id) of the adapter's allocated block.
        """
        if adapter_id not in self.active_adapters:
            raise ValueError(f"Adapter {adapter_id} not loaded")
        return self.active_adapters[adapter_id].start_idx

    def get_adapter_weights(self, adapter_id: str, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the weight slices for a LoRA adapter at a given layer.
        The allocated block (of size rank * 4) is split into two halves:
          - The first half for matrix A (lora_A)
          - The second half for matrix B (lora_B)
        """
        if adapter_id not in self.active_adapters:
            raise ValueError(f"Adapter {adapter_id} is not active")
        info = self.active_adapters[adapter_id]
        start = info.start_idx
        half = info.size // 2
        info.last_used = self.access_counter
        self.access_counter += 1
        A = self.key_buffer[layer_id][start: start + half]
        B = self.value_buffer[layer_id][start + half: start + info.size]
        return A, B

    def get_tensor_from_adapter(self, adapter_id: str, layer_id: int, lora_type: int) -> torch.Tensor:
        """Alias for get_tensor(adapter_id, layer_id, lora_type)."""
        return self.get_tensor(adapter_id, layer_id, lora_type)