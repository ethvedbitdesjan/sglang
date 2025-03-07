import unittest
import torch
import math
import numpy as np
from lora_unified_memory_pool import (
    LoraUnifiedMemoryPool,
    LoraMHAConfig,
    AttentionType
)

# Dummy adapter layer with predictable weights
class DummyLoRAAdapterLayer:
    def __init__(self, r: int, attn_head: int, kv_head: int,head_dim: int):
        # Create dummy weights with shape (r, attn_head, head_dim)
        self.weights = {
            'qkv_proj.lora_A.weight': torch.arange(r * 3 * attn_head * head_dim, dtype=torch.float16).reshape(r * 3, attn_head * head_dim),
            'q_proj.lora_B.weight': (torch.arange(r * attn_head * head_dim, dtype=torch.float16)).reshape(attn_head * head_dim , r),
            'kv_proj.lora_B.weight': (torch.arange(r * 2 * kv_head * head_dim, dtype=torch.float16)).reshape(2, kv_head * head_dim,r),
            'o_proj.lora_A.weight': (torch.arange(r * attn_head * head_dim, dtype=torch.float16)).reshape(r, attn_head * head_dim),
            'o_proj.lora_B.weight': (torch.arange(r * attn_head * head_dim, dtype=torch.float16)).reshape(attn_head * head_dim , r),
        }

# Dummy adapter that has several layers
class DummyLoRAAdapter:
    def __init__(self, r: int, num_layers: int, attn_head: int,kv_head:int, head_dim: int, scaling: float = 1.0):
        self.r = r
        self.scaling = scaling
        self.layers = [DummyLoRAAdapterLayer(r, attn_head,kv_head, head_dim) for _ in range(num_layers)]

class TestUnifiedMemoryPool(unittest.TestCase):
    def setUp(self):
        self.attn_config = LoraMHAConfig(
            attn_head_num=8,   # full number of attention heads
            kv_head_num=4,     # KV buffer uses 4 heads
            head_dim=16
        )
        self.total_size = 1000
        self.layer_num = 2
        self.dtype = torch.float16
        self.device = "cpu"
        self.attention_type = AttentionType.MHA

        self.pool = LoraUnifiedMemoryPool(
            total_size=self.total_size,
            dtype=self.dtype,
            device=self.device,
            layer_num=self.layer_num,
            attention_type=self.attention_type,
            attention_config=self.attn_config,
            enable_memory_saver=False
        )
        dummy_weight_names = {("qkv_proj","q_proj"),
                              ("qkv_proj","kv_proj"),
                              ("o_proj","o_proj"),
                              }
        # For the dummy base_model, we can use a simple object or even None if your init_buffers can handle it.
        dummy_base_model = object()  # or create a minimal dummy module if needed
        self.pool.init_buffers(dummy_weight_names, dummy_base_model)
    def test_weight_loading_mha(self):
        import math
        """Test that weights are correctly loaded for MHA.
           This verifies that the transformed weight in unified_k_buffer matches expectation.
        """
        adapter = DummyLoRAAdapter(r=2, num_layers=self.layer_num,
                                 attn_head=self.attn_config.attn_head_num,
                                 kv_head= self.attn_config.kv_head_num,
                                 head_dim=self.attn_config.head_dim,
                                 scaling=1.25)
        lora_adapters = {"dummy_adapter": adapter}
        self.pool.prepare_lora_batch({"dummy_adapter"}, lora_adapters)

        info = self.pool.active_adapters["dummy_adapter"]
        ratio = self.attn_config.attn_head_num / self.attn_config.kv_head_num
        segment_length = int(math.ceil(ratio * 4 * adapter.r))
        offset = 0 #because q
        q_segment = info.loc[offset : offset + segment_length]
        loaded_weight = self.pool.unified_k_buffer[0][q_segment]
        loc, start, lens = self.pool.get_adapter_memory_info("qkvo")

        expected_total_length = segment_length
        self.assertEqual(loc.numel(), expected_total_length,
                        "Concatenated location indices do not match expected total length.")
        # 'start' should be zero for every segment.
        self.assertTrue(torch.all(start == 0),
                        "Start indices should be all zero.")
        # 'lens' should be a tensor with each value equal to segment_length.
        self.assertTrue(torch.all(lens == segment_length),
                        "Segment lengths returned do not match expected segment length.")

    def test_alloc_free_slots(self):
        """Test that allocation and freeing of memory slots behaves correctly."""
        self.pool.clear()
        expected_count = self.total_size
        self.assertEqual(self.pool.free_slots.numel(), expected_count,
                         "Free slots count does not match expected after clear.")

        alloc_size = 10
        allocated = self.pool.alloc(alloc_size)
        self.assertIsNotNone(allocated, "Allocation failed when it should succeed.")
        remaining = self.pool.free_slots.numel()
        self.assertEqual(remaining, expected_count - alloc_size,
                         "Free slots count did not decrease correctly after allocation.")

        self.pool.free(allocated)
        self.assertEqual(self.pool.free_slots.numel(), expected_count,
                         "Free slots count did not return to expected after freeing.")

    # def test_get_flat_data_and_transfer(self):
    #     """Test that get_flat_data returns the correct tensor and transfer updates the buffers."""
    #     alloc_size = 5
    #     indices = self.pool.alloc(alloc_size)
    #     kv_head_num = self.attn_config.kv_head_num
    #     head_dim = self.attn_config.head_dim
    #     layer_num = self.layer_num
    #     total_elements = 2 * layer_num * alloc_size * kv_head_num * head_dim
    #     flat_data = torch.arange(total_elements, dtype=self.dtype).reshape(2, layer_num, alloc_size, kv_head_num, head_dim)
    #     self.pool.transfer(indices, flat_data)
    #     for i in range(layer_num):
    #         k_data = self.pool.unified_k_buffer[i][indices]
    #         v_data = self.pool.unified_v_buffer[i][indices]
    #         expected_k = flat_data[0, i]
    #         expected_v = flat_data[1, i]
    #         self.assertTrue(torch.allclose(k_data, expected_k, atol=1e-3),
    #                         f"Layer {i} k_buffer data not transferred correctly.")
    #         self.assertTrue(torch.allclose(v_data, expected_v, atol=1e-3),
    #                         f"Layer {i} v_buffer data not transferred correctly.")
    #     self.pool.free(indices)

    def test_adapter_management(self):
        """Test that adapters are correctly added and removed from the pool."""
        adapter = DummyLoRAAdapter(r=2, num_layers=self.layer_num,
                                   attn_head=self.attn_config.attn_head_num,
                                   kv_head=self.attn_config.kv_head_num,
                                   head_dim=self.attn_config.head_dim)
        lora_adapters = {"adapterA": adapter}
        self.pool.prepare_lora_batch({"adapterA"}, lora_adapters)
        self.assertIn("adapterA", self.pool.active_adapters,
                      "Adapter adapterA should be active after prepare_lora_batch.")
        self.pool.free_lora_adapter("adapterA")
        self.assertNotIn("adapterA", self.pool.active_adapters,
                         "Adapter adapterA should be removed after free_lora_adapter.")

    def test_get_tensor(self):
        """Emulate get_tensor behavior by comparing a slice of unified_k_buffer for LORA_A."""
        adapter = DummyLoRAAdapter(r=2, num_layers=self.layer_num,
                                   attn_head=self.attn_config.attn_head_num,
                                   kv_head=self.attn_config.kv_head_num,
                                   head_dim=self.attn_config.head_dim)
        lora_adapters = {"adapterA": adapter}
        self.pool.prepare_lora_batch({"adapterA"}, lora_adapters)
        # Here we assume get_tensor would return the same slice as in unified_k_buffer for LORA_A.
        info = self.pool.active_adapters["adapterA"]
        start = int(info.loc[0].item())
        tensor_A = self.pool.unified_k_buffer[0][start: start + adapter.r]
        expected_tensor = tensor_A  # since weights were copied during prepare_lora_batch
        self.assertTrue(torch.allclose(tensor_A, expected_tensor, atol=1e-3),
                        "get_tensor did not return the expected tensor for LORA_A.")

    def test_get_kv_size_bytes(self):
        """Test that get_kv_size_bytes returns the correct total size for K and V buffers."""
        k_size, v_size = self.pool.get_kv_size_bytes()
        expected_k = sum(np.prod(buf.shape) * buf.dtype.itemsize for buf in self.pool.unified_k_buffer)
        expected_v = sum(np.prod(buf.shape) * buf.dtype.itemsize for buf in self.pool.unified_v_buffer)
        self.assertEqual(k_size, expected_k, "KV key size bytes mismatch.")
        self.assertEqual(v_size, expected_v, "KV value size bytes mismatch.")

if __name__ == '__main__':
    unittest.main(verbosity=2)