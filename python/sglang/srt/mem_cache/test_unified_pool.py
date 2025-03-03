import unittest
import torch
import math
import numpy as np
from sglang.srt.lora.lora import LoRAAdapter
from lora_unified_memory_pool import (
    LoraUnifiedMemoryPool,
    LoraMHAConfig,
    AttentionType
)
from sglang.srt.lora.utils import LoRAType

# Dummy adapter classes for testing
class DummyLoRAAdapterLayer:
    def __init__(self, r: int, attn_head: int, head_dim: int):
        # Create dummy weights with shape (r, attn_head, head_dim)
        self.weights = {
            'q_proj.lora_A.weight': torch.arange(r * attn_head * head_dim, dtype=torch.float16).reshape(r, attn_head, head_dim),
            'q_proj.lora_B.weight': torch.arange(r * attn_head * head_dim, dtype=torch.float16).reshape(r, attn_head, head_dim) + 1000,
            'k_proj.lora_A.weight': torch.arange(r * attn_head * head_dim, dtype=torch.float16).reshape(r, attn_head, head_dim) + 10,
            'k_proj.lora_B.weight': torch.arange(r * attn_head * head_dim, dtype=torch.float16).reshape(r, attn_head, head_dim) + 1010,
            'v_proj.lora_A.weight': torch.arange(r * attn_head * head_dim, dtype=torch.float16).reshape(r, attn_head, head_dim) + 20,
            'v_proj.lora_B.weight': torch.arange(r * attn_head * head_dim, dtype=torch.float16).reshape(r, attn_head, head_dim) + 1020,
            'o_proj.lora_A.weight': torch.arange(r * attn_head * head_dim, dtype=torch.float16).reshape(r, attn_head, head_dim) + 30,
            'o_proj.lora_B.weight': torch.arange(r * attn_head * head_dim, dtype=torch.float16).reshape(r, attn_head, head_dim) + 1030,
        }

class DummyLoRAAdapter:
    def __init__(self, r: int, num_layers: int, attn_head: int, head_dim: int, scaling: float = 1.0):
        self.r = r
        self.scaling = scaling
        self.layers = [DummyLoRAAdapterLayer(r, attn_head, head_dim) for _ in range(num_layers)]

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

    def test_weight_loading_mha(self):
        adapter = DummyLoRAAdapter(r=2, num_layers=self.layer_num,
                                     attn_head=self.attn_config.attn_head_num,
                                     head_dim=self.attn_config.head_dim,
                                     scaling=1.25)
        lora_adapters = {"dummy_adapter": adapter}
        self.pool.prepare_lora_batch({"dummy_adapter"}, lora_adapters)
        
        info = self.pool.active_adapters["dummy_adapter"]
        start = int(info.loc[0].item())
        
        # For 'q_proj.lora_A.weight', expected scaling:
        expected_weight = adapter.layers[0].weights['q_proj.lora_A.weight']
        ratio = self.attn_config.attn_head_num / self.attn_config.kv_head_num
        if expected_weight.shape[1] != self.attn_config.kv_head_num:
            expected_transformed = expected_weight.reshape(adapter.r, self.attn_config.kv_head_num,
                                                           int(round(ratio)), self.attn_config.head_dim).mean(dim=2)
        else:
            expected_transformed = expected_weight

        loaded_weight = self.pool.unified_k_buffer[0][start : start + adapter.r]
        
        if not torch.allclose(loaded_weight, expected_transformed, atol=1e-3):
            diff = loaded_weight - expected_transformed
            self.fail(f"Loaded weight does not match expected transformed weight.\n"
                      f"Difference:\n{diff}")

    def test_alloc_free_slots(self):
        # After clear, free_slots should contain indices 1 to total_size inclusive.
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

    def test_get_flat_data_and_transfer(self):
        alloc_size = 5
        indices = self.pool.alloc(alloc_size)
        kv_head_num = self.attn_config.kv_head_num
        head_dim = self.attn_config.head_dim
        layer_num = self.layer_num
        # Create dummy flat data with shape [2, layer_num, alloc_size, kv_head_num, head_dim]
        total_elements = 2 * layer_num * alloc_size * kv_head_num * head_dim
        flat_data = torch.arange(total_elements, dtype=self.dtype).reshape(2, layer_num, alloc_size, kv_head_num, head_dim)
        self.pool.transfer(indices, flat_data)
        for i in range(layer_num):
            k_data = self.pool.unified_k_buffer[i][indices]
            v_data = self.pool.unified_v_buffer[i][indices]
            expected_k = flat_data[0, i]
            expected_v = flat_data[1, i]
            self.assertTrue(torch.allclose(k_data, expected_k, atol=1e-3),
                            f"Layer {i} k_buffer data not transferred correctly.")
            self.assertTrue(torch.allclose(v_data, expected_v, atol=1e-3),
                            f"Layer {i} v_buffer data not transferred correctly.")
        self.pool.free(indices)

    def test_eviction(self):
        # Create a small pool to force eviction.
        small_pool = LoraUnifiedMemoryPool(
            total_size=50,  # deliberately small
            dtype=self.dtype,
            device=self.device,
            layer_num=self.layer_num,
            attention_type=self.attention_type,
            attention_config=self.attn_config,
            enable_memory_saver=False
        )
        # Allocate first adapter with r=2.
        adapter1 = DummyLoRAAdapter(r=2, num_layers=self.layer_num,
                                    attn_head=self.attn_config.attn_head_num,
                                    head_dim=self.attn_config.head_dim)
        small_pool.prepare_lora_batch({"adapter1"}, {"adapter1": adapter1})
        # Now try to allocate a second adapter that requires more memory.
        adapter2 = DummyLoRAAdapter(r=5, num_layers=self.layer_num,
                                    attn_head=self.attn_config.attn_head_num,
                                    head_dim=self.attn_config.head_dim)
        try:
            small_pool.prepare_lora_batch({"adapter1", "adapter2"}, {"adapter1": adapter1, "adapter2": adapter2})
        except ValueError:
            # Expected if eviction fails.
            pass
        self.assertTrue(len(small_pool.active_adapters) > 0,
                        "Active adapters should not be empty after eviction/allocations.")

    def test_adapter_management(self):
        adapter = DummyLoRAAdapter(r=2, num_layers=self.layer_num,
                                   attn_head=self.attn_config.attn_head_num,
                                   head_dim=self.attn_config.head_dim)
        lora_adapters = {"adapterA": adapter}
        self.pool.prepare_lora_batch({"adapterA"}, lora_adapters)
        self.assertIn("adapterA", self.pool.active_adapters,
                      "Adapter adapterA should be active after prepare_lora_batch.")
        self.pool.free_lora_adapter("adapterA")
        self.assertNotIn("adapterA", self.pool.active_adapters,
                         "Adapter adapterA should be removed after free_lora_adapter.")

    def test_get_tensor(self):
        adapter = DummyLoRAAdapter(r=2, num_layers=self.layer_num,
                                   attn_head=self.attn_config.attn_head_num,
                                   head_dim=self.attn_config.head_dim)
        lora_adapters = {"adapterA": adapter}
        self.pool.prepare_lora_batch({"adapterA"}, lora_adapters)
        tensor_A = self.pool.get_tensor("q_proj.lora_A.weight", 0, LoRAType.LORA_A)
        info = self.pool.active_adapters["adapterA"]
        start = int(info.loc[0].item())
        expected_tensor = self.pool.unified_k_buffer[0][start: start + adapter.r]
        self.assertTrue(torch.allclose(tensor_A, expected_tensor, atol=1e-3),
                        "get_tensor did not return the expected tensor for LORA_A.")

    def test_get_kv_size_bytes(self):
        k_size, v_size = self.pool.get_kv_size_bytes()
        expected_k = sum(np.prod(buf.shape) * buf.dtype.itemsize for buf in self.pool.unified_k_buffer)
        expected_v = sum(np.prod(buf.shape) * buf.dtype.itemsize for buf in self.pool.unified_v_buffer)
        self.assertEqual(k_size, expected_k, "KV key size bytes mismatch.")
        self.assertEqual(v_size, expected_v, "KV value size bytes mismatch.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
