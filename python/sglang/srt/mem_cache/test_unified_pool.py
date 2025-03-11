import unittest
import torch
import math
import numpy as np
from lora_unified_memory_pool import (
    LoraUnifiedMemoryPool,
    LoraMHAConfig,
    AttentionType
)

# Dummy HF configuration to supply intermediate_size and hidden_size
class DummyHFConfig:
    def __init__(self, intermediate_size: int, hidden_size: int):
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

# Updated Dummy adapter layer with predictable weights for all projections
class DummyLoRAAdapterLayer:
    def __init__(self, r: int, attn_head: int, kv_head: int, head_dim: int, mlp_ratio: float = 4.0):
        head_ratio = attn_head // kv_head
        
        mlp_segment_length = math.ceil(r * head_ratio * mlp_ratio)
        print("test mlp_segment_length", mlp_segment_length, f"r={r}, head_ratio={head_ratio}, mlp_ratio={mlp_ratio}")
        self.weights = {
            'qkv_proj.lora_A.weight': torch.arange(r * 3 * head_ratio * (kv_head * head_dim), dtype=torch.float16).reshape(r * 3 * head_ratio, kv_head * head_dim),
            'o_proj.lora_A.weight': torch.arange(r * head_ratio * (kv_head * head_dim), dtype=torch.float16).reshape(r * head_ratio, kv_head * head_dim),
            'q_proj.lora_B.weight': torch.arange(r * head_ratio * (kv_head * head_dim), dtype=torch.float16).reshape(r * head_ratio, kv_head * head_dim),
            'kv_proj.lora_B.weight': torch.arange(2 * r * (kv_head * head_dim), dtype=torch.float16).reshape(2, r, kv_head, head_dim),
            'o_proj.lora_B.weight': torch.arange(r * head_ratio * (kv_head * head_dim), dtype=torch.float16).reshape(r * head_ratio, kv_head * head_dim),
            'down_proj.lora_A.weight': torch.arange(mlp_segment_length * (kv_head * head_dim), dtype=torch.float16).reshape(mlp_segment_length, kv_head * head_dim),
            'down_proj.lora_B.weight': torch.arange(r * head_ratio * (kv_head * head_dim), dtype=torch.float16).reshape(r * head_ratio, kv_head * head_dim),
            'gate_up_proj.lora_A.weight': torch.arange(2 * r * head_ratio * (kv_head * head_dim), dtype=torch.float16).reshape(2 * r * head_ratio, kv_head * head_dim),
            'gate_up_proj.lora_B.weight': torch.arange(2 * mlp_segment_length * (kv_head * head_dim), dtype=torch.float16).reshape(2, mlp_segment_length, kv_head * head_dim),
        }

# Dummy adapter that has several layers
class DummyLoRAAdapter:
    def __init__(self, r: int, num_layers: int, attn_head: int, kv_head: int, head_dim: int, scaling: float = 1.0, mlp_ratio: float = 4.0):
        self.rank = r
        self.scaling = scaling
        self.layers = [DummyLoRAAdapterLayer(r, attn_head, kv_head, head_dim, mlp_ratio) for _ in range(num_layers)]

class TestUnifiedMemoryPool(unittest.TestCase):
    def setUp(self):
        self.attn_config = LoraMHAConfig(
            attn_head_num=8,   # full number of attention heads
            kv_head_num=4,     # KV buffer uses 4 heads
            head_dim=8
        )
        self.size = 500
        self.layer_num = 2
        self.dtype = torch.float16
        self.device = "cpu"
        self.attention_type = AttentionType.MHA

        # Pass the dummy HF config to supply intermediate_size and hidden_size
        self.mlp_ratio = 3.5
        self.pool = LoraUnifiedMemoryPool(
            size=self.size,
            dtype=self.dtype,
            device=self.device,
            layer_num=self.layer_num,
            attention_type=self.attention_type,
            attention_config=self.attn_config,
            enable_memory_saver=False,
            base_hf_config=DummyHFConfig(intermediate_size=self.attn_config.attn_head_num*self.attn_config.head_dim*self.mlp_ratio, 
                                         hidden_size=self.attn_config.attn_head_num*self.attn_config.head_dim)
        )
        dummy_weight_names = {("qkv_proj", "q_proj"),
                              ("qkv_proj", "kv_proj"),
                              ("o_proj", "o_proj"),
                              ("down_proj", "down_proj"),
                              ("gate_up_proj", "gate_up_proj")}
        # For the dummy base_model, we can use a simple object
        dummy_base_model = object()  
        self.pool.init_buffers(dummy_weight_names, dummy_base_model)

    def test_weight_loading_mha(self):
        """Test that weights are correctly loaded for MHA for qkvo projections."""
        adapter1 = DummyLoRAAdapter(r=4, num_layers=self.layer_num,
                                    attn_head=self.attn_config.attn_head_num,
                                    kv_head=self.attn_config.kv_head_num,
                                    head_dim=self.attn_config.head_dim,
                                    scaling=1.25, mlp_ratio=self.mlp_ratio)
        adapter2 = DummyLoRAAdapter(r=2, num_layers=self.layer_num,
                                    attn_head=self.attn_config.attn_head_num,
                                    kv_head=self.attn_config.kv_head_num,
                                    head_dim=self.attn_config.head_dim,
                                    scaling=1.25, mlp_ratio=self.mlp_ratio)
        lora_adapters = {"dummy_adapter1": adapter1, "dummy_adapter2": adapter2}
        cur_uids = {"dummy_adapter1", "dummy_adapter2"}
        self.pool.prepare_lora_batch(cur_uids, lora_adapters)
        
        # Get memory info for qkvo projections
        loc, starts, lens = self.pool.get_adapter_memory_info("qkvo")
        # Check all layers for both adapters
        for adapter_idx, adapter_name in enumerate(["dummy_adapter1", "dummy_adapter2"]):
            adapter = lora_adapters[adapter_name]
            start_pos = starts[adapter_idx].item()
            length = lens[adapter_idx].item()
            adapter_locs = loc[start_pos:start_pos+length]
            
            for layer_id in range(self.layer_num):
                # Check A weights (stored in k_buffer)
                actual_A = self.pool.unified_k_buffer[layer_id][adapter_locs].reshape(-1, self.attn_config.kv_head_num, self.attn_config.head_dim)
                
                # Expected A weights: concatenated qkv and o projections
                expected_A = torch.cat([
                    adapter.layers[layer_id].weights['qkv_proj.lora_A.weight'],
                    adapter.layers[layer_id].weights['o_proj.lora_A.weight']
                ], dim=0).reshape(-1, self.attn_config.kv_head_num, self.attn_config.head_dim)
                
                self.assertTrue(torch.allclose(actual_A, expected_A),
                            f"A weights mismatch for adapter {adapter_name}, layer {layer_id}")
                
                # Check B weights (stored in v_buffer)
                actual_B = self.pool.unified_v_buffer[layer_id][adapter_locs]
                
                # Part of the B weights might be stacked for kv_proj, so we need special handling
                q_weights = adapter.layers[layer_id].weights['q_proj.lora_B.weight'].reshape(-1, self.attn_config.kv_head_num, self.attn_config.head_dim)
                kv_weights = adapter.layers[layer_id].weights['kv_proj.lora_B.weight']
                o_weights = adapter.layers[layer_id].weights['o_proj.lora_B.weight'].reshape(-1, self.attn_config.kv_head_num, self.attn_config.head_dim)
                
                # Combine all projection weights
                ratio = self.attn_config.attn_head_num // self.attn_config.kv_head_num
                q_length = int(ratio * adapter.rank)
                k_length = int(ratio * adapter.rank)
                v_length = int(ratio * adapter.rank)
                o_length = int(ratio * adapter.rank)
                
                # Check each segment separately
                # Q projection weights
                q_actual = actual_B[:q_length]
                self.assertTrue(torch.allclose(q_actual, q_weights),
                            f"Q projection B weights mismatch for adapter {adapter_name}, layer {layer_id}")
                
                # KV projection weights (stacked)
                kv_actual_k = actual_B[q_length:q_length+adapter.rank]
                kv_actual_v = actual_B[q_length+k_length:q_length+k_length+adapter.rank]
                self.assertTrue(torch.allclose(kv_actual_k, kv_weights[0]),
                            f"K projection B weights mismatch for adapter {adapter_name}, layer {layer_id}")
                self.assertTrue(torch.allclose(kv_actual_v, kv_weights[1]),
                            f"""V projection B weights mismatch for adapter {adapter_name}, layer {layer_id}, 
                            actual_v={kv_actual_v}, expected_v={kv_weights[1]}
                            diff={kv_actual_v - kv_weights[1]}
                            diffsum={torch.sum(kv_actual_v - kv_weights[1])}
                            diff indices={torch.nonzero(kv_actual_v - kv_weights[1])}
                            """)
                
                # O projection weights
                o_actual = actual_B[q_length+k_length+v_length:q_length+k_length+v_length+o_length]
                self.assertTrue(torch.allclose(o_actual, o_weights),
                            f"O projection B weights mismatch for adapter {adapter_name}, layer {layer_id}")
        
        # Free one adapter and add a new one to check re-allocation behavior
        self.pool.free_lora_adapter("dummy_adapter1")
        self.pool.prepare_lora_batch({"dummy_adapter3"}, {"dummy_adapter3": adapter1})

        # For a rank=4 adapter with head_ratio=8//4=2, expected qkvo segment length is:
        # expected_length = head_ratio * 4 * rank = 2 * 4 * 4 = 32.
        ratio = self.attn_config.attn_head_num / self.attn_config.kv_head_num
        expected_length = int(ratio * 4 * adapter1.rank)

        loc, start, lens = self.pool.get_adapter_memory_info("qkvo")
        self.assertEqual(lens.numel(), 1, "Expected one segment length for a single active adapter.")
        self.assertTrue(torch.allclose(lens, torch.tensor([expected_length], dtype=torch.long, device=self.device)),
                        "Segment length for qkvo does not match expected value.")
        self.assertEqual(loc.numel(), expected_length, "The number of location indices does not match expected qkvo segment length.")
        

    def test_alloc_free_slots(self):
        """Test that allocation and freeing of memory slots behaves correctly."""
        self.pool.clear()
        expected_count = self.size  # using self.size instead of self.total_size
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

    def test_up_down_projections(self):
        """Test that gate_up and down projection weights are loaded correctly."""
        # Use an adapter that includes gate_up and down projection weights.
        adapter = DummyLoRAAdapter(r=4, num_layers=self.layer_num,
                                attn_head=self.attn_config.attn_head_num,
                                kv_head=self.attn_config.kv_head_num,
                                head_dim=self.attn_config.head_dim, mlp_ratio=self.mlp_ratio)
        lora_adapters = {"adapter_up_down": adapter}
        self.pool.prepare_lora_batch({"adapter_up_down"}, lora_adapters)
        
        head_ratio = self.attn_config.attn_head_num // self.attn_config.kv_head_num
        expected_gate_up_length = math.ceil(2 * 4 * head_ratio * self.mlp_ratio)
        expected_down_length = math.ceil(4 * head_ratio * self.mlp_ratio)

        # Test gate_up projection
        for layer_id in range(self.layer_num):
            # Check gate_up weights
            loc_gate, start_gate, lens_gate = self.pool.get_adapter_memory_info("gate_up")
            self.assertEqual(lens_gate.numel(), 1, "Expected one segment for gate_up projection.")
            self.assertTrue(torch.allclose(lens_gate, torch.tensor([expected_gate_up_length], dtype=torch.long, device=self.device)),
                            f"Gate up segment length does not match expected value. Expected: {expected_gate_up_length}, Actual: {lens_gate.item()}")
            self.assertEqual(loc_gate.numel(), expected_gate_up_length, "Number of location indices for gate up projection is incorrect.")
            # Get the actual weights from the memory pool
            gate_up_locs = loc_gate[start_gate[0]:start_gate[0]+lens_gate[0]]
            # Check A weights for gate_up (stored in k_buffer)
            actual_A = self.pool.unified_k_buffer[layer_id][gate_up_locs].reshape(-1, self.attn_config.kv_head_num, self.attn_config.head_dim)
            expected_A = adapter.layers[layer_id].weights['gate_up_proj.lora_A.weight'].reshape(-1, self.attn_config.kv_head_num, self.attn_config.head_dim)
            actual_A = actual_A[:expected_A.shape[0]]  # Truncate to expected length because we alloc mlp_segment, but need only segment for A matrix
            self.assertTrue(torch.allclose(actual_A, expected_A),
                        f"""Gate-up A weights mismatch for layer {layer_id},
                        diffsum={torch.sum(actual_A - expected_A)}
                        diff indices={torch.nonzero(actual_A - expected_A)}
                        """)
            
            # Check B weights for gate_up (stored in v_buffer, might be stacked)
            actual_B = self.pool.unified_v_buffer[layer_id][gate_up_locs].reshape(2, -1, self.attn_config.kv_head_num, self.attn_config.head_dim)
            expected_B = adapter.layers[layer_id].weights['gate_up_proj.lora_B.weight'].reshape(2, -1, self.attn_config.kv_head_num, self.attn_config.head_dim) # 2, mlp_segment, kv_head * head_dim
            
            
            self.assertTrue(torch.allclose(actual_B[0], expected_B[0]),
                        f"Gate projection B weights mismatch for layer {layer_id}")
            self.assertTrue(torch.allclose(actual_B[1], expected_B[1]),
                        f"Up projection B weights mismatch for layer {layer_id}")

        # Test down projection
        for layer_id in range(self.layer_num):
            # Check down weights
            loc_down, start_down, lens_down = self.pool.get_adapter_memory_info("down")
            self.assertEqual(lens_down.numel(), 1, "Expected one segment for down projection.")
            self.assertTrue(torch.allclose(lens_down, torch.tensor([expected_down_length], dtype=torch.long, device=self.device)),
                            f"Down segment length doesn't match expected. Expected: {expected_down_length}, Actual: {lens_down.item()}")
            self.assertEqual(loc_down.numel(), expected_down_length, "Number of location indices for down projection is incorrect.")

            # Get the actual weights from the memory pool
            down_locs = loc_down[start_down[0]:start_down[0]+lens_down[0]]
            
            # Check A weights for down (stored in k_buffer)
            actual_A = self.pool.unified_k_buffer[layer_id][down_locs].reshape(-1, self.attn_config.kv_head_num, self.attn_config.head_dim)
            expected_A = adapter.layers[layer_id].weights['down_proj.lora_A.weight'].reshape(-1, self.attn_config.kv_head_num, self.attn_config.head_dim)
            self.assertTrue(torch.allclose(actual_A, expected_A),
                        f"Down projection A weights mismatch for layer {layer_id}")
            
            # Check B weights for down (stored in v_buffer)
            actual_B = self.pool.unified_v_buffer[layer_id][down_locs]
            expected_B = adapter.layers[layer_id].weights['down_proj.lora_B.weight'].reshape(-1, self.attn_config.kv_head_num, self.attn_config.head_dim)
            actual_B = actual_B[:expected_B.shape[0]] # we only need the segment for B matrix in down proj though we allocate mlp_segment to be consistent with A
            self.assertTrue(torch.allclose(actual_B, expected_B),
                        f"Down projection B weights mismatch for layer {layer_id}")
            
    def test_adapter_management(self):
        """Test that adapters are correctly added and removed from the pool."""
        adapter = DummyLoRAAdapter(r=2, num_layers=self.layer_num,
                                   attn_head=self.attn_config.attn_head_num,
                                   kv_head=self.attn_config.kv_head_num,
                                   head_dim=self.attn_config.head_dim, mlp_ratio=self.mlp_ratio)
        lora_adapters = {"adapterA": adapter}
        self.pool.prepare_lora_batch({"adapterA"}, lora_adapters)
        self.assertIn("adapterA", self.pool.active_adapters,
                      "Adapter adapterA should be active after prepare_lora_batch.")
        self.pool.free_lora_adapter("adapterA")
        self.assertNotIn("adapterA", self.pool.active_adapters,
                         "Adapter adapterA should be removed after free_lora_adapter.")

    def test_get_kv_size_bytes(self):
        """Test that get_kv_size_bytes returns the correct total size for K and V buffers."""
        k_size, v_size = self.pool.get_kv_size_bytes()
        expected_k = sum(np.prod(buf.shape) * buf.dtype.itemsize for buf in self.pool.unified_k_buffer)
        expected_v = sum(np.prod(buf.shape) * buf.dtype.itemsize for buf in self.pool.unified_v_buffer)
        self.assertEqual(k_size, expected_k, "KV key size bytes mismatch.")
        self.assertEqual(v_size, expected_v, "KV value size bytes mismatch.")

if __name__ == '__main__':
    unittest.main(verbosity=2)