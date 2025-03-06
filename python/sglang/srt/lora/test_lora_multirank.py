import torch
import numpy as np
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.lora.triton_ops.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.srt.lora.triton_ops.sgemm_lora_b import sgemm_lora_b_fwd
from sglang.srt.lora.triton_ops.gate_up_lora_b import gate_up_lora_b_fwd
from sglang.srt.lora.triton_ops.qkv_lora_b import qkv_lora_b_fwd

def test_sgemm_multirank():
    """
    Test basic LoRA multirank functionality
    """
    print("===== Testing Basic LoRA Multirank Functionality =====")
    
    # Test data setup
    bs = 2  # batch size
    s = 10  # sequence length
    input_dim = 64  # input dimension
    
    # Create three different rank LoRA weights
    r4 = 4
    r8 = 8 
    r16 = 16
    max_r = 16
    
    # Input tensor
    x = torch.randn((s, input_dim), device="cuda")
    
    # Weight tensors - three different rank weights
    weights_a = torch.randn((3, max_r, input_dim), device="cuda")
    # First adapter only uses r4
    weights_a[0, r4:, :] = 0
    # Second adapter uses r8
    weights_a[1, r8:, :] = 0
    
    weights_b = torch.randn((3, input_dim, max_r), device="cuda")
    # First adapter only uses r4
    weights_b[0, :, r4:] = 0
    # Second adapter uses r8
    weights_b[1, :, r8:] = 0
    
    # Create batch information
    seg_lens = torch.tensor([3, 3, 4], dtype=torch.int32, device="cuda")
    seg_indptr = torch.tensor([0, 3, 6, 10], dtype=torch.int32, device="cuda")
    weight_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")  # Using three different weights
    
    # Create different rank values
    rank_values = torch.tensor([r4, r8, r16], dtype=torch.int32, device="cuda")
    
    # Create batch info
    batch_info = LoRABatchInfo(
        bs=bs+1,  # Note: this is 3 requests
        seg_lens=seg_lens,
        seg_indptr=seg_indptr,
        max_len=4,
        weight_indices=weight_indices
    )
    # Set rank values
    batch_info.rank_values = rank_values
    
    # Run LoRA computation
    print("Running LoRA A and B computation...")
    output_a = sgemm_lora_a_fwd(x, weights_a, batch_info)
    output_b = sgemm_lora_b_fwd(output_a, weights_b, batch_info)
    
    # Verify results
    print(f"Input tensor shape: {x.shape}")
    print(f"LoRA A weights shape: {weights_a.shape}")
    print(f"LoRA B weights shape: {weights_b.shape}")
    print(f"LoRA A output shape: {output_a.shape}")
    print(f"LoRA B output shape: {output_b.shape}")
    print(f"rank_values: {rank_values}")
    
    # Validate that first sequence only used r4 portion
    first_seq_output_a = output_a[:3, r4:]
    if torch.allclose(first_seq_output_a, torch.zeros_like(first_seq_output_a)):
        print("✓ First sequence successfully limited to rank=4")
    else:
        print("✗ First sequence not limited to rank=4")
        print(f"Number of non-zero values: {torch.sum(first_seq_output_a != 0)}")
    
    # Validate that second sequence used r8 portion
    second_seq_output_a = output_a[3:6, :r8]
    if torch.all(second_seq_output_a != 0):
        print("✓ Second sequence successfully used rank=8")
    else:
        print("✗ Second sequence did not use rank=8")
        print(f"Number of zero values: {torch.sum(second_seq_output_a == 0)}")
    
    # Validate that second sequence did not use beyond r8
    second_seq_r8_plus = output_a[3:6, r8:]
    if torch.allclose(second_seq_r8_plus, torch.zeros_like(second_seq_r8_plus)):
        print("✓ Second sequence successfully limited to rank=8")
    else:
        print("✗ Second sequence not limited to rank=8") 
        print(f"Number of non-zero values: {torch.sum(second_seq_r8_plus != 0)}")
    
    # Validate that third sequence used full r16
    third_seq_output_a = output_a[6:, :r16]
    if torch.all(third_seq_output_a != 0):
        print("✓ Third sequence successfully used rank=16")
    else:
        print("✗ Third sequence did not use rank=16")
        print(f"Number of zero values: {torch.sum(third_seq_output_a == 0)}")
    
    print("Basic test completed!\n")


def test_gate_up_multirank():
    """
    Test Gate/Up LoRA multirank functionality
    """
    print("===== Testing Gate/Up LoRA Multirank Functionality =====")
    
    # Test data setup
    bs = 3  # batch size  
    s = 12  # sequence length
    output_dim = 128  # output dimension
    
    # Create three different rank LoRA weights
    r4 = 4
    r8 = 8
    r16 = 16
    max_r = 16
    
    # Input tensor (simulating output from lora_a)
    x = torch.randn(s, 2 * max_r, device="cuda")
    
    # gate_up_lora_b weight tensor
    gate_up_weights = torch.randn(3, 2 * output_dim, max_r, device="cuda")
    
    # Create batch information
    seg_lens = torch.tensor([4, 4, 4], dtype=torch.int32, device="cuda")
    seg_indptr = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device="cuda")
    weight_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")
    
    # Create different rank values
    rank_values = torch.tensor([r4, r8, r16], dtype=torch.int32, device="cuda")
    
    # Create batch info
    batch_info = LoRABatchInfo(
        bs=bs,
        seg_lens=seg_lens,
        seg_indptr=seg_indptr,
        max_len=4,
        weight_indices=weight_indices
    )
    batch_info.rank_values = rank_values
    
    # Run gate_up_lora_b computation
    print("Running Gate/Up LoRA computation...")
    output = gate_up_lora_b_fwd(x, gate_up_weights, batch_info, output_dim)
    
    # Verify results
    print(f"Input tensor shape: {x.shape}")
    print(f"Gate/Up weights shape: {gate_up_weights.shape}")
    print(f"Output shape: {output.shape}")
    print(f"rank_values: {rank_values}")
    
    # Validation: Check if gate output behaves as expected
    # For rank=4 adapter, result should only depend on first 4 elements
    x_copy = x.clone()
    x_copy[:4, 4:] = 0  # Setting beyond rank=4 to zero should not affect output
    output_copy = gate_up_lora_b_fwd(x_copy, gate_up_weights, batch_info, output_dim)
    
    # Check if first sequence output is the same (indicating that elements beyond r4 were not used)
    if torch.allclose(output[:4], output_copy[:4]):
        print("✓ First sequence (rank=4) successfully limited to first 4 elements")
    else:
        print("✗ First sequence rank limitation failed")
        print(f"Maximum difference: {torch.max(torch.abs(output[:4] - output_copy[:4]))}")
    
    print("Gate/Up test completed!\n")


def test_qkv_multirank():
    """
    Test QKV LoRA multirank functionality
    """
    print("===== Testing QKV LoRA Multirank Functionality =====")
    
    # Test data setup
    bs = 3  # batch size
    s = 12  # sequence length
    q_dim = 64  # Q output dimension
    kv_dim = 32  # K/V output dimension
    
    # Create three different rank LoRA weights
    r4 = 4
    r8 = 8
    r16 = 16
    max_r = 16
    
    # Input tensor (simulating output from lora_a)
    x = torch.randn(s, 3 * max_r, device="cuda")
    
    # qkv_lora_b weight tensor
    qkv_weights = torch.randn(3, q_dim + 2 * kv_dim, max_r, device="cuda")
    
    # Output offsets
    output_offset = torch.tensor(
        [0, q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim], 
        dtype=torch.int32, device="cuda"
    )
    
    # Create batch information
    seg_lens = torch.tensor([4, 4, 4], dtype=torch.int32, device="cuda")
    seg_indptr = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device="cuda")
    weight_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")
    
    # Create different rank values
    rank_values = torch.tensor([r4, r8, r16], dtype=torch.int32, device="cuda")
    
    # Create batch info
    batch_info = LoRABatchInfo(
        bs=bs,
        seg_lens=seg_lens,
        seg_indptr=seg_indptr,
        max_len=4,
        weight_indices=weight_indices
    )
    batch_info.rank_values = rank_values
    
    # Run qkv_lora_b computation
    print("Running QKV LoRA computation...")
    max_qkv_dim = max(q_dim, kv_dim)
    output = qkv_lora_b_fwd(x, qkv_weights, batch_info, output_offset, max_qkv_dim)
    
    # Verify results
    print(f"Input tensor shape: {x.shape}")
    print(f"QKV weights shape: {qkv_weights.shape}")
    print(f"Output shape: {output.shape}")
    print(f"rank_values: {rank_values}")
    
    # Similar validation method: create a copy of x, set elements beyond rank to zero, confirm output is unchanged
    x_copy = x.clone()
    # Modify first sequence input, set elements beyond rank=4 to zero
    x_copy[:4, 4:r4+4] = 0  # q part
    x_copy[:4, max_r+4:max_r+r4+4] = 0  # k part
    x_copy[:4, 2*max_r+4:2*max_r+r4+4] = 0  # v part
    
    output_copy = qkv_lora_b_fwd(x_copy, qkv_weights, batch_info, output_offset, max_qkv_dim)
    
    # Check if first sequence output is the same (indicating rank=4 is effective)
    if torch.allclose(output[:4], output_copy[:4], rtol=1e-5, atol=1e-5):
        print("✓ First sequence (rank=4) computation result is correct")
    else:
        print("✗ First sequence computation result is incorrect")
        print(f"Maximum difference: {torch.max(torch.abs(output[:4] - output_copy[:4]))}")
    
    print("QKV test completed!\n")


if __name__ == "__main__":
    # Run all tests
    test_sgemm_multirank()
    test_gate_up_multirank()
    test_qkv_multirank()
    
    print("All tests completed!") 