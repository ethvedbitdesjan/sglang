import torch
import numpy as np
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.lora.triton_ops.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.srt.lora.triton_ops.sgemm_lora_b import sgemm_lora_b_fwd
from sglang.srt.lora.triton_ops.gate_up_lora_b import gate_up_lora_b_fwd
from sglang.srt.lora.triton_ops.qkv_lora_b import qkv_lora_b_fwd

def test_sgemm_multirank():
    """
    测试基本的LoRA multirank功能
    """
    print("===== 测试基本 LoRA multirank 功能 =====")
    
    # 测试数据设置
    bs = 2  # 批大小
    s = 10  # 序列长度
    input_dim = 64  # 输入维度
    
    # 创建三个不同rank的LoRA权重
    r4 = 4
    r8 = 8 
    r16 = 16
    max_r = 16
    
    # 输入张量
    x = torch.randn((s, input_dim), device="cuda")
    
    # 权重张量 - 三个不同rank的权重
    weights_a = torch.randn((3, max_r, input_dim), device="cuda")
    # 第一个适配器只用到r4
    weights_a[0, r4:, :] = 0
    # 第二个适配器用到r8
    weights_a[1, r8:, :] = 0
    
    weights_b = torch.randn((3, input_dim, max_r), device="cuda")
    # 第一个适配器只用到r4
    weights_b[0, :, r4:] = 0
    # 第二个适配器用到r8
    weights_b[1, :, r8:] = 0
    
    # 创建批次信息
    seg_lens = torch.tensor([3, 3, 4], dtype=torch.int32, device="cuda")
    seg_indptr = torch.tensor([0, 3, 6, 10], dtype=torch.int32, device="cuda")
    weight_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")  # 分别使用三个不同的权重
    
    # 创建不同的rank值
    rank_values = torch.tensor([r4, r8, r16], dtype=torch.int32, device="cuda")
    
    # 创建批次信息
    batch_info = LoRABatchInfo(
        bs=bs+1,  # 注意这里是3个请求
        seg_lens=seg_lens,
        seg_indptr=seg_indptr,
        max_len=4,
        weight_indices=weight_indices
    )
    # 设置rank值
    batch_info.rank_values = rank_values
    
    # 运行LoRA计算
    print("运行LoRA A和B计算...")
    output_a = sgemm_lora_a_fwd(x, weights_a, batch_info)
    output_b = sgemm_lora_b_fwd(output_a, weights_b, batch_info)
    
    # 验证结果
    print(f"输入张量形状: {x.shape}")
    print(f"LoRA A 权重形状: {weights_a.shape}")
    print(f"LoRA B 权重形状: {weights_b.shape}")
    print(f"LoRA A 输出形状: {output_a.shape}")
    print(f"LoRA B 输出形状: {output_b.shape}")
    print(f"rank_values: {rank_values}")
    
    # 验证第一个序列只用了r4的部分
    first_seq_output_a = output_a[:3, r4:]
    if torch.allclose(first_seq_output_a, torch.zeros_like(first_seq_output_a)):
        print("✓ 第一个序列成功限制使用 rank=4")
    else:
        print("✗ 第一个序列未能限制使用 rank=4")
        print(f"非零值数量: {torch.sum(first_seq_output_a != 0)}")
    
    # 验证第二个序列用了r8的部分
    second_seq_output_a = output_a[3:6, :r8]
    if torch.all(second_seq_output_a != 0):
        print("✓ 第二个序列成功使用 rank=8")
    else:
        print("✗ 第二个序列未能成功使用 rank=8")
        print(f"零值数量: {torch.sum(second_seq_output_a == 0)}")
    
    # 验证第二个序列没用超过r8的部分
    second_seq_r8_plus = output_a[3:6, r8:]
    if torch.allclose(second_seq_r8_plus, torch.zeros_like(second_seq_r8_plus)):
        print("✓ 第二个序列成功限制使用 rank=8")
    else:
        print("✗ 第二个序列未能限制使用 rank=8") 
        print(f"非零值数量: {torch.sum(second_seq_r8_plus != 0)}")
    
    # 验证第三个序列用了完整的r16
    third_seq_output_a = output_a[6:, :r16]
    if torch.all(third_seq_output_a != 0):
        print("✓ 第三个序列成功使用 rank=16")
    else:
        print("✗ 第三个序列未能成功使用 rank=16")
        print(f"零值数量: {torch.sum(third_seq_output_a == 0)}")
    
    print("基本测试完成!\n")


def test_gate_up_multirank():
    """
    测试Gate/Up LoRA multirank功能
    """
    print("===== 测试 Gate/Up LoRA multirank 功能 =====")
    
    # 测试数据设置
    bs = 3  # 批大小  
    s = 12  # 序列长度
    output_dim = 128  # 输出维度
    
    # 创建三个不同rank的LoRA权重
    r4 = 4
    r8 = 8
    r16 = 16
    max_r = 16
    
    # 输入张量（模拟从lora_a输出的结果）
    x = torch.randn(s, 2 * max_r, device="cuda")
    
    # gate_up_lora_b权重张量
    gate_up_weights = torch.randn(3, 2 * output_dim, max_r, device="cuda")
    
    # 创建批次信息
    seg_lens = torch.tensor([4, 4, 4], dtype=torch.int32, device="cuda")
    seg_indptr = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device="cuda")
    weight_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")
    
    # 创建不同的rank值
    rank_values = torch.tensor([r4, r8, r16], dtype=torch.int32, device="cuda")
    
    # 创建批次信息
    batch_info = LoRABatchInfo(
        bs=bs,
        seg_lens=seg_lens,
        seg_indptr=seg_indptr,
        max_len=4,
        weight_indices=weight_indices
    )
    batch_info.rank_values = rank_values
    
    # 运行gate_up_lora_b计算
    print("运行Gate/Up LoRA计算...")
    output = gate_up_lora_b_fwd(x, gate_up_weights, batch_info, output_dim)
    
    # 验证结果
    print(f"输入张量形状: {x.shape}")
    print(f"Gate/Up 权重形状: {gate_up_weights.shape}")
    print(f"输出形状: {output.shape}")
    print(f"rank_values: {rank_values}")
    
    # 验证：输出的gate部分是否符合预期
    # 对于rank=4的适配器，计算结果应该只依赖前4个元素
    x_copy = x.clone()
    x_copy[:4, 4:] = 0  # 将超出rank=4的部分置零不应影响输出
    output_copy = gate_up_lora_b_fwd(x_copy, gate_up_weights, batch_info, output_dim)
    
    # 检查第一个序列的输出是否相同（说明超出r4的部分确实没有被使用）
    if torch.allclose(output[:4], output_copy[:4]):
        print("✓ 第一个序列（rank=4）成功限制只使用前4个元素")
    else:
        print("✗ 第一个序列未能成功限制rank")
        print(f"差异的最大值: {torch.max(torch.abs(output[:4] - output_copy[:4]))}")
    
    print("Gate/Up 测试完成!\n")


def test_qkv_multirank():
    """
    测试QKV LoRA multirank功能
    """
    print("===== 测试 QKV LoRA multirank 功能 =====")
    
    # 测试数据设置
    bs = 3  # 批大小
    s = 12  # 序列长度
    q_dim = 64  # Q输出维度
    kv_dim = 32  # K/V输出维度
    
    # 创建三个不同rank的LoRA权重
    r4 = 4
    r8 = 8
    r16 = 16
    max_r = 16
    
    # 输入张量（模拟从lora_a输出的结果）
    x = torch.randn(s, 3 * max_r, device="cuda")
    
    # qkv_lora_b权重张量
    qkv_weights = torch.randn(3, q_dim + 2 * kv_dim, max_r, device="cuda")
    
    # 输出偏移量
    output_offset = torch.tensor(
        [0, q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim], 
        dtype=torch.int32, device="cuda"
    )
    
    # 创建批次信息
    seg_lens = torch.tensor([4, 4, 4], dtype=torch.int32, device="cuda")
    seg_indptr = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device="cuda")
    weight_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")
    
    # 创建不同的rank值
    rank_values = torch.tensor([r4, r8, r16], dtype=torch.int32, device="cuda")
    
    # 创建批次信息
    batch_info = LoRABatchInfo(
        bs=bs,
        seg_lens=seg_lens,
        seg_indptr=seg_indptr,
        max_len=4,
        weight_indices=weight_indices
    )
    batch_info.rank_values = rank_values
    
    # 运行qkv_lora_b计算
    print("运行QKV LoRA计算...")
    max_qkv_dim = max(q_dim, kv_dim)
    output = qkv_lora_b_fwd(x, qkv_weights, batch_info, output_offset, max_qkv_dim)
    
    # 验证结果
    print(f"输入张量形状: {x.shape}")
    print(f"QKV 权重形状: {qkv_weights.shape}")
    print(f"输出形状: {output.shape}")
    print(f"rank_values: {rank_values}")
    
    # 同样的验证方法：创建x的副本，超出rank的部分置零，确认输出不变
    x_copy = x.clone()
    # 修改第一个序列的输入，超出rank=4的部分置零
    x_copy[:4, 4:r4+4] = 0  # q部分
    x_copy[:4, max_r+4:max_r+r4+4] = 0  # k部分
    x_copy[:4, 2*max_r+4:2*max_r+r4+4] = 0  # v部分
    
    output_copy = qkv_lora_b_fwd(x_copy, qkv_weights, batch_info, output_offset, max_qkv_dim)
    
    # 检查第一个序列的输出是否相同（说明rank=4起作用）
    if torch.allclose(output[:4], output_copy[:4], rtol=1e-5, atol=1e-5):
        print("✓ 第一个序列（rank=4）计算结果正确")
    else:
        print("✗ 第一个序列计算结果错误")
        print(f"差异的最大值: {torch.max(torch.abs(output[:4] - output_copy[:4]))}")
    
    print("QKV 测试完成!\n")


if __name__ == "__main__":
    # 运行所有测试
    test_sgemm_multirank()
    test_gate_up_multirank()
    test_qkv_multirank()
    
    print("所有测试完成!") 