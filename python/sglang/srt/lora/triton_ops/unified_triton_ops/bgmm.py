import torch

import triton
import triton.language as tl
import math
import torch.nn.functional as F

from sglang.srt.lora.utils import LoRABatchInfo, UnifiedLoRABatchInfo

if triton.__version__ >= "2.1.0":
    @triton.jit
    def _expand_fwd_kernel(
        X, W, scale, B_Loc, B_Lora_Start_Loc, B_Lora_Ranks, B_Start_Loc, B_Seqlen, B_Indicies,
        Out,
        qkvo,
        stride_xbs, stride_xh,
        stride_wbs, stride_wh,
        stride_obs, stride_oh,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_RANK: tl.constexpr,
        TILE_N: tl.constexpr
    ):
        cur_batch = tl.program_id(0)
        cur_tile = tl.program_id(1)
        start_m = tl.program_id(2)
        cur_adapter = tl.load(B_Indicies + cur_batch)

        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_rank_size = tl.load(B_Lora_Ranks + cur_adapter) // 4
        cur_batch_adapter_start_index = tl.load(B_Lora_Start_Loc + cur_adapter) + cur_batch_rank_size * qkvo
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
        cur_batch_scale = tl.load(scale + cur_adapter)

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_RANK)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_x = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_xbs + offs_d[None, :] * stride_xh
        x = tl.load(X + off_x, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

        for start_n in range(cur_tile * TILE_N, (cur_tile+1)*TILE_N, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute xw ----
            w_loc = tl.load(B_Loc + cur_batch_adapter_start_index + ((start_n + offs_n)*cur_batch_rank_size//BLOCK_DMODEL), mask=(start_n + offs_n) < BLOCK_DMODEL, other=0)
            off_w = w_loc[None, :] * stride_wbs + (((start_n + offs_n)*cur_batch_rank_size+offs_d[:, None])%BLOCK_DMODEL) * stride_wh
            w = tl.load(W + off_w, mask=offs_d[:, None] < cur_batch_rank_size, other=0.0)
            
            off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + (start_n + offs_n[None, :]) * stride_oh
            out_ptrs = Out + off_o
            wx = tl.load(out_ptrs, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

            wx += tl.dot(x, w) * cur_batch_scale

            tl.store(out_ptrs, wx, mask=offs_m[:, None] < cur_batch_seq_len)

        return
    
    #TODO please fix the case <kv_head_num != attn_head_num> @Chaobo Jia
    @triton.jit
    def _shrink_fwd_kernel(
        X, W, B_Loc, B_Lora_Start_Loc, B_Lora_Ranks, B_Start_Loc, B_Seqlen, B_Indicies,
        Out,
        qkvo,
        stride_xbs, stride_xh,
        stride_wbs, stride_wh,
        stride_obs, stride_oh,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        start_n = tl.program_id(1)
        start_m = tl.program_id(2)
        cur_adapter = tl.load(B_Indicies + cur_batch)

        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_rank_size = tl.load(B_Lora_Ranks + cur_adapter) // 4
        cur_batch_adapter_start_index = tl.load(B_Lora_Start_Loc + cur_adapter) + cur_batch_rank_size * qkvo
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_x = (cur_batch_in_all_start_index + offs_m) * stride_xbs

        offs_k = tl.arange(0, BLOCK_K)
        
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        w_loc = tl.load(B_Loc + cur_batch_adapter_start_index + offs_n, mask=offs_n < cur_batch_rank_size, other=0)
        off_w = w_loc * stride_wbs
        
        wx = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        
        for start_k in range(0, BLOCK_DMODEL, BLOCK_K):
            start_k = tl.multiple_of(start_k, BLOCK_K)
            # -- compute xw ----
            x = tl.load(X + off_x[:, None] + (start_k+offs_k[None, :]) * stride_xh, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
            w = tl.load(W + off_w[None, :] + (start_k+offs_k[:, None]) * stride_wh, mask=offs_n[None, :] < cur_batch_rank_size, other=0.0)
            wx += tl.dot(x, w)
        
        c = wx.to(tl.float16)
        # initialize pointers to output
        off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + offs_n[None, :] * stride_oh
        out_ptrs = Out + off_o
        tl.store(out_ptrs, c, mask=offs_m[:, None] < cur_batch_seq_len)

        return

    @torch.inference_mode()
    def lora_expand_fwd(
        x: torch.Tensor,
        weight: torch.Tensor, 
        batch_info: UnifiedLoRABatchInfo,
        feat_out: torch.Tensor,
        qkvo: int, 
        scale: torch.Tensor,
        base_output: torch.Tensor = None,
        ) -> torch.Tensor:
        # good for large input_len (prefill stage) better than bgmv, worse than cutlass
        BLOCK_N = 128
        N = 1
        TILE = N * BLOCK_N
        BLOCK_M = 32
        # BLOCK_N = 16
        # N = 32
        # TILE = N * BLOCK_N
        # BLOCK_M = 16

        S = x.shape[0]
        H = feat_out

        max_input_len = batch_info.max_len
        max_rank = batch_info.max_lora_dim
        batch = batch_info.bs

        lora_loc = batch_info.lora_loc
        lora_start = batch_info.lora_start
        lora_ranks = batch_info.lora_ranks

        start_loc = batch_info.seg_indptr
        seq_len = batch_info.seg_lens
        weight_indicies = batch_info.weight_indices

        grid = (batch, triton.cdiv(feat_out, TILE), triton.cdiv(max_input_len, BLOCK_M))  # batch, head,

        if base_output is None:
            output = torch.empty((S, N), device=x.device, dtype=x.dtype)
            fuse_scaling_add = False
        else:
            output = base_output
            fuse_scaling_add = True

        num_warps = 4
        _expand_fwd_kernel[grid](
            x, weight, scale, lora_loc, lora_start, lora_ranks, start_loc, seq_len, weight_indicies,
            output,
            qkvo,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=feat_out,
            BLOCK_N=BLOCK_N,
            BLOCK_RANK=max_rank,
            TILE_N=TILE,
            num_warps=num_warps,
            num_stages=2,
        )
        return output
    
    @torch.inference_mode()
    def lora_shrink_fwd(
        x: torch.Tensor,
        weight: torch.Tensor,
        batch_info: UnifiedLoRABatchInfo,
        qkvo: int
        ) -> torch.Tensor:

        # good for large input_len (prefill stage) better than bgmv, worse than cutlass
        assert len(x.shape) == 2
        assert len(weight.shape) == 2

        S = x.shape[0]
        max_input_len = batch_info.max_len
        max_rank = batch_info.max_lora_dim
        batch = batch_info.bs
        hidden_size = batch_info.hidden_size

        lora_loc = batch_info.lora_loc
        lora_start = batch_info.lora_start
        lora_ranks = batch_info.lora_ranks

        start_loc = batch_info.seg_indptr
        seq_len = batch_info.seg_lens
        weight_indicies = batch_info.weight_indices

        BLOCK_R = 16 if max_rank > 8 else max_rank
        BLOCK_S = 32
        BLOCK_K = 128

        grid = (batch, triton.cdiv(max_rank, BLOCK_R), triton.cdiv(max_input_len, BLOCK_S))  # batch, head,

        output = torch.empty((S, max_rank), device=x.device, dtype=x.dtype)
        num_warps = 4
        _shrink_fwd_kernel[grid](
            x, weight, lora_loc, lora_start, lora_ranks, start_loc, seq_len, weight_indicies,
            output,
            qkvo,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=BLOCK_S,
            BLOCK_DMODEL=hidden_size,
            BLOCK_N=BLOCK_R,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=1,
        )
        return output