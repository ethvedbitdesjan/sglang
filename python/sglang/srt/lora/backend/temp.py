from typing import List

import torch

from sglang.srt.lora.backend import BaseLoRABackend
from sglang.srt.lora.backend.base_backend import (
    get_fuse_output_add_from_name,
    get_fuse_stacked_lora_b_from_name,
)
from sglang.srt.lora.triton_ops.unified_triton_ops.bgmm import (
    lora_expand_fwd,
    lora_shrink_fwd,
)
from sglang.srt.lora.utils import UnifiedLoRABatchInfo

QKV_COUNT = 0
import time
import os
CURRENT_TIME = time.strftime("%Y%m%d-%H%M%S")

class UnifiedTritonLoRABackend:
    def __init__(self, name: str, batch_info: UnifiedLoRABatchInfo = None):
        self.name = name
        self.batch_info = batch_info
        self.fuse_output_scaling_add = get_fuse_output_add_from_name(name)
        self.fuse_stacked_lora_b = get_fuse_stacked_lora_b_from_name(name)

    def set_batch_info(self, batch_info: UnifiedLoRABatchInfo):
        self.batch_info = batch_info

    def run_o_or_down_lora(
        self,
        x: torch.Tensor,
        unified_k_buffer: torch.Tensor,  # A weights from unified memory pool
        unified_v_buffer: torch.Tensor,  # B weights from unified memory pool
        base_output: torch.Tensor = None,
        scaling: torch.Tensor = None,
        *args,
        **kwargs
    ) -> torch.Tensor:

        # x: (s, input_dim)
        # lora_a: (num_lora, r, input_dim)
        # lora_b: (num_lora, output_dim, r)
        # base_output: (s, output_dim)

        assert isinstance(unified_v_buffer, torch.Tensor)
        output_dim_kv = self.batch_info.output_dim_kv
        output_dim_o_or_down = self.batch_info.output_dim_o_or_down
        
        # Clone base_output to avoid modifying the original tensor
        base_output_clone = base_output.clone() if base_output is not None else None
        
        lora_a_output = lora_shrink_fwd(
            x=x,
            weight=unified_k_buffer.view(-1, output_dim_kv),
            batch_info=self.batch_info,
            qkvo=3,
        )
        lora_output = lora_expand_fwd(
            x=lora_a_output,
            weight=unified_v_buffer.view(-1, output_dim_kv),
            batch_info=self.batch_info,
            feat_out=output_dim_o_or_down,
            qkvo=3,
            scale=scaling,
            base_output=base_output_clone,
        )

        return lora_output

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        unified_k_buffer: torch.Tensor,  # A weights from unified memory pool
        unified_v_buffer: torch.Tensor,  # B weights from unified memory pool
        base_output: torch.Tensor = None,
        scaling: torch.Tensor = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        global QKV_COUNT
        global CURRENT_TIME
        # x: (s, input_dim)
        # qkv_lora_a: (num_lora, 3 * r, input_dim)
        # qkv_lora_b: (num_lora, output_dim_q + 2 * output_dim_kv, r)
        # base_output: (s, output_dim_q + 2 * output_dim_kv)
        base_output_clone = base_output.clone() if base_output is not None else None
        assert isinstance(unified_v_buffer, torch.Tensor)
        output_dim_q = self.batch_info.output_dim_q
        output_dim_kv = self.batch_info.output_dim_kv
        if base_output_clone is not None:
            q_base_output = base_output_clone[:, :output_dim_q].clone()
            k_base_output = base_output_clone[:, output_dim_q:output_dim_q+output_dim_kv].clone()
            v_base_output = base_output_clone[:, output_dim_q+output_dim_kv:].clone()
        else:
            q_base_output, k_base_output, v_base_output = None, None, None
            
        lora_start_clone = self.batch_info.lora_start.clone()
        lora_loc_clone = self.batch_info.lora_loc.clone()
        lora_ranks_clone = self.batch_info.lora_ranks.clone()
        unified_k_buffer_clone = unified_k_buffer.clone()
        unified_v_buffer_clone = unified_v_buffer.clone()
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR = os.path.join("/u/vvjain3/sglang_logs/unified/backend", CURRENT_TIME)
            os.makedirs(LOG_DIR, exist_ok=True)
            torch.save(base_output, os.path.join(LOG_DIR, f"base_output_{QKV_COUNT}.pt"))
            torch.save(unified_k_buffer, os.path.join(LOG_DIR, f"unified_k_buffer_{QKV_COUNT}.pt"))
            torch.save(unified_v_buffer, os.path.join(LOG_DIR, f"unified_v_buffer_{QKV_COUNT}.pt"))
            torch.save(x, os.path.join(LOG_DIR, f"x_{QKV_COUNT}.pt"))
            torch.save(self.batch_info.lora_start, os.path.join(LOG_DIR, f"lora_start_{QKV_COUNT}.pt"))
            torch.save(self.batch_info.lora_loc, os.path.join(LOG_DIR, f"lora_loc_{QKV_COUNT}.pt"))
            torch.save(self.batch_info.lora_ranks, os.path.join(LOG_DIR, f"lora_ranks_{QKV_COUNT}.pt"))
            
        q_lora_a_output = lora_shrink_fwd(
            x=x,
            weight=unified_k_buffer.view(-1, output_dim_kv),
            batch_info=self.batch_info,
            qkvo=0,
        )
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR = os.path.join("/u/vvjain3/sglang_logs/unified/backend", CURRENT_TIME)
            os.makedirs(LOG_DIR, exist_ok=True)
            torch.save(q_lora_a_output, os.path.join(LOG_DIR, f"q_lora_a_output_{QKV_COUNT}.pt"))
        if not torch.allclose(self.batch_info.lora_start, lora_start_clone) or not torch.allclose(
            self.batch_info.lora_loc, lora_loc_clone
        ) or not torch.allclose(self.batch_info.lora_ranks, lora_ranks_clone):
            raise ValueError(
                f"""Batch info mismatch: original batch info and cloned batch info are not equal. q lora a output. 
                with max diff start: {torch.max(torch.abs(self.batch_info.lora_start - lora_start_clone))}
                with max diff loc: {torch.max(torch.abs(self.batch_info.lora_loc - lora_loc_clone))}
                with max diff ranks: {torch.max(torch.abs(self.batch_info.lora_ranks - lora_ranks_clone))}"""
            )
        # if not torch.allclose(self.batch_info.lora_start, lora_start_clone) or not torch.allclose(
        #     self.batch_info.lora_loc, lora_loc_clone
        # ) or not torch.allclose(self.batch_info.lora_ranks, lora_ranks_clone):
        #     raise ValueError(
        #         f"""Batch info mismatch: original batch info and cloned batch info are not equal. q lora a output. 
        #         with max diff start: {torch.max(torch.abs(self.batch_info.lora_start - lora_start_clone))}
        #         with max diff loc: {torch.max(torch.abs(self.batch_info.lora_loc - lora_loc_clone))}
        #         with max diff ranks: {torch.max(torch.abs(self.batch_info.lora_ranks - lora_ranks_clone))}"""
        #     )
        if not torch.allclose(unified_k_buffer, unified_k_buffer_clone):
            raise ValueError(
                f"Unified K buffer mismatch: original unified K buffer and cloned unified K buffer are not equal. q lora a output. with max diff: {torch.max(torch.abs(unified_k_buffer - unified_k_buffer_clone))}"
            )
        if not torch.allclose(unified_v_buffer, unified_v_buffer_clone):
            raise ValueError(
                f"Unified V buffer mismatch: original unified V buffer and cloned unified V buffer are not equal. q lora a output. with max diff: {torch.max(torch.abs(unified_v_buffer - unified_v_buffer_clone))}"
            )
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR1 = os.path.join("/u/vvjain3/sglang_logs/unified/backend_finegrained", CURRENT_TIME)
            os.makedirs(LOG_DIR1, exist_ok=True)
            torch.save(q_lora_a_output,os.path.join(LOG_DIR1, f"q_lora_a_output_{QKV_COUNT}.pt"), )
            torch.save(q_base_output, os.path.join(LOG_DIR1, f"q_base_output_{QKV_COUNT}.pt"))
            
        q_lora_output = lora_expand_fwd(
            x=q_lora_a_output,
            weight=unified_v_buffer.view(-1, output_dim_kv),
            batch_info=self.batch_info,
            feat_out=output_dim_q,
            qkvo=0,
            scale=scaling,
            base_output=q_base_output,
        )
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR = os.path.join("/u/vvjain3/sglang_logs/unified/backend", CURRENT_TIME)
            os.makedirs(LOG_DIR, exist_ok=True)
            torch.save(q_lora_output, os.path.join(LOG_DIR, f"q_lora_output_{QKV_COUNT}.pt"))
        if not torch.allclose(self.batch_info.lora_start, lora_start_clone) or not torch.allclose(
            self.batch_info.lora_loc, lora_loc_clone
        ) or not torch.allclose(self.batch_info.lora_ranks, lora_ranks_clone):
            raise ValueError(
                f"""Batch info mismatch: original batch info and cloned batch info are not equal. q lora a output. 
                with max diff start: {torch.max(torch.abs(self.batch_info.lora_start - lora_start_clone))}
                with max diff loc: {torch.max(torch.abs(self.batch_info.lora_loc - lora_loc_clone))}
                with max diff ranks: {torch.max(torch.abs(self.batch_info.lora_ranks - lora_ranks_clone))}"""
            )
        if not torch.allclose(unified_k_buffer, unified_k_buffer_clone):
            raise ValueError(
                f"Unified K buffer mismatch: original unified K buffer and cloned unified K buffer are not equal. q lora a output. with max diff: {torch.max(torch.abs(unified_k_buffer - unified_k_buffer_clone))}"
            )
        if not torch.allclose(unified_v_buffer, unified_v_buffer_clone):
            raise ValueError(
                f"Unified V buffer mismatch: original unified V buffer and cloned unified V buffer are not equal. q lora a output. with max diff: {torch.max(torch.abs(unified_v_buffer - unified_v_buffer_clone))}"
            )
            
        k_lora_a_output = lora_shrink_fwd(
            x=x,
            weight=unified_k_buffer.view(-1, output_dim_kv),
            batch_info=self.batch_info,
            qkvo=1,
        )
        
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR = os.path.join("/u/vvjain3/sglang_logs/unified/backend", CURRENT_TIME)
            os.makedirs(LOG_DIR, exist_ok=True)
            torch.save(k_lora_a_output, os.path.join(LOG_DIR, f"k_lora_a_output_{QKV_COUNT}.pt"))
        if not torch.allclose(self.batch_info.lora_start, lora_start_clone) or not torch.allclose(
            self.batch_info.lora_loc, lora_loc_clone
        ) or not torch.allclose(self.batch_info.lora_ranks, lora_ranks_clone):
            raise ValueError(
                f"""Batch info mismatch: original batch info and cloned batch info are not equal. q lora a output. 
                with max diff start: {torch.max(torch.abs(self.batch_info.lora_start - lora_start_clone))}
                with max diff loc: {torch.max(torch.abs(self.batch_info.lora_loc - lora_loc_clone))}
                with max diff ranks: {torch.max(torch.abs(self.batch_info.lora_ranks - lora_ranks_clone))}"""
            )
        if not torch.allclose(unified_k_buffer, unified_k_buffer_clone):
            raise ValueError(
                f"Unified K buffer mismatch: original unified K buffer and cloned unified K buffer are not equal. q lora a output. with max diff: {torch.max(torch.abs(unified_k_buffer - unified_k_buffer_clone))}"
            )
        if not torch.allclose(unified_v_buffer, unified_v_buffer_clone):
            raise ValueError(
                f"Unified V buffer mismatch: original unified V buffer and cloned unified V buffer are not equal. q lora a output. with max diff: {torch.max(torch.abs(unified_v_buffer - unified_v_buffer_clone))}"
            )
            
        k_lora_output = lora_expand_fwd(
            x=k_lora_a_output,
            weight=unified_v_buffer.view(-1, output_dim_kv),
            batch_info=self.batch_info,
            feat_out=output_dim_kv,
            qkvo=1,
            scale=scaling,
            base_output=k_base_output,
        )
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR = os.path.join("/u/vvjain3/sglang_logs/unified/backend", CURRENT_TIME)
            os.makedirs(LOG_DIR, exist_ok=True)
            torch.save(k_lora_output, os.path.join(LOG_DIR, f"k_lora_output_{QKV_COUNT}.pt"))
        if not torch.allclose(self.batch_info.lora_start, lora_start_clone) or not torch.allclose(
            self.batch_info.lora_loc, lora_loc_clone
        ) or not torch.allclose(self.batch_info.lora_ranks, lora_ranks_clone):
            raise ValueError(
                f"""Batch info mismatch: original batch info and cloned batch info are not equal. q lora a output. 
                with max diff start: {torch.max(torch.abs(self.batch_info.lora_start - lora_start_clone))}
                with max diff loc: {torch.max(torch.abs(self.batch_info.lora_loc - lora_loc_clone))}
                with max diff ranks: {torch.max(torch.abs(self.batch_info.lora_ranks - lora_ranks_clone))}"""
            )
        if not torch.allclose(unified_k_buffer, unified_k_buffer_clone):
            raise ValueError(
                f"Unified K buffer mismatch: original unified K buffer and cloned unified K buffer are not equal. q lora a output. with max diff: {torch.max(torch.abs(unified_k_buffer - unified_k_buffer_clone))}"
            )
        if not torch.allclose(unified_v_buffer, unified_v_buffer_clone):
            raise ValueError(
                f"Unified V buffer mismatch: original unified V buffer and cloned unified V buffer are not equal. q lora a output. with max diff: {torch.max(torch.abs(unified_v_buffer - unified_v_buffer_clone))}"
            )
            
        v_lora_a_output = lora_shrink_fwd(
            x=x,
            weight=unified_k_buffer.view(-1, output_dim_kv),
            batch_info=self.batch_info,
            qkvo=2,
        )
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR = os.path.join("/u/vvjain3/sglang_logs/unified/backend", CURRENT_TIME)
            os.makedirs(LOG_DIR, exist_ok=True)
            torch.save(v_lora_a_output, os.path.join(LOG_DIR, f"v_lora_a_output_{QKV_COUNT}.pt"))
        if not torch.allclose(self.batch_info.lora_start, lora_start_clone) or not torch.allclose(
            self.batch_info.lora_loc, lora_loc_clone
        ) or not torch.allclose(self.batch_info.lora_ranks, lora_ranks_clone):
            raise ValueError(
                f"""Batch info mismatch: original batch info and cloned batch info are not equal. q lora a output. 
                with max diff start: {torch.max(torch.abs(self.batch_info.lora_start - lora_start_clone))}
                with max diff loc: {torch.max(torch.abs(self.batch_info.lora_loc - lora_loc_clone))}
                with max diff ranks: {torch.max(torch.abs(self.batch_info.lora_ranks - lora_ranks_clone))}"""
            )
        if not torch.allclose(unified_k_buffer, unified_k_buffer_clone):
            raise ValueError(
                f"Unified K buffer mismatch: original unified K buffer and cloned unified K buffer are not equal. q lora a output. with max diff: {torch.max(torch.abs(unified_k_buffer - unified_k_buffer_clone))}"
            )
        if not torch.allclose(unified_v_buffer, unified_v_buffer_clone):
            raise ValueError(
                f"Unified V buffer mismatch: original unified V buffer and cloned unified V buffer are not equal. q lora a output. with max diff: {torch.max(torch.abs(unified_v_buffer - unified_v_buffer_clone))}"
            )
        v_lora_output = lora_expand_fwd(
            x=v_lora_a_output,
            weight=unified_v_buffer.view(-1, output_dim_kv),
            batch_info=self.batch_info,
            feat_out=output_dim_kv,
            qkvo=2,
            scale=scaling,
            base_output=v_base_output,
        )
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR = os.path.join("/u/vvjain3/sglang_logs/unified/backend", CURRENT_TIME)
            os.makedirs(LOG_DIR, exist_ok=True)
            torch.save(v_lora_output, os.path.join(LOG_DIR, f"v_lora_output_{QKV_COUNT}.pt"))
        if not torch.allclose(self.batch_info.lora_start, lora_start_clone) or not torch.allclose(
            self.batch_info.lora_loc, lora_loc_clone
        ) or not torch.allclose(self.batch_info.lora_ranks, lora_ranks_clone):
            raise ValueError(
                f"""Batch info mismatch: original batch info and cloned batch info are not equal. q lora a output. 
                with max diff start: {torch.max(torch.abs(self.batch_info.lora_start - lora_start_clone))}
                with max diff loc: {torch.max(torch.abs(self.batch_info.lora_loc - lora_loc_clone))}
                with max diff ranks: {torch.max(torch.abs(self.batch_info.lora_ranks - lora_ranks_clone))}"""
            )
        if not torch.allclose(unified_k_buffer, unified_k_buffer_clone):
            raise ValueError(
                f"Unified K buffer mismatch: original unified K buffer and cloned unified K buffer are not equal. q lora a output. with max diff: {torch.max(torch.abs(unified_k_buffer - unified_k_buffer_clone))}"
            )
        if not torch.allclose(unified_v_buffer, unified_v_buffer_clone):
            raise ValueError(
                f"Unified V buffer mismatch: original unified V buffer and cloned unified V buffer are not equal. q lora a output. with max diff: {torch.max(torch.abs(unified_v_buffer - unified_v_buffer_clone))}"
            )
        lora_output = torch.cat((q_lora_output, k_lora_output, v_lora_output), dim=-1)

        QKV_COUNT += 1
        return lora_output
