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
        
        # Create copies of inputs and weights to avoid reference sharing
        x_clone = x.detach().clone()
        
        # Create copies of views to avoid shared storage
        k_buffer_view = unified_k_buffer.view(-1, output_dim_kv).detach().clone()
        v_buffer_view = unified_v_buffer.view(-1, output_dim_kv).detach().clone()
        
        # If there is base_output, create a copy
        base_output_clone = base_output.detach().clone() if base_output is not None else None
        
        lora_a_output = lora_shrink_fwd(
            x=x_clone,
            weight=k_buffer_view,
            batch_info=self.batch_info,
            qkvo=3,
        )
        lora_output = lora_expand_fwd(
            x=lora_a_output,
            weight=v_buffer_view,
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
        # x: (s, input_dim)
        # qkv_lora_a: (num_lora, 3 * r, input_dim)
        # qkv_lora_b: (num_lora, output_dim_q + 2 * output_dim_kv, r)
        # base_output: (s, output_dim_q + 2 * output_dim_kv)
        global QKV_COUNT
        global CURRENT_TIME
        
        assert isinstance(unified_v_buffer, torch.Tensor)
        output_dim_q = self.batch_info.output_dim_q
        output_dim_kv = self.batch_info.output_dim_kv
        
        # Create independent copies of inputs to avoid shared storage
        x_q = x.detach().clone()
        x_k = x.detach().clone()
        x_v = x.detach().clone()
        
        # Create independent views of weight buffers
        k_buffer_view_q = unified_k_buffer.view(-1, output_dim_kv).detach().clone()
        k_buffer_view_k = unified_k_buffer.view(-1, output_dim_kv).detach().clone()
        k_buffer_view_v = unified_k_buffer.view(-1, output_dim_kv).detach().clone()
        
        v_buffer_view_q = unified_v_buffer.view(-1, output_dim_kv).detach().clone()
        v_buffer_view_k = unified_v_buffer.view(-1, output_dim_kv).detach().clone()
        v_buffer_view_v = unified_v_buffer.view(-1, output_dim_kv).detach().clone()
        
        # Prepare independent views of base_output
        if base_output is not None:
            base_output_full = base_output.detach().clone()
            q_base_output = base_output_full[:, :output_dim_q].detach().clone()
            k_base_output = base_output_full[:, output_dim_q:output_dim_q+output_dim_kv].detach().clone()
            v_base_output = base_output_full[:, output_dim_q+output_dim_kv:].detach().clone()
        else:
            q_base_output, k_base_output, v_base_output = None, None, None

        # if QKV_COUNT in [0, 1, 1056, 1057]:
        #     LOG_DIR = os.path.join("~/sglang_logs/unified/backend", CURRENT_TIME)
        #     os.makedirs(LOG_DIR, exist_ok=True)
        #     torch.save(base_output, os.path.join(LOG_DIR, f"base_output_{QKV_COUNT}.pt"))
        #     torch.save(unified_k_buffer, os.path.join(LOG_DIR, f"unified_k_buffer_{QKV_COUNT}.pt"))
        #     torch.save(unified_v_buffer, os.path.join(LOG_DIR, f"unified_v_buffer_{QKV_COUNT}.pt"))
        #     torch.save(x, os.path.join(LOG_DIR, f"x_{QKV_COUNT}.pt"))
        #     torch.save(self.batch_info.lora_start, os.path.join(LOG_DIR, f"lora_start_{QKV_COUNT}.pt"))
        #     torch.save(self.batch_info.lora_loc, os.path.join(LOG_DIR, f"lora_loc_{QKV_COUNT}.pt"))
        #     torch.save(self.batch_info.lora_ranks, os.path.join(LOG_DIR, f"lora_ranks_{QKV_COUNT}.pt"))
            
        # Q processing - use independent tensor copies
        q_lora_a_output = lora_shrink_fwd(
            x=x_q,
            weight=k_buffer_view_q,
            batch_info=self.batch_info,
            qkvo=0,
        )
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR = os.path.join("~/sglang_logs/unified/backend", CURRENT_TIME)
            os.makedirs(LOG_DIR, exist_ok=True)
            torch.save(q_lora_a_output, os.path.join(LOG_DIR, f"q_lora_a_output_{QKV_COUNT}.pt"))
            
        q_lora_output = lora_expand_fwd(
            x=q_lora_a_output,
            weight=v_buffer_view_q,
            batch_info=self.batch_info,
            feat_out=output_dim_q,
            qkvo=0,
            scale=scaling,
            base_output=q_base_output,
        )
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR = os.path.join("~/sglang_logs/unified/backend", CURRENT_TIME)
            os.makedirs(LOG_DIR, exist_ok=True)
            torch.save(q_lora_output, os.path.join(LOG_DIR, f"q_lora_output_{QKV_COUNT}.pt"))
        # K processing - use independent tensor copies
        k_lora_a_output = lora_shrink_fwd(
            x=x_k,
            weight=k_buffer_view_k,
            batch_info=self.batch_info,
            qkvo=1,
        )
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR = os.path.join("~/sglang_logs/unified/backend", CURRENT_TIME)
            os.makedirs(LOG_DIR, exist_ok=True)
            torch.save(k_lora_a_output, os.path.join(LOG_DIR, f"k_lora_a_output_{QKV_COUNT}.pt"))
        k_lora_output = lora_expand_fwd(
            x=k_lora_a_output,
            weight=v_buffer_view_k,
            batch_info=self.batch_info,
            feat_out=output_dim_kv,
            qkvo=1,
            scale=scaling,
            base_output=k_base_output,
        )
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR = os.path.join("~/sglang_logs/unified/backend", CURRENT_TIME)
            os.makedirs(LOG_DIR, exist_ok=True)
            torch.save(k_lora_output, os.path.join(LOG_DIR, f"k_lora_output_{QKV_COUNT}.pt"))
        # V processing - use independent tensor copies
        v_lora_a_output = lora_shrink_fwd(
            x=x_v,
            weight=k_buffer_view_v,
            batch_info=self.batch_info,
            qkvo=2,
        )
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR = os.path.join("~/sglang_logs/unified/backend", CURRENT_TIME)
            os.makedirs(LOG_DIR, exist_ok=True)
            torch.save(v_lora_a_output, os.path.join(LOG_DIR, f"v_lora_a_output_{QKV_COUNT}.pt"))
        v_lora_output = lora_expand_fwd(
            x=v_lora_a_output,
            weight=v_buffer_view_v,
            batch_info=self.batch_info,
            feat_out=output_dim_kv,
            qkvo=2,
            scale=scaling,
            base_output=v_base_output,
        )
        if QKV_COUNT in [0, 1, 1056, 1057]:
            LOG_DIR = os.path.join("~/sglang_logs/unified/backend", CURRENT_TIME)
            os.makedirs(LOG_DIR, exist_ok=True)
            torch.save(v_lora_output, os.path.join(LOG_DIR, f"v_lora_output_{QKV_COUNT}.pt"))
        # Merge outputs - ensure using independent tensor copies
        lora_output = torch.cat([
            q_lora_output.detach().clone(), 
            k_lora_output.detach().clone(), 
            v_lora_output.detach().clone()
        ], dim=-1)
        QKV_COUNT += 1
        return lora_output
