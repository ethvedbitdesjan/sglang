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
            base_output=base_output,
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

        assert isinstance(unified_v_buffer, torch.Tensor)
        output_dim_q = self.batch_info.output_dim_q
        output_dim_kv = self.batch_info.output_dim_kv
        q_base_output, k_base_output, v_base_output = base_output.split(
            [output_dim_q, output_dim_kv, output_dim_kv], dim=-1
        )

        q_lora_a_output = lora_shrink_fwd(
            x=x,
            weight=unified_k_buffer.view(-1, output_dim_kv),
            batch_info=self.batch_info,
            qkvo=0,
        )
        q_lora_output = lora_expand_fwd(
            x=q_lora_a_output,
            weight=unified_v_buffer.view(-1, output_dim_kv),
            batch_info=self.batch_info,
            feat_out=output_dim_q,
            qkvo=0,
            scale=scaling,
            base_output=q_base_output,
        )

        k_lora_a_output = lora_shrink_fwd(
            x=x,
            weight=unified_k_buffer.view(-1, output_dim_kv),
            batch_info=self.batch_info,
            qkvo=1,
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

        v_lora_a_output = lora_shrink_fwd(
            x=x,
            weight=unified_k_buffer.view(-1, output_dim_kv),
            batch_info=self.batch_info,
            qkvo=2,
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

        lora_output = torch.cat((q_lora_output, k_lora_output, v_lora_output), dim=-1)

        return lora_output
