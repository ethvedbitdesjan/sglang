import torch
from torch import nn

from sglang.srt.lora.backend.unified_triton_backend import UnifiedTritonLoRABackend
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
# from sglang.srt.lora.backend import BaseLoRABackend
from sglang.srt.lora.backend import UnifiedTritonLoRABackend


class BaseLayerWithUnifiedLoRA(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        scaling: torch.Tensor,
        lora_backend: UnifiedTritonLoRABackend,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        self.scaling: float = scaling
        self.set_lora: bool = False
        self.lora_backend: UnifiedTritonLoRABackend = lora_backend

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args):
        pass


class VocabParallelEmbeddingWithUnifiedLoRA(BaseLayerWithUnifiedLoRA):
    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
        scaling: torch.Tensor,
        lora_backend: UnifiedTritonLoRABackend,
    ) -> None:
        super().__init__(base_layer, scaling, lora_backend)
        self.weight = base_layer.weight


class ColumnParallelLinearWithUnifiedLoRA(BaseLayerWithUnifiedLoRA):
    def __init__(
        self,
        base_layer: ColumnParallelLinear,
        scaling: torch.Tensor,
        lora_backend: UnifiedTritonLoRABackend,
    ) -> None:
        super().__init__(base_layer, scaling, lora_backend)

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output, "scaling": self.scaling}
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            lora_a_output,
            self.B_buffer[0],
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_scaling_add
            else base_output + lora_output * self.scaling
        )

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in ColumnParallelLinear
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_)

        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias


class MergedColumnParallelLinearWithUnifiedLoRA(ColumnParallelLinearWithUnifiedLoRA):
    def __init__(
        self,
        base_layer: MergedColumnParallelLinear,
        scaling: torch.Tensor,
        lora_backend: UnifiedTritonLoRABackend,
    ) -> None:
        super().__init__(base_layer, scaling, lora_backend)

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_gate_up = A_buffer
        if self.lora_backend.fuse_stacked_lora_b:
            # B_buffer_gate_up: (num_lora, 2 * output_dim, r)
            self.B_buffer_gate_up = torch.cat(
                (B_buffer[0], B_buffer[1]), dim=-2
            ).contiguous()
        else:
            self.B_buffer_gate_up = (B_buffer[0], B_buffer[1])

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output, "scaling": self.scaling}

        lora_output = self.lora_backend.run_gate_up_lora(
            x,
            self.A_buffer_gate_up,
            self.B_buffer_gate_up,
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_scaling_add
            else base_output + lora_output * self.scaling
        )

class QKVParallelLinearWithUnifiedLoRA(ColumnParallelLinearWithUnifiedLoRA):
    def __init__(
        self,
        base_layer: QKVParallelLinear,
        scaling: torch.Tensor,
        lora_backend: UnifiedTritonLoRABackend,
    ) -> None:
        super().__init__(base_layer, scaling, lora_backend)

    def set_lora_info(
        self,
        unified_k_buffer: torch.Tensor,
        unified_v_buffer: torch.Tensor,
        ):
        self.unified_k_buffer = unified_k_buffer
        self.unified_v_buffer = unified_v_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output, "scaling": self.scaling}
        lora_output = self.lora_backend.run_qkv_lora(
            x,
            self.unified_k_buffer,
            self.unified_v_buffer,
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_scaling_add
            else base_output + lora_output * self.scaling
        )


class RowParallelLinearWithUnifiedLoRA(BaseLayerWithUnifiedLoRA):
    def __init__(
        self,
        base_layer: RowParallelLinear,
        scaling: torch.Tensor,
        lora_backend: UnifiedTritonLoRABackend,
    ) -> None:
        super().__init__(base_layer, scaling, lora_backend)

    def set_lora_info(self, A_buffer: torch.Tensor, B_buffer: torch.Tensor):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output, "scaling": self.scaling}
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            lora_a_output,
            self.B_buffer[0],
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_scaling_add
            else base_output + lora_output * self.scaling
        )

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in RowParallelLinear
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_parallel
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_parallel)

        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

def get_unified_lora_layer(
    layer: nn.Module, scaling: torch.Tensor, lora_backend: UnifiedTritonLoRABackend
) -> BaseLayerWithUnifiedLoRA:
    supported_layer_types = {
        # the order matters
        VocabParallelEmbedding: VocabParallelEmbeddingWithUnifiedLoRA,
        QKVParallelLinear: QKVParallelLinearWithUnifiedLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithUnifiedLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithUnifiedLoRA,
        RowParallelLinear: RowParallelLinearWithUnifiedLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(layer, scaling, lora_backend)
            return ret
    raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")