# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Integrates "S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
# and "Punica: Multi-Tenant LoRA Serving"

import logging
from typing import Dict, List, Set, Tuple

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.hf_transformers_utils import AutoConfig
from sglang.srt.lora.backend import BaseLoRABackend, get_backend_from_name
from sglang.srt.lora.layers import get_lora_layer,get_unified_lora_layer
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.mem_cache.lora_unified_memory_pool import LoraUnifiedMemoryPool
from sglang.srt.hf_transformers_utils import AutoConfig
from sglang.srt.lora.utils import (
    UnifiedLoRABatchInfo,
    LoRAType,
    get_customized_names_from_hf_names,
    get_layer_id,
    get_stacked_name,
    get_weight_name,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import replace_submodule

logger = logging.getLogger(__name__)


class UnifiedLoRAManager:
    def __init__(
        self,
        base_model: torch.nn.Module,
        lora_paths: Dict[str, str],
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        load_config: LoadConfig,
        dtype: torch.dtype,
        lora_unified_memory_pool: LoraUnifiedMemoryPool,
        lora_backend: str = "unified_triton",
    ):
        self.base_model: torch.nn.Module = base_model
        self.lora_paths: Dict[str, str] = lora_paths
        self.base_hf_config: AutoConfig = base_hf_config
        self.max_loras_per_batch: int = max_loras_per_batch
        self.load_config: LoadConfig = load_config
        self.dtype: torch.dtype = dtype

        # LoRA backend for running sgemm kernels
        logger.info(f"Using {lora_backend} as backend of LoRA kernels.")
        backend_type = get_backend_from_name(lora_backend)
        self.lora_backend: UnifiedTritonLoRABackend = backend_type(lora_backend)

        self.memory_pool = lora_unified_memory_pool

        self.init_loras()

    def init_loras(self):
        # Config of each LoRA adapter
        self.configs: Dict[str, LoRAConfig] = {}

        # Target module names in huggingface lora configs.
        # e.g., {"k_proj", "q_proj", "v_proj", "o_proj"}
        self.hf_target_names: Set[str] = set()
        for name, path in self.lora_paths.items():
            self.configs[name] = LoRAConfig(path)
            self.hf_target_names = set(self.hf_target_names) | set(
                self.configs[name].target_modules
            )

        # Target lora weight names for lora_a and lora_b modules repectively.
        # e.g., {("qkv_proj", "q_proj"), ("qkv_proj", "kv_proj")}
        self.lora_weight_names: Set[Tuple[str]] = set(
            [get_stacked_name(module) for module in self.hf_target_names]
        )

        # load all weights to cpu
        self.loras: Dict[str, LoRAAdapter] = {}
        for name in self.lora_paths.keys():
            lora_adapter = LoRAAdapter(
                name,
                self.configs[name],
                self.base_hf_config,
                self.load_config,
                self.lora_backend,
            )
            lora_adapter.initialize_weights()
            self.loras[name] = lora_adapter

        # misc lora configs
        self.max_lora_dim: int = max([x.hf_config["r"] for x in self.configs.values()])
        self.scaling: torch.Tensor = torch.tensor([adapter.scaling for adapter in list(self.loras.values())], dtype=torch.float16, device="cuda")
        
        # Convert original model layers to layers with LoRA
        self.convert_to_lora_layers()

    def prepare_lora_batch(self, forward_batch: ForwardBatch):
        # load active loras into lora memory pool
        cur_uids = set(forward_batch.lora_paths)
        assert len(cur_uids) <= self.max_loras_per_batch
        self.memory_pool.prepare_lora_batch(cur_uids, self.loras)

        # FIXME: Handle lora uid with None more safely
        if cur_uids == set([None]):
            return

        # set up batch info shared by all lora moruldes
        bs = forward_batch.batch_size
        seg_lens = (
            forward_batch.extend_seq_lens
            if forward_batch.forward_mode.is_extend()
            else torch.ones(bs, device="cuda")
        )
        seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
        max_len = int(torch.max(seg_lens))
        weight_indices = torch.empty((bs,), dtype=torch.int64, device="cuda")
        for i, lora_path in enumerate(forward_batch.lora_paths):
            weight_indices[i] = self.memory_pool.get_buffer_id(lora_path)

        lora_loc,lora_start,lora_ranks = self.memory_pool.get_adapter_memory_info()
        batch_info = UnifiedLoRABatchInfo(
            bs=bs,
            seg_lens=seg_lens,
            seg_indptr=seg_indptr,
            max_len=max_len,
            weight_indices=weight_indices,
            max_lora_dim = self.max_lora_dim,
            hidden_size = self.base_hf_config.hidden_size,
            output_dim_q = self.base_hf_config.hidden_size,
            output_dim_kv = self.base_hf_config.hidden_size // (self.base_hf_config.num_attention_heads // self.base_hf_config.num_key_value_heads),
            lora_loc = lora_loc,
            lora_start = lora_start,
            lora_ranks = lora_ranks
        )
        self.lora_backend.set_batch_info(batch_info)

        # call set_lora_info for each lora modules
        for module_name, module in self.lora_modules:
            layer_id = get_layer_id(module_name)
            if "qkv_proj" not in module_name:
                # gate_up_proj,down_proj
                weight_name = get_weight_name(
                    module_name, self.lora_weight_names, LoRAType.LORA_A
                )
                unified_k_buffer,unified_v_buffer = self.memory_pool.get_unified_memory_pool(layer_id = layer_id)
                module.set_unified_lora_info(
                    unified_k_buffer,
                    unified_v_buffer
                )
            else:
                unified_k_buffer,unified_v_buffer = self.memory_pool.get_unified_memory_pool(layer_id = layer_id)
                module.set_unified_lora_info(
                    unified_k_buffer,
                    unified_v_buffer
                )

    def set_lora_module(self, module_name, module):
        lora_module = get_unified_lora_layer(
            layer = module,
            scaling = self.scaling,
            lora_backend=self.lora_backend
        )
        replace_submodule(self.base_model, module_name, lora_module)
        return lora_module

    def convert_to_lora_layers(self):
        # Target module names of customized layers defined in python/sglang/srt/layers
        # e.g., {"qkv_proj", "o_proj"}
        customized_target_names = get_customized_names_from_hf_names(
            self.hf_target_names, self.base_model
        )

        # Monkey patch to use the LoRA version layers
        self.lora_modules: List[Tuple[str, torch.nn.Module]] = []
        for module_name, module in self.base_model.named_modules():
            # The module should be converted if it is included in target_names
            if module_name.split(".")[-1] in customized_target_names:
                self.lora_modules.append(
                    (module_name, self.set_lora_module(module_name, module))
                )
