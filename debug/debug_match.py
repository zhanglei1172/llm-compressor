import contextlib
import copy
import re
from typing import Mapping, Optional, Union
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.hooks import attach_align_device_hook, remove_hook_from_module
from compressed_tensors import get_execution_device
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
    enable_quantization,
    forward_quantize,
)
from datasets import load_dataset, load_from_disk
from easydict import EasyDict
from loguru import logger
from qwen_omni_utils import process_mm_info
from qwen_vl_utils import process_vision_info
from torch.distributed.fsdp import (
    FullStateDictConfig,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as PT_FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)
from trl.trainer.utils import (
    DataCollatorForCompletionOnlyLM,
    DataCollatorForLanguageModeling,
)

from llmcompressor import oneshot
from llmcompressor.core.state import State
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq import mappings as awq_mappings
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings
from llmcompressor.pipelines.sequential.helpers import SequentialTracer
from llmcompressor.recipe import Recipe
from llmcompressor.train.fsdp_trainer import MyTrainer
from llmcompressor.train.train_utils import LLMCTrainingArguments, TeacherModel
from llmcompressor.utils import dispatch_for_generation, helpers
from llmcompressor.utils.pytorch.module import patch_module_non_persistent_buffers

MODEL_ID = "/dataset/workspace/zhangl98/models/Qwen3-Omni-30B-A3B-Instruct/"

mappings.SPINQUANT_MAPPING_REGISTRY["Qwen3OmniMoeThinkerForConditionalGeneration"] = (
    mappings.SpinQuantMapping(
        mm_proj=[r"re:.*audio_tower\.proj2$", r"re:.*visual\.merger.*mlp\.2$"],
        embedding="re:.*embed_tokens$",
        attn_q="re:.*model.*q_proj$",
        attn_k="re:.*model.*k_proj$",
        attn_v="re:.*model.*v_proj$",
        attn_o="re:.*model.*o_proj$",
        mlp_in=[r"re:.*mlp\.gate$"]
        + [
            rf"re:.*model.*\.{i}\.{x}$"
            for x in ["up_proj", "gate_proj"]
            for i in range(128)
        ],
        mlp_out=[rf"re:.*model.*\.{i}\.down_proj$" for i in range(128)],
        lm_head="lm_head",
    )
)
norm_mappings.NORM_MAPPING_REGISTRY["Qwen3OmniMoeThinkerForConditionalGeneration"] = [
    norm_mappings.NormMapping(
        norm="re:.*model.*input_layernorm$",
        linears=["re:.*model.*q_proj$", "re:.*model.*k_proj$", "re:.*model.*v_proj$"],
    ),
    norm_mappings.NormMapping(
        norm="re:.*model.*post_attention_layernorm$",
        linears=[r"re:.*mlp\.gate$"]
        + [
            rf"re:.*model.*\.{i}\.{x}$"
            for x in ["up_proj", "gate_proj"]
            for i in range(128)
        ],
    ),
    norm_mappings.NormMapping(
        norm="model.norm",
        linears=["lm_head"],
    ),
]

recipe_quantization = """
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      ignore: ["re:lm_head", "re:visual.*", "re:model.visual.*", "re:audio_tower.*"]
      config_groups:
        group_0:
          weights:
            observer: mse
            observer_kwargs:
              maxshrink: 0.1
              patience: 10
              averaging_constant: 0.05
              grid: 128.0
              norm: 2.0
            num_bits: 4
            type: int
            symmetric: true
            strategy: channel
          targets: [Linear]
"""

recipe_quarot = """
quant_stage:
  quant_modifiers:
    SpinQuantModifier:
      rotations: ["R1", "R2"]
      transform_block_size_R1: 2048
      learnable: True
      transform_type: "random-hadamard"
"""

recipe_awq = """
quant_stage:
  quant_modifiers:
    # SmoothQuantModifier:
    #   smoothing_strength: 0.8
    AWQModifier:
      ignore: ["re:lm_head", "re:visual.*", "re:model.visual.*", "re:audio_tower.*"]
      mappings:
        -
          smooth_layer: "re:.*input_layernorm$"
          balance_layers: ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"]
        -
          smooth_layer: "re:.*v_proj$"
          balance_layers: ["re:.*o_proj$"]
        - 
          smooth_layer: "re:.*post_attention_layernorm$"
          balance_layers: ["re:.*mlp.experts.*.gate_proj$", "re:.*mlp.experts.*.up_proj$", "re:.*mlp.gate$"]
        - 
          smooth_layer: "re:.*up_proj$"
          balance_layers: ["re:.*down_proj$"]
      config_groups:
        group_0:
          weights:
            observer: mse
            observer_kwargs:
              maxshrink: 0.1
              patience: 10
              averaging_constant: 0.05
              grid: 128.0
              norm: 2.0
            num_bits: 4
            type: int
            symmetric: true
            strategy: channel
          targets: [Linear]
"""


def test_match_quantization(model):
    print("=" * 20)
    state = State()
    state.update(
        model=model,
    )
    recipe_ = Recipe.create_instance(path_or_modifiers=recipe_quantization)
    recipe_.modifiers[0].on_initialize(state=state)
    names = set()
    for name, module in model.named_modules():
        if hasattr(module, "quantization_status"):
            names.add(re.sub(r"\d+", "[]", name))
    for name in names:
        print("Quantize module name: ", name)


def test_match_quarot(model):
    print("=" * 20)
    state = State()
    state.update(
        model=model,
    )
    recipe_ = Recipe.create_instance(path_or_modifiers=recipe_quarot)
    recipe_.modifiers[0].on_initialize(state=state)
    recipe_.modifiers[0].on_start(state=state, event=None)
    print(model)


def test_match_awq(model):
    print("=" * 20)
    state = State()
    state.update(
        model=model,
    )
    recipe_ = Recipe.create_instance(path_or_modifiers=recipe_awq)
    recipe_.modifiers[0].on_initialize(state=state)
    for mapping in recipe_.modifiers[0]._resolved_mappings:
      print(mapping)


if __name__ == "__main__":
    model_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    with init_empty_weights():
        model = Qwen3OmniMoeForConditionalGeneration._from_config(
            model_config,
            # trust_remote_code=True,
            dtype=None,
            # low_cpu_mem_usage=True,
            # attn_implementation=ATTN_IMPL,
        )
    _tmp_config = copy.deepcopy(model.thinker.config)
    _tmp_config.update(model.thinker.config.text_config.to_dict())
    with contextlib.ExitStack() as stack:
        stack.enter_context(helpers.patch_attr(model.thinker, "config", _tmp_config))
        test_match_quantization(model.thinker)
        test_match_awq(model.thinker)
        test_match_quarot(model.thinker)
