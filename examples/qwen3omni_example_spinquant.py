import argparse
import contextlib
import copy
import datetime
import os
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

torch.fx.experimental._config.meta_nonzero_assume_all_nonzero = True
# awq_mappings.AWQ_MAPPING_REGISTRY["Qwen3OmniMoeThinkerForConditionalGeneration"] = awq_mappings._moe_default_mappings
USE_AUDIO_IN_VIDEO = True


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

#################### configurations ####################
calibrate_moe_context = True
# Select model and load it.
pretrain = "ostq"
recipe = "examples/qwen3_omni_configs/text/spinquant.yaml"
flag = "spinquant"
NUM_CALIBRATION_SAMPLES = 256
#################### configurations ####################


if pretrain == "ostq":
    MODEL_ID = "/code/omni_ostq_wa_bf16/transformed_model/"
else:
    MODEL_ID = "/dataset/workspace/zhangl98/models/Qwen3-Omni-30B-A3B-Instruct/"

if calibrate_moe_context:
    flag += "-calmoe"


def pre_compression_thinker(model):
    # session = active_session()
    # session.reset()
    state = State()
    state.update(
        model=model.thinker,
    )
    recipe_ = Recipe.create_instance(path_or_modifiers=recipe, target_stage=None)

    _tmp_config = copy.deepcopy(model.thinker.config)
    _tmp_config.update(model.thinker.config.text_config.to_dict())

    with contextlib.ExitStack() as stack:
        stack.enter_context(helpers.patch_attr(model.thinker, "config", _tmp_config))
        for mod in recipe_.modifiers:
            mod.on_initialize(state=state)
        recipe_.modifiers[0].on_start(state=state, event=None)
        model.thinker.apply(enable_quantization)

    return state, recipe_, model


class DataCollatorForQwen3OmniDataset(DataCollatorForCompletionOnlyLM):
    def __init__(self, processor):
        self.processor = processor
        # Prepare the constants
        self.assistant_start_tokens = processor(text=["<|im_start|>assistant\n"])[
            "input_ids"
        ][0]
        # assert self.assistant_start_tokens == [151644, 77091, 198]
        self.assistant_end_tokens = processor(text=["<|im_end|>\n"])["input_ids"][0]
        # assert self.assistant_end_tokens == [151645, 198]

    def __call__(self, examples):
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": example["image"],
                        },
                        {
                            "type": "text",
                            # "text": example["text"].capitalize(),
                            "text": "What does the image show?",
                        },
                    ],
                }
            ]
            for example in examples
        ]
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        batch = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )

        labels = (
            torch.ones_like(batch["input_ids"], dtype=batch["input_ids"].dtype) * -100
        )

        for batch_idx, token_ids in enumerate(batch["input_ids"].tolist()):
            pos = 0
            is_assistant_response = False
            while pos < len(token_ids):
                if is_assistant_response:
                    if token_ids[pos] == self.assistant_end_tokens[0]:
                        # Check if the assistant response ends
                        is_assistant_end = True
                        for i, assistant_end_token in enumerate(
                            self.assistant_end_tokens[1:], 1
                        ):
                            if token_ids[pos + i] != assistant_end_token:
                                is_assistant_end = False
                                break
                        if is_assistant_end:  # End of the assistant response
                            is_assistant_response = False
                            for i in range(pos, pos + len(self.assistant_end_tokens)):
                                # Update the labels (including the end of the assistant response)
                                labels[batch_idx, i] = token_ids[i]
                            pos += len(self.assistant_end_tokens)
                        else:
                            labels[batch_idx, pos] = token_ids[pos]  # Update the labels
                            pos += 1
                    else:
                        labels[batch_idx, pos] = token_ids[pos]  # Update the labels
                        pos += 1
                else:
                    if token_ids[pos] == self.assistant_start_tokens[0]:
                        is_assistant_start = True
                        for i, assistant_start_token in enumerate(
                            self.assistant_start_tokens[1:], 1
                        ):
                            if token_ids[pos + i] != assistant_start_token:
                                is_assistant_start = False
                                break
                        if is_assistant_start:
                            is_assistant_response = True
                            pos += len(self.assistant_start_tokens)
                        else:
                            pos += 1
                    else:
                        pos += 1

        batch["labels"] = labels
        return batch


def pt_fsdp_state_dict(model: torch.nn.Module):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with PT_FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        return model.state_dict()


def setup():
    # initialize the process group
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def fsdp_main(model, config):
    for name, param in model.named_parameters():
        if param.requires_grad and name.endswith("bias"):
            param.requires_grad = False

    state_dict = model.thinker.state_dict()
    model.thinker.to("meta")
    model_to_train = copy.deepcopy(model.thinker)
    if not RANK_OTHER:
        model_to_train.load_state_dict(state_dict, assign=True)
    del state_dict

    DATASET_ID = "lmms-lab/flickr30k"
    DATASET_SPLIT = "test[:256]"
    # DATASET_SPLIT = "test"
    MAX_SEQUENCE_LENGTH = 2048
    # Select number of samples. 256 samples is a good place to start.
    # Increasing the number of samples can improve accuracy.
    # NUM_CALIBRATION_SAMPLES = 256
    MAX_SEQUENCE_LENGTH = 2048

    # Load dataset and preprocess.
    # ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42)
    # fsdp
    train_processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        model_max_length=config.seq_len,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    train_args = LLMCTrainingArguments(**config.train_args)
    need_teacher = train_args.special.get("loss_type", "origin") not in (
        "origin",
        "DFT",
    )
    if need_teacher:
        model_path = train_args.special.get("teacher_path", MODEL_ID)

        teacher_model, _ = dist_load_model(model_path)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False
        model_to_train.teacher = TeacherModel(teacher_model)
    # Now you can train the model
    trainer = MyTrainer(
        model=model_to_train,
        # tokenizer=train_processor.tokenizer,
        args=train_args,
        train_dataset=ds,
        eval_dataset=None,
        # data_collator=default_data_collator,
        data_collator=DataCollatorForQwen3OmniDataset(train_processor),
        # data_collator=patch.CustomDataCollatorForCompletionOnlyLM([-1], tokenizer=train_tokenizer, pad_to_multiple_of=8),
        # optimizers=(optimizer, None),
        # optimizers=(None, None),
        # ignored_modules=ignored_modules,
    )
    trainer.train()
    dist.barrier()
    state_dict = pt_fsdp_state_dict(model_to_train)
    if not RANK_OTHER:
        model.thinker.load_state_dict(state_dict, assign=True)


def post_compression_thinker(state, recipe_, model, processor):
    recipe_.modifiers[0].on_end(state=state, event=None)
    from collections import OrderedDict

    _h = set()
    transform_state_dict = OrderedDict()
    from compressed_tensors.transform.factory.base import TransformBase

    for name, module in model.thinker.named_modules():
        if isinstance(module, TransformBase):
            if module in _h or id(module.scheme) in _h:
                continue
            _h.add((module if module.scheme.block_wise else id(module.scheme)))
            print(f"{name}: {module}")
            transform_state_dict.update({name: module.state_dict()})

    to_removes = []
    for name, module in model.thinker.named_modules():
        for child_name, child_module in module.named_children():
            if isinstance(child_module, TransformBase):
                to_removes.append((module, child_name))
    for module, child_name in to_removes:
        delattr(module, child_name)
    # Confirm generations of the quantized model look sane.
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    # dispatch_for_generation(model)

    print("==========================================\n\n")

    from compressed_tensors.quantization import QuantizationStatus
    from compressed_tensors.utils.match import match_named_modules

    SAVE_DIR = (
        "/tmp/"
        + MODEL_ID.rstrip("/").split("/")[-1]
        + f"-{pretrain}-{flag}-sym-com-text"
        + "-trans"
    )

    quantized_name_set = set()
    import re

    for _, module in match_named_modules(
        model, recipe_.modifiers[-1].resolved_targets, recipe_.modifiers[-1].ignore
    ):
        if hasattr(module, "quantization_status"):
            assert (
                module.quantization_status == QuantizationStatus.FROZEN
            ), f"{module.quantization_status}"
            quantized_name_set.add(re.sub(r"\d+", "X", _))
            scheme = getattr(module, "quantization_scheme", None)

            delattr(module, "quantization_status")
            delattr(module, "quantization_enabled")
            delattr(module, "quantization_scheme")
            for key in list(module._parameters.keys()):
                if key.endswith("_scale") or key.endswith("_zero_point"):
                    delattr(module, key)
    print(f"Total quantized modules: {quantized_name_set}")
    model.save_pretrained(SAVE_DIR)  # , save_compressed=True) # fakequant
    processor.save_pretrained(SAVE_DIR)
    torch.save(transform_state_dict, f"{SAVE_DIR}/transform_state_dict.pt")
    print(SAVE_DIR)


def dist_load_model(model_path=MODEL_ID, load_processor=False):
    processor = None
    if RANK_OTHER:
        with init_empty_weights():
            model = Qwen3OmniMoeForConditionalGeneration._from_config(
                model_config,
                # trust_remote_code=True,
                dtype=None,
                # low_cpu_mem_usage=True,
                # attn_implementation=ATTN_IMPL,
            )
    else:
        if load_processor:
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto"
        )
    return model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        config = EasyDict(config)
    setup()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    RANK_OTHER = dist.is_initialized() and dist.get_rank() != 0
    model_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if RANK_OTHER:
        logger.remove()
    model, processor = dist_load_model(load_processor=True)
    dist.barrier()
    with patch_module_non_persistent_buffers(model):
        model.eval()
        # no_grad for compression
        model.apply(
            lambda m: m.weight.requires_grad_(False) if hasattr(m, "weight") else None
        )
        state, recipe_, model = pre_compression_thinker(model)
        dist.barrier()
        fsdp_main(model, config)

    if not RANK_OTHER:
        post_compression_thinker(state, recipe_, model, processor)
    cleanup()
