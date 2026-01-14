import argparse
import base64
import contextlib
import copy
import datetime
import functools
import os
import random
from collections import defaultdict
from io import BytesIO
from typing import Mapping, Optional, Union
from unittest.mock import patch

import numpy as np
import soundfile as sf
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
from compressed_tensors.transform import TransformLocation
from compressed_tensors.transform.factory.base import TransformBase
from compressed_tensors.transform.utils.hadamard import random_hadamard_matrix
from compressed_tensors.transform.utils.matrix import apply_transform_weight
from compressed_tensors.utils import remove_dispatch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from easydict import EasyDict
from loguru import logger
from PIL import Image
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
from torch.nn.utils.parametrize import is_parametrized
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)
from trl.trainer.utils import (
    DataCollatorForCompletionOnlyLM,
    DataCollatorForLanguageModeling,
)

from llmcompressor import oneshot
from llmcompressor.core.state import State
from llmcompressor.modeling.qwen3_omni_moe import replace_vit_attention_inv
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq import mappings as awq_mappings
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings
from llmcompressor.pipelines.sequential.helpers import SequentialTracer
from llmcompressor.recipe import Recipe
from llmcompressor.train.fsdp_trainer import MyTrainer
from llmcompressor.train.train_utils import LLMCTrainingArguments, TeacherModel
from llmcompressor.utils import dispatch_for_generation, helpers
from llmcompressor.utils.pytorch.module import (
    build_weight_tied_map_with_unionfind,
    patch_module_non_persistent_buffers,
)

torch.fx.experimental._config.meta_nonzero_assume_all_nonzero = True

USE_AUDIO_IN_VIDEO = True

mappings.SPINQUANT_MAPPING_REGISTRY["Qwen2_5_VLForConditionalGeneration"] = (
    mappings.SpinQuantMapping(
        mm_proj=[r"re:.*visual\.merger.*mlp\.2$"],
        embedding="re:.*embed_tokens$",
        attn="re:.*self_attn$",
        attn_q="re:.*language_model.*q_proj$",
        attn_k="re:.*language_model.*k_proj$",
        attn_v="re:.*language_model.*v_proj$",
        attn_o="re:.*language_model.*o_proj$",
        mlp_in=[
            r"re:.*language_model.*mlp\.up_proj$",
            r"re:.*language_model.*mlp\.gate_proj$",
        ],
        mlp_out=[r"re:.*language_model.*mlp\.down_proj$"],
        lm_head="lm_head",
    )
)
norm_mappings.NORM_MAPPING_REGISTRY["Qwen2_5_VLForConditionalGeneration"] = [
    norm_mappings.NormMapping(
        norm="re:.*language_model.*input_layernorm$",
        linears=[
            "re:.*language_model.*q_proj$",
            "re:.*language_model.*k_proj$",
            "re:.*language_model.*v_proj$",
        ],
    ),
    norm_mappings.NormMapping(
        norm="re:.*language_model.*post_attention_layernorm$",
        linears=[
            r"re:.*language_model.*mlp\.up_proj$",
            r"re:.*language_model.*mlp\.gate_proj$",
        ],
    ),
    norm_mappings.NormMapping(
        norm="model.language_model.norm",
        linears=["lm_head"],
    ),
]

mappings.SPINQUANT_MAPPING_REGISTRY["Qwen2_5_VisionTransformerPretrainedModel"] = (
    mappings.SpinQuantMapping(
        mm_proj=["patch_embed"],
        embedding=[],
        attn="re:.*attn$",
        # embedding="conv_out",
        attn_q="re:.*q_proj$",
        attn_k="re:.*k_proj$",
        attn_v="re:.*v_proj$",
        attn_o=r"re:.*attn\.proj$",
        mlp_in=[r"re:.*mlp\.up_proj$", r"re:.*mlp\.gate_proj$"],
        mlp_out=[r"re:.*mlp\.down_proj$"],
        lm_head=[r"re:merger.*mlp\.0$"],
    )
)
norm_mappings.NORM_MAPPING_REGISTRY["Qwen2_5_VisionTransformerPretrainedModel"] = [
    norm_mappings.NormMapping(
        norm="re:.*norm1$",
        linears=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    norm_mappings.NormMapping(
        norm="re:.*norm2$",
        linears=[r"re:.*mlp\.up_proj$", r"re:.*mlp\.gate_proj$"],
    ),
    norm_mappings.NormMapping(
        norm="re:.*ln_q$",
        linears=[r"re:merger.*mlp\.0$"],
    ),
]

#################### configurations ####################
# Select model and load it.
pretrain = "origin"
flag = "ostquant"
NUM_CALIBRATION_SAMPLES = 256
enable_modality = {
    # "vit",
    "text"
}
#################### configurations ####################

model_dtype = torch.bfloat16

MODEL_ID = "/dataset/620/v1/models/qwenvl_2.5/v3-20260108-131155/checkpoint-30600/"


flag += str(tuple(enable_modality)).replace("'", "").replace(",", "|")

SAVE_DIR = (
    "/tmp/" + MODEL_ID.rstrip("/").split("/")[-1] + f"-{pretrain}-{flag}" + "-trans"
)

MAX_SEQUENCE_LENGTH = 2048
# Load dataset and preprocess.
# ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds_vl = load_dataset(
    "lmms-lab/LLaVA-OneVision-Data", "FigureQA(MathV360K)", split="train[:128]"
)
ds_text = load_dataset("hkust-nlp/deita-6k-v0", split="train[:128]")
ds_wiki = load_from_disk("/dataset/workspace/zhangl98/dataset/calib/wikitext2/")


def encode_base64_img(img) -> str:
    with BytesIO() as buffer:
        img.save(buffer, format="PNG")
        data = buffer.getvalue()

    return base64.b64encode(data).decode("utf-8")


def format_as_vl_messages(example, prompt: str | None = None):
    """Format single example into messages format for TRL."""
    if not prompt:
        prompt = "What does the image show?"
    labels = example["caption"]
    response = labels[0]
    # example["image"] is PIL image, convert it to base64
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image;base64,{encode_base64_img(example['image'])}",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": response}],
        },
    ]

    return {
        "messages": messages,
    }


def format_as_messages(example):
    role_map = {
        "human": "user",
        "gpt": "assistant",
    }
    """Format single example into messages format for TRL."""
    # example["image"] is PIL image, convert it to base64
    messages = []
    for conversation in example["conversations"]:
        message = {
            "role": role_map[conversation["from"]],
            "content": [],
        }
        content = conversation["value"]
        if "<image>" in content:
            parts = content.split("<image>")
            for i, part in enumerate(parts):
                part = part.strip()
                if i < len(parts) - 1:
                    message["content"].append(
                        {
                            "type": "image",
                            "image": f"data:image;base64,{encode_base64_img(example['image'])}",
                            "audio": None,
                        }
                    )
                if part:
                    message["content"].append({"type": "text", "text": part})
        else:
            message["content"].append({"type": "text", "text": content})
        messages.append(message)

    return {
        "messages": messages,
    }


def format_as_text_messages(example, prompt: str | None = None):
    """Format single example into messages format for TRL."""
    problem = example["problem"]
    solution = example["generated_solution"]
    conversations = [
        {"role": "user", "content": [{"type": "text", "text": problem}]},
        {"role": "assistant", "content": [{"type": "text", "text": solution}]},
    ]
    return {"messages": conversations}


ds_vl = ds_vl.map(
    format_as_messages,
    remove_columns=ds_vl.column_names,
    # num_proc=6,
    # fn_kwargs={"prompt": "What does the image show?"},
)

ds_text = ds_text.map(
    format_as_messages,
    remove_columns=ds_text.column_names,
    # num_proc=6,
    # fn_kwargs={"prompt": "Please transcribe the audio."},
)

from datasets import Features, Value

target_features = Features(
    {
        "messages": [
            {
                "content": [
                    {
                        "image": Value(dtype="string"),
                        "text": Value(dtype="string"),
                        "type": Value(dtype="string"),
                    }
                ],
                "role": Value(dtype="string"),
            }
        ]
    }
)

ds = []
if "vit" in enable_modality:
    ds.append(ds_vl)
if "text" in enable_modality:
    ds.append(ds_text)
ds = concatenate_datasets(ds)
ds = ds.shuffle(seed=42)


@torch.no_grad()
def pre_compression_thinker_vit(model):
    from llmcompressor.modeling.qwen3_omni_moe import replace_vit_attention

    replace_vit_attention(model.model.visual)
    state = State()
    state.update(
        model=model.model.visual,
    )
    recipe_ = [
        SpinQuantModifier(
            do_fold=False,
            backe_mean=True,
            learnable=True,
            rotations=["R1", "R2"],
            transform_block_size_R1=1152,
            transform_type="random-hadamard",
            sequential_onload=not RANK_OTHER,
        )
    ]

    _tmp_config = copy.deepcopy(model.model.visual.config)
    _tmp_config.update({"head_dim": _tmp_config.hidden_size // _tmp_config.num_heads})

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            helpers.patch_attr(model.model.visual, "config", _tmp_config)
        )
        for mod in recipe_:
            mod.on_initialize(state=state)
        # for param in model.model.visual.parameters():
        #     param.requires_grad = False
        recipe_[0].on_start(state=state, event=None)

    return state, recipe_, model


@torch.no_grad()
def pre_compression_thinker_text(model):
    # session = active_session()
    # session.reset()
    state = State()
    state.update(
        model=model,
    )
    recipe_ = [
        SpinQuantModifier(
            do_fold=False,
            backe_mean=False,
            learnable=True,
            rotations=["R1", "R2"],
            transform_block_size_R1=3584,
            transform_type="random-hadamard",
            sequential_onload=not RANK_OTHER,
        )
    ]

    with contextlib.ExitStack() as stack:
        for mod in recipe_:
            mod.on_initialize(state=state)
        # for param in model.model.parameters():
        #     param.requires_grad = False
        recipe_[0].on_start(state=state, event=None)

    return state, recipe_, model


@torch.no_grad()
def pre_compression_thinker(model):
    state = State()
    state.update(
        model=model,
    )
    ignore = ["re:.*lm_head"]
    if "vit" not in enable_modality:
        ignore.append("re:.*visual.*")

    if "text" not in enable_modality:
        ignore.append("re:.*language_model.*")
    recipe_ = [
        QuantizationModifier(
            ignore=ignore,
            config_groups={
                "group_0": {
                    "weights": {
                        "observer": "minmax",
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "dynamic": True,
                    },
                    "input_activations": {
                        "observer": "minmax",
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "tensor",
                        "dynamic": True,
                    },
                    "targets": [
                        r"re:.*up_proj$",
                        r"re:.*gate_proj$",
                        r"re:.*q_proj$",
                        r"re:.*k_proj$",
                        r"re:.*v_proj$",
                        r"re:.*o_proj$",
                        r"re:.*out_proj$",
                        r"re:.*attn\.proj$",
                    ],
                    "ste": True,
                },
                "group_1": {
                    "weights": {
                        "observer": "minmax",
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "dynamic": True,
                    },
                    "input_activations": {
                        "observer": "minmax",
                        "num_bits": 16,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "tensor",
                        "dynamic": True,
                    },
                    "targets": [
                        r"re:.*down_proj$",
                        # r"re:.*proj2$",
                    ],
                    "ste": True,
                },
            },
        )
    ]

    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.nn.utils.parametrize.cached())
        for mod in recipe_:
            mod.on_initialize(state=state)
        model.apply(enable_quantization)

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
                    "role": turn["role"],
                    "content": [
                        {k: v for k, v in content.items() if v is not None}
                        for content in turn["content"]
                    ],
                }
                for turn in example["messages"]
            ]
            for example in examples
        ]
        # conversations = [example["messages"] for example in examples]
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=False, tokenize=False
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
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )

        labels = (
            torch.ones_like(batch["input_ids"], dtype=batch["input_ids"].dtype) * -100
        )

        for batch_idx, token_ids in enumerate(batch["input_ids"].tolist()):
            valid = False
            pos = 0
            is_assistant_response = False
            while pos < len(token_ids):
                if is_assistant_response:
                    valid = True
                    if token_ids[pos] == self.assistant_end_tokens[0]:
                        # Check if the assistant response ends
                        is_assistant_end = True
                        for i, assistant_end_token in enumerate(
                            self.assistant_end_tokens[1:], 1
                        ):
                            if (
                                pos + i >= len(token_ids)
                                or token_ids[pos + i] != assistant_end_token
                            ):
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
                            if (
                                pos + i >= len(token_ids)
                                or token_ids[pos + i] != assistant_start_token
                            ):
                                is_assistant_start = False
                                break
                        if is_assistant_start:
                            is_assistant_response = True
                            pos += len(self.assistant_start_tokens)
                        else:
                            pos += 1
                    else:
                        pos += 1
            if not valid:
                labels[batch_idx, :] = token_ids

        batch["labels"] = labels  # batch["input_ids"]
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(dtype=model_dtype)
        return batch


@torch.no_grad()
def post_compression_thinker_vit(model):
    replace_vit_attention_inv(model.model.visual)


@torch.no_grad()
def post_compression_thinker_text(model):
    pass


@torch.no_grad()
def post_compression_thinker(model, processor):
    # recipe_[0].on_end(state=state, event=None)
    from collections import OrderedDict

    _h = set()
    transform_state_dict = OrderedDict()

    for name, module in model.named_modules():
        if isinstance(module, TransformBase):
            if module in _h or id(module.scheme) in _h:
                continue
            _h.add((module if module.scheme.block_wise else id(module.scheme)))
            print(f"{name}: {module}")
            transform_state_dict.update({name: module.state_dict()})

    to_removes = []
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if isinstance(child_module, TransformBase):
                to_removes.append((module, child_name))
    for module, child_name in to_removes:
        delattr(module, child_name)
    # Confirm generations of the quantized model look sane.
    print("\n\n")
    quantized_name_set = set()
    import re

    from compressed_tensors.utils.match import match_named_modules

    for _, module in match_named_modules(
        model, recipe_[-1].resolved_targets, recipe_[-1].ignore
    ):
        if hasattr(module, "quantization_status"):
            quantized_name_set.add(re.sub(r"\d+", "X", _))
            scheme = getattr(module, "quantization_scheme", None)

            delattr(module, "quantization_status")
            delattr(module, "quantization_enabled")
            delattr(module, "quantization_scheme")
            for key in list(module._parameters.keys()):
                if key.endswith("_scale") or key.endswith("_zero_point"):
                    delattr(module, key)
    print(f"Total quantized modules: {quantized_name_set}")
    if "vit" in enable_modality:
        post_compression_thinker_vit(model)

    if "text" in enable_modality:
        post_compression_thinker_text(model)

    model.save_pretrained(SAVE_DIR)  # , save_compressed=True) # fakequant
    processor.save_pretrained(SAVE_DIR)
    torch.save(transform_state_dict, f"{SAVE_DIR}/transform_state_dict.pt")
    print(SAVE_DIR)


def dist_load_model(model_path=MODEL_ID, load_processor=False):
    processor = None
    if RANK_OTHER:
        with init_empty_weights():
            model = Qwen2_5_VLForConditionalGeneration._from_config(
                model_config,
                # trust_remote_code=True,
                dtype=model_dtype,
                # low_cpu_mem_usage=True,
                # attn_implementation=ATTN_IMPL,
            )
    else:
        if load_processor:
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=model_dtype
        )
    return model, processor


def fsdp_main(model, config):
    training_params_cnt = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            training_params_cnt += 1
    weight_tied_name_map = build_weight_tied_map_with_unionfind(model)

    state_dict = model.state_dict()
    model.to("meta")
    model_to_train = copy.deepcopy(model)
    if not RANK_OTHER:
        model_to_train.load_state_dict(state_dict, assign=True)
    del state_dict

    # fsdp
    train_processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        model_max_length=MAX_SEQUENCE_LENGTH,
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
    model_to_train.train()

    if need_teacher:
        model_path = train_args.special.get("teacher_path", MODEL_ID)

        teacher_model, _ = dist_load_model(model_path)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.text_config.use_cache = False

        model_to_train.teacher = TeacherModel(teacher_model)

    model_to_train.config.text_config.use_cache = False  # make activation ckpt
    assert len(set(weight_tied_name_map.values())) == training_params_cnt
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
        weight_tied_name_map=weight_tied_name_map,
        ignored_modules=[],
    )
    trainer.train()
    dist.barrier()
    if hasattr(trainer.model, "_orig_mod"):
        unwrapped_model = trainer.model._orig_mod
    else:
        unwrapped_model = trainer.model
    state_dict = trainer.accelerator.get_state_dict(unwrapped_model)
    if not RANK_OTHER:
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("teacher")
        }
        model.load_state_dict(state_dict, assign=True)
        trainer.register_tied_parameters(model, weight_tied_name_map)


def pt_fsdp_state_dict(model: torch.nn.Module):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with PT_FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        return model.state_dict()


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def setup():
    # initialize the process group
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))


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
    # dist.barrier()
    with patch_module_non_persistent_buffers(model):
        model.eval()
        # no_grad for compression
        for param in model.parameters():
            param.requires_grad = False
        if "vit" in enable_modality:
            state_vit, recipe_vit, model = pre_compression_thinker_vit(model)

        if "text" in enable_modality:
            state_text, recipe_text, model = pre_compression_thinker_text(model)

        state, recipe_, model = pre_compression_thinker(model)
        dist.barrier()
        fsdp_main(model, config)
    torch.cuda.empty_cache()
    dist.barrier()
    if not RANK_OTHER:
        if "vit" in enable_modality:
            recipe_vit[0]._fold_transforms_into_weights(state_vit.model)

        if "text" in enable_modality:
            recipe_text[0]._fold_transforms_into_weights(state_text.model)

        post_compression_thinker(model, processor)
    cleanup()
