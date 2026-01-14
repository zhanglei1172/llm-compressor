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
# awq_mappings.AWQ_MAPPING_REGISTRY["Qwen3OmniMoeThinkerForConditionalGeneration"] = awq_mappings._moe_default_mappings
USE_AUDIO_IN_VIDEO = True


mappings.SPINQUANT_MAPPING_REGISTRY["Qwen3OmniMoeThinkerForConditionalGeneration"] = (
    mappings.SpinQuantMapping(
        mm_proj=[r"re:.*audio_tower\.proj2$", r"re:.*visual\.merger.*mlp\.2$"],
        embedding="re:.*embed_tokens$",
        attn="re:.*self_attn$",
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

mappings.SPINQUANT_MAPPING_REGISTRY["Qwen3OmniMoeVisionEncoder"] = (
    mappings.SpinQuantMapping(
        mm_proj=["patch_embed.proj"],
        embedding="pos_embed",
        attn="re:.*attn$",
        # embedding="conv_out",
        attn_q="re:.*q_proj$",
        attn_k="re:.*k_proj$",
        attn_v="re:.*v_proj$",
        attn_o=r"re:.*attn\.proj$",
        mlp_in=["re:.*linear_fc1$"],
        mlp_out=["re:.*linear_fc2$"],
        lm_head=[r"re:merger.*mlp\.0$"],
    )
)
norm_mappings.NORM_MAPPING_REGISTRY["Qwen3OmniMoeVisionEncoder"] = [
    norm_mappings.NormMapping(
        norm="re:.*norm1$",
        linears=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    norm_mappings.NormMapping(
        norm="re:.*norm2$",
        linears=["re:.*fc1$"],
    ),
    norm_mappings.NormMapping(
        norm="re:.*ln_q$",
        linears=[r"re:merger.*mlp\.0$"],
    ),
]

mappings.SPINQUANT_MAPPING_REGISTRY["Qwen3OmniMoeAudioEncoder"] = (
    mappings.SpinQuantMapping(
        mm_proj=["conv_out"],
        embedding="re:.*positional_embedding$",
        attn="re:.*self_attn$",
        # embedding="conv_out",
        attn_q="re:.*q_proj$",
        attn_k="re:.*k_proj$",
        attn_v="re:.*v_proj$",
        attn_o="re:.*out_proj$",
        mlp_in=["re:.*fc1$"],
        mlp_out=["re:.*fc2$"],
        lm_head="proj1",
    )
)
norm_mappings.NORM_MAPPING_REGISTRY["Qwen3OmniMoeAudioEncoder"] = [
    norm_mappings.NormMapping(
        norm="re:.*self_attn_layer_norm$",
        linears=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    norm_mappings.NormMapping(
        norm="re:.*final_layer_norm$",
        linears=["re:.*fc1$"],
    ),
    norm_mappings.NormMapping(
        norm="ln_post",
        linears=["proj1"],
    ),
]

#################### configurations ####################
# Select model and load it.
pretrain = "origin"
flag = "dartquant"
NUM_CALIBRATION_SAMPLES = 256
enable_modality = {
    # "vit",
    "aut",
    # "text"
}
#################### configurations ####################

model_dtype = torch.bfloat16

MODEL_ID = "/dataset/workspace/zhangl98/models/Qwen3-Omni-30B-A3B-Instruct/"


flag += str(tuple(enable_modality)).replace("'", "")

SAVE_DIR = (
    "/tmp/" + MODEL_ID.rstrip("/").split("/")[-1] + f"-{pretrain}-{flag}" + "-trans"
)

MAX_SEQUENCE_LENGTH = 2048
# Load dataset and preprocess.
# ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds_vl = load_dataset(
    "lmms-lab/LLaVA-OneVision-Data", "FigureQA(MathV360K)", split="train[:128]"
)
ds_al = load_dataset(
    "/dataset/workspace/zhangl98/dataset/peoples_speech/test", split="test[:128]"
)
ds_text = load_dataset("hkust-nlp/deita-6k-v0", split="train[:128]")
ds_wiki = load_from_disk("/dataset/workspace/zhangl98/dataset/calib/wikitext2/")


def encode_base64_img(img) -> str:
    with BytesIO() as buffer:
        img.save(buffer, format="PNG")
        data = buffer.getvalue()

    return base64.b64encode(data).decode("utf-8")


def encode_base64_audio(audio_array: np.ndarray, sampling_rate: int) -> str:
    with BytesIO() as buffer:
        sf.write(buffer, audio_array, samplerate=sampling_rate, format="WAV")
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


def format_as_al_messages(example, prompt: str | None = None):
    """Format single example into messages format for TRL."""
    if not prompt:
        prompt = "Please transcribe the audio."
    labels = example["text"]
    # example["audio"]["array"] is numpy array, convert it to base64
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio": f"data:audio/wav;base64,{encode_base64_audio(example['audio']['array'], example['audio']['sampling_rate'])}",
                    "image": None,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": labels.capitalize()}],
        },
    ]
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

ds_al = ds_al.map(
    format_as_al_messages,
    remove_columns=ds_al.column_names,
    # num_proc=6,
    fn_kwargs={"prompt": "Please transcribe the audio."},
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
                        "audio": Value(dtype="string"),
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
if "aut" in enable_modality:
    ds.append(ds_al)
if "text" in enable_modality:
    ds.append(ds_text)
ds = concatenate_datasets(ds)
ds = ds.shuffle(seed=42)


@torch.no_grad()
def pre_compression_thinker_vit(model):
    from llmcompressor.modeling.qwen3_omni_moe import replace_vit_attention

    replace_vit_attention(model.thinker.visual)
    state = State()
    state.update(
        model=model.thinker.visual,
    )
    recipe_ = [
        SpinQuantModifier(
            do_fold=False,
            backe_mean=True,
            learnable=True,
            rotations=["R1", "R2"],
            transform_block_size_R1=1152,
            transform_type="identity",
            sequential_onload=True,
        )
    ]

    _tmp_config = copy.deepcopy(model.thinker.visual.config)
    _tmp_config.update({"head_dim": _tmp_config.hidden_size // _tmp_config.num_heads})

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            helpers.patch_attr(model.thinker.visual, "config", _tmp_config)
        )
        for mod in recipe_:
            mod.on_initialize(state=state)
        # for param in model.thinker.visual.parameters():
        #     param.requires_grad = False
        recipe_[0].on_start(state=state, event=None)

    return state, recipe_, model


@torch.no_grad()
def pre_compression_thinker_aut(model):
    from llmcompressor.modeling.qwen3_omni_moe import replace_audio_embedding

    replace_audio_embedding(model.thinker.audio_tower)
    model.thinker.audio_tower.positional_embedding.positional_embedding = (
        model.thinker.audio_tower.positional_embedding.weight
    )
    # session = active_session()
    # session.reset()
    state = State()
    state.update(
        model=model.thinker.audio_tower,
    )
    recipe_ = [
        SpinQuantModifier(
            do_fold=False,
            backe_mean=True,
            learnable=True,
            rotations=["R1", "R2"],
            transform_block_size_R1=1280,
            transform_type="identity",
            sequential_onload=True,
        )
    ]

    _tmp_config = copy.deepcopy(model.thinker.audio_tower.config)
    _tmp_config.update(
        {"head_dim": _tmp_config.d_model // _tmp_config.encoder_attention_heads}
    )

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            helpers.patch_attr(model.thinker.audio_tower, "config", _tmp_config)
        )
        for mod in recipe_:
            mod.on_initialize(state=state)
        # for param in model.thinker.audio_tower.parameters():
        #     param.requires_grad = False
        recipe_[0].on_start(state=state, event=None)

    return state, recipe_, model


@torch.no_grad()
def pre_compression_thinker_text(model):
    # session = active_session()
    # session.reset()
    state = State()
    state.update(
        model=model.thinker,
    )
    recipe_ = [
        SpinQuantModifier(
            do_fold=False,
            backe_mean=False,
            learnable=True,
            rotations=["R1", "R2"],
            transform_block_size_R1=2048,
            transform_type="identity",
            sequential_onload=True,
        )
    ]
    _tmp_config = copy.deepcopy(model.thinker.config)
    _tmp_config.update(model.thinker.config.text_config.to_dict())

    with contextlib.ExitStack() as stack:
        stack.enter_context(helpers.patch_attr(model.thinker, "config", _tmp_config))
        for mod in recipe_:
            mod.on_initialize(state=state)
        # for param in model.thinker.model.parameters():
        #     param.requires_grad = False
        recipe_[0].on_start(state=state, event=None)

    return state, recipe_, model


@torch.no_grad()
def pre_compression_thinker_text_sequential(model):
    from compressed_tensors.utils import remove_dispatch

    def preprocess(example):
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
        ]
        # conversations = [example["messages"] for example in examples]
        text = processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        return processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )

    ds = ds_vl.map(preprocess, remove_columns=ds_vl.column_names)

    def data_collator(batch):
        assert len(batch) == 1
        return {
            key: torch.tensor(
                value, dtype=model_dtype if key == "pixel_values" else None
            )
            for key, value in batch[0].items()
        }

    original_init = SequentialTracer.__init__

    def my_init(self, ancestors, offloaded):
        original_init(
            self,
            ancestors,
            offloaded,
        )
        # Force onload all modules.
        device = get_execution_device(model)
        remove_hook_from_module(model.thinker.visual.pos_embed, recurse=True)
        model.thinker.visual.pos_embed.to(device)
        for module in model.thinker.visual.pos_embed.modules():
            if module in self.offloaded:
                self.offloaded.remove(module)

    state = State()
    state.update(
        model=model.thinker,
    )

    # for param in model.thinker.model.parameters():
    #     param.requires_grad = False

    recipe_ = [
        SpinQuantModifier(
            do_fold=False,
            backe_mean=False,
            learnable=True,
            rotations=["R1", "R2"],
            transform_block_size_R1=2048,
            transform_type="identity",
        )
    ]
    _tmp_config = copy.deepcopy(model.thinker.config)
    _tmp_config.update(model.thinker.config.text_config.to_dict())

    ori_save = model.save_pretrained
    with contextlib.ExitStack() as stack:
        stack.enter_context(helpers.patch_attr(SequentialTracer, "__init__", my_init))
        stack.enter_context(helpers.patch_attr(model.thinker, "config", _tmp_config))
        oneshot(
            model=model.thinker,
            processor=model.config._name_or_path,
            dataset=ds,
            recipe=recipe_,
            tie_word_embeddings=True,
            data_collator=data_collator,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=1,
            sequential_targets=["Qwen3OmniMoeThinkerTextDecoderLayer"],
        )
        remove_dispatch(model)
    model.save_pretrained = ori_save

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
            if not valid:
                labels[batch_idx, :] = token_ids

        # batch["labels"] = labels  # batch["input_ids"]
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(dtype=model_dtype)
        return batch


@torch.no_grad()
def post_compression_thinker_vit(model):
    replace_vit_attention_inv(model.thinker.visual)


@torch.no_grad()
def post_compression_thinker_aut(model):
    delattr(model.thinker.audio_tower.positional_embedding, "positional_embedding")


@torch.no_grad()
def post_compression_thinker_text(model):
    pass


@torch.no_grad()
def post_compression_thinker(model, processor):
    # recipe_[0].on_end(state=state, event=None)
    from collections import OrderedDict

    _h = set()
    transform_state_dict = OrderedDict()

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

    if "vit" in enable_modality:
        post_compression_thinker_vit(model)
    if "aut" in enable_modality:
        post_compression_thinker_aut(model)
    if "text" in enable_modality:
        post_compression_thinker_text(model)

    model.save_pretrained(SAVE_DIR)  # , save_compressed=True) # fakequant
    processor.save_pretrained(SAVE_DIR)
    torch.save(transform_state_dict, f"{SAVE_DIR}/transform_state_dict.pt")
    print(SAVE_DIR)


def dist_load_model(model_path=MODEL_ID, load_processor=False):
    processor = None

    if load_processor:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto"
    )
    return model, processor


def regist_hook(model, stat_tensors):
    def stat_transform_input_hook(m, x, y):
        if isinstance(x, tuple):  # TransformLocation.INPUT
            x = x[0]
        ori_shape = x.shape
        x = x.view(-1, ori_shape[-1])
        # Deduplicate by original tensor data pointer and store CPU copy
        src_id = (
            float(x.max().item()),
            float(x.min().item()),
            float(x.mean().item()),
        )  # x.data_ptr()
        d = stat_tensors[id(m.weight)]
        if src_id not in d:
            d[src_id] = x.detach().cpu()

    def stat_linear_input_hook(m, x, y, subm):
        if isinstance(x, tuple):
            x = x[0]
        ori_shape = x.shape
        x = x.view(-1, ori_shape[-1])

        # Deduplicate by original tensor data pointer and store CPU copy
        src_id = (
            float(x.max().item()),
            float(x.min().item()),
            float(x.mean().item()),
        )  # x.data_ptr()
        d = stat_tensors[id(subm.weight)]
        if src_id not in d:
            d[src_id] = x.detach().cpu()

    hooks = []
    for name, m in model.thinker.named_modules():
        if is_parametrized(m):
            for name, subm in m.named_children():
                if name.endswith("_weight_input"):  # TransformLocation.WEIGHT_OUTPUT
                    break
            else:
                continue
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_linear_input_hook, subm=subm)
                )
            )
        elif (
            isinstance(m, TransformBase) and m.args.location == TransformLocation.INPUT
        ):
            hooks.append(m.register_forward_hook(stat_transform_input_hook))
    return hooks


def inference_model(model):
    dispatch_for_generation(model.thinker)
    train_processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        model_max_length=MAX_SEQUENCE_LENGTH,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    dataloader = DataLoader(
        ds,
        batch_size=1,
        collate_fn=DataCollatorForQwen3OmniDataset(train_processor),
        pin_memory=True,
        persistent_workers=False,
    )

    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch = {k: v.to(model.thinker.device) for k, v in batch.items()}
            _ = model.thinker(**batch)
            # stat_tensors holds CPU tensors already; nothing to cast per-batch.
    remove_dispatch(model.thinker)


def train_rotate(
    name,
    transform_module,
    train_datas,
    optim="sgd",
    lr=1.5e-3,
    mom=0.9,
    cos_lr=False,
    ep=10,
    bsz=64,
    accumulation_steps=2,
    val_ratio: float = 0.1,
):
    class R_QR(torch.nn.Module):
        def __init__(self, size: int, location, module_type):
            super(R_QR, self).__init__()
            self.size = size
            self.matrix = torch.nn.Parameter(torch.eye(size))
            self.location = location
            self.module_type = module_type

        def forward(self, x):
            self.rotate, _ = torch.linalg.qr(self.matrix, mode="complete")
            # o_x = torch.matmul(x, self.rotate)
            o_x = apply_transform_weight(
                self.rotate.to(device=x.device, dtype=torch.float64),
                x.to(dtype=torch.float64),
                self.location,
                self.module_type,
            )
            return o_x

    device = "cuda" if torch.cuda.is_available() else "cpu"
    construct_device = device
    precision = transform_module.weight.dtype
    size = transform_module.weight.shape[0]
    location = TransformLocation.INPUT
    module_type = torch.nn.Linear
    data = random_hadamard_matrix(size, precision, construct_device)
    data = data.to(device=device)
    _scale = torch.tensor(size, dtype=precision, device=device).sqrt()
    data = data / _scale

    R = R_QR(size=size, location=location, module_type=module_type).to(device)
    R.matrix.data = data

    if optim == "sgd":
        optimizer = torch.optim.SGD(R.parameters(), lr=lr, momentum=mom)
    elif optim == "adam":
        optimizer = torch.optim.Adam(
            R.parameters(),
            lr=lr,
        )
    else:
        raise NotImplementedError

    if cos_lr:  # 设置余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ep, eta_min=0
        )

    R.train()

    print(f"---> start training R of layer {name} ")

    def pad_collate(batch):
        # batch: list of tensors [L_i, hidden]
        lengths = [t.shape[0] for t in batch]
        max_len = max(lengths)
        hidden = batch[0].shape[-1]
        out = torch.zeros((len(batch), max_len, hidden), dtype=batch[0].dtype)
        mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, t in enumerate(batch):
            out[i, : t.shape[0], :] = t
            mask[i, : t.shape[0]] = 1
        return out, mask

    train_dataloader = DataLoader(
        train_datas,
        batch_size=bsz,
        shuffle=True,
        pin_memory=True,
        collate_fn=pad_collate,
    )
    # Prepare validation dataloader if a validation split is requested
    val_dataloader = None
    if val_ratio and 0.0 < val_ratio < 1.0:
        n = len(train_datas)
        if n > 0:
            val_size = max(1, int(n * val_ratio))
            indices = list(range(n))
            random.shuffle(indices)
            val_indices = set(indices[:val_size])
            train_indices = indices[val_size:]
            train_list = [train_datas[i] for i in train_indices]
            val_list = [train_datas[i] for i in sorted(val_indices)]
            train_dataloader = DataLoader(
                train_list,
                batch_size=bsz,
                shuffle=True,
                pin_memory=True,
                collate_fn=pad_collate,
            )
            val_dataloader = DataLoader(
                val_list,
                batch_size=bsz,
                shuffle=False,
                pin_memory=True,
                collate_fn=pad_collate,
            )
    for epoch in range(ep):
        loss_log = []

        for batch_idx, batch in enumerate(train_dataloader):
            batch_samples, mask = batch
            batch_samples = batch_samples.to(device)
            mask = mask.to(device)
            outputs = R(batch_samples)
            # outputs shape: (B, L, hidden) -> per-token score: (B, L)
            per_token = torch.sum(torch.exp((-outputs.abs())), dim=-1)
            # mask: bool -> float
            mask_f = mask.to(dtype=per_token.dtype)
            # avoid division by zero
            denom = mask_f.sum()
            if denom.item() == 0:
                denom = torch.tensor(
                    1.0, device=per_token.device, dtype=per_token.dtype
                )
            loss = (per_token * mask_f).sum() / denom / accumulation_steps
            loss.backward()

            # 如果达到指定的累计步数，更新梯度并清零
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_log.append(loss.detach())

        if cos_lr:
            scheduler.step()

        # 计算平均损失 并打印日志
        if len(loss_log) == 0:
            mean_loss = torch.tensor(0.0, device=device)
        else:
            mean_loss = torch.stack(loss_log).mean()
        # If we have a validation set evaluate it once per epoch
        val_mean = None
        if val_dataloader is not None:
            R.eval()
            val_loss_log = []
            with torch.no_grad():
                for batch in val_dataloader:
                    batch_samples, mask = batch
                    batch_samples = batch_samples.to(device)
                    mask = mask.to(device)
                    outputs = R(batch_samples)
                    per_token = torch.sum(torch.exp((-outputs.abs())), dim=-1)
                    mask_f = mask.to(dtype=per_token.dtype)
                    denom = mask_f.sum()
                    if denom.item() == 0:
                        denom = torch.tensor(
                            1.0, device=per_token.device, dtype=per_token.dtype
                        )
                    loss = (per_token * mask_f).sum() / denom
                    val_loss_log.append(loss.detach())
            if len(val_loss_log) > 0:
                val_mean = torch.stack(val_loss_log).mean()
            R.train()

        log_message = f"Epoch [{epoch + 1}/{ep}], Train Loss: {mean_loss.item():.4f}"
        if val_mean is not None:
            log_message += f", Val Loss: {val_mean.item():.4f}"
        log_message += ", "
        if cos_lr:
            log_message += f", LR: {scheduler.get_last_lr()[0]:.4e}"
        print(log_message)

    print(f"---> R of layer {name} training done ")

    return R.rotate.data.detach().clone()


if __name__ == "__main__":
    model_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model, processor = dist_load_model(load_processor=True)
    # dist.barrier()
    with patch_module_non_persistent_buffers(model):
        model.eval()
        # no_grad for compression
        for param in model.parameters():
            param.requires_grad = False
        if "vit" in enable_modality:
            state_vit, recipe_vit, model = pre_compression_thinker_vit(model)
        if "aut" in enable_modality:
            state_aut, recipe_aut, model = pre_compression_thinker_aut(model)
        if "text" in enable_modality:
            state_text, recipe_text, model = pre_compression_thinker_text(model)

        weight_tied_name_map = build_weight_tied_map_with_unionfind(model.thinker)
        # Map from weight_id -> {src_data_ptr: cpu_tensor}
        stat_tensors = defaultdict(dict)
        hooks = regist_hook(model, stat_tensors)
        inference_model(model)
        for h in hooks:
            h.remove()
        for module_name, param_name in set(weight_tied_name_map.values()):
            module = model.thinker.get_submodule(module_name)
            param = model.thinker.get_parameter(f"{module_name}.{param_name}")
            R = train_rotate(
                name=f"{module_name}.{param_name}",
                transform_module=module,
                train_datas=list(stat_tensors[id(param)].values()),
                optim="adam",
                lr=1.5e-3,
                mom=0.9,
                cos_lr=False,
                ep=10,
                bsz=64,
                accumulation_steps=2,
                val_ratio=0.1,
            )
            module.weight.data.copy_(
                R.to(device=module.weight.device, dtype=module.weight.dtype)
            )
        # state, recipe_, model = pre_compression_thinker(model)
        if "vit" in enable_modality:
            recipe_vit[0]._fold_transforms_into_weights(state_vit.model)
        if "aut" in enable_modality:
            recipe_aut[0]._fold_transforms_into_weights(state_aut.model)
        if "text" in enable_modality:
            recipe_text[0]._fold_transforms_into_weights(state_text.model)

    post_compression_thinker(model, processor)
