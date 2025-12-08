import base64
import contextlib
import copy
import os
from io import BytesIO
from typing import Mapping
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
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
from torch.utils.data import DataLoader
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
from llmcompressor.utils.analysis.graphwise import graphwise_error_analyse
from llmcompressor.utils.analysis.layerwise import layerwise_error_analyse
from llmcompressor.utils.pytorch.module import (
    build_weight_tied_map_with_unionfind,
    patch_module_non_persistent_buffers,
)

torch.fx.experimental._config.meta_nonzero_assume_all_nonzero = True
# awq_mappings.AWQ_MAPPING_REGISTRY["Qwen3OmniMoeThinkerForConditionalGeneration"] = awq_mappings._moe_default_mappings
USE_AUDIO_IN_VIDEO = True


#################### configurations ####################

NUM_CALIBRATION_SAMPLES = 256
enable_modality = {
    # "vit",
    "aut",
    # "text"
}
#################### configurations ####################

model_dtype = torch.bfloat16

MODEL_ID = (
    "/tmp/Qwen3-Omni-30B-A3B-Instruct-origin-spinquant-calmoe(aut,)-sym-com-text-trans/"
)

MAX_SEQUENCE_LENGTH = 2048
# Load dataset and preprocess.
# ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds_vl = load_dataset(
    "lmms-lab/LLaVA-OneVision-Data", "FigureQA(MathV360K)", split="train[:512]"
)
ds_al = load_dataset(
    "/dataset/workspace/zhangl98/dataset/peoples_speech/test", split="test[:3072]"
)
ds_text = load_dataset("hkust-nlp/deita-6k-v0", split="train[:512]")
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
                    "image": f'data:image;base64,{encode_base64_img(example["image"])}',
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
                            "image": f'data:image;base64,{encode_base64_img(example["image"])}',
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
                    "audio": f'data:audio/wav;base64,{encode_base64_audio(example["audio"]["array"], example["audio"]["sampling_rate"])}',
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

    return model


@torch.no_grad()
def pre_compression_thinker_aut(model):
    from llmcompressor.modeling.qwen3_omni_moe import (
        replace_audio_embedding,
        replace_rmsnorm,
    )

    replace_audio_embedding(model.thinker.audio_tower)
    model.thinker.audio_tower.positional_embedding.positional_embedding = (
        model.thinker.audio_tower.positional_embedding.weight
    )
    replace_rmsnorm(model.thinker.audio_tower)
    tensor = model.thinker.audio_tower.positional_embedding.weight
    ori_device = tensor.device
    ori_shape = tensor.shape
    model_device = tensor.device
    Q1 = torch.load(f"{MODEL_ID}/transform_state_dict.pt")[
        "audio_tower.positional_embedding.R1_weight_output"
    ]["weight"].to(dtype=torch.float64, device=model_device)
    model.thinker.audio_tower.positional_embedding.weight.data = (
        (
            (tensor - tensor.mean(-1, keepdim=True))
            .to(dtype=Q1.dtype, device=model_device)
            .reshape(-1, ori_shape[-1] // Q1.shape[0], Q1.shape[0])
            @ Q1
        )
        .to(dtype=model_dtype, device=ori_device)
        .reshape(ori_shape)
    )
    return model


@torch.no_grad()
def pre_compression_thinker_text(model):
    return model


@torch.no_grad()
def pre_compression_thinker(model):
    state = State()
    state.update(
        model=model.thinker,
    )
    ignore = ["lm_head"]
    if "vit" not in enable_modality:
        ignore.append("re:visual.*")
    if "aut" not in enable_modality:
        ignore.append("re:audio_tower.*")
    if "text" not in enable_modality:
        ignore.append("re:model.*")
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
                    # "input_activations": {
                    #     "observer": "minmax",
                    #     "num_bits": 8,
                    #     "type": "int",
                    #     "symmetric": True,
                    #     "strategy": "tensor",
                    #     "dynamic": True,
                    # },
                    "targets": [
                        r"re:.*up_proj$",
                        r"re:.*gate_proj$",
                        r"re:.*q_proj$",
                        r"re:.*k_proj$",
                        r"re:.*v_proj$",
                        r"re:.*o_proj$",
                        r"re:.*out_proj$",
                        # r"re:.*proj1$",
                        r"re:.*fc1$",
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
                    # "input_activations": {
                    #     "observer": "minmax",
                    #     "num_bits": 16,
                    #     "type": "int",
                    #     "symmetric": True,
                    #     "strategy": "tensor",
                    #     "dynamic": True,
                    # },
                    "targets": [
                        r"re:.*down_proj$",
                        r"re:.*fc2$",
                        # r"re:.*proj2$",
                    ],
                    "ste": True,
                },
            },
        )
    ]

    _tmp_config = copy.deepcopy(model.thinker.config)
    _tmp_config.update(model.thinker.config.text_config.to_dict())

    with contextlib.ExitStack() as stack:
        stack.enter_context(helpers.patch_attr(model.thinker, "config", _tmp_config))
        stack.enter_context(torch.nn.utils.parametrize.cached())
        for mod in recipe_:
            mod.on_initialize(state=state)
        # for param in model.thinker.model.parameters():
        #     param.requires_grad = False
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

        batch["labels"] = labels  # batch["input_ids"]
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(dtype=model_dtype)
        return batch


def load_model(model_path=MODEL_ID, load_processor=False):
    processor = None
    if load_processor:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto"
    )
    return model, processor


if __name__ == "__main__":
    model_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    model, processor = load_model(load_processor=False)
    train_processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        model_max_length=MAX_SEQUENCE_LENGTH,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    dataloader_params = {
        "batch_size": 2,
        "collate_fn": DataCollatorForQwen3OmniDataset(train_processor),
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
    }
    dataloader = DataLoader(ds, **dataloader_params)
    # dist.barrier()
    with patch_module_non_persistent_buffers(model):
        model.eval()
        # no_grad for compression
        for param in model.parameters():
            param.requires_grad = False
        if "vit" in enable_modality:
            model = pre_compression_thinker_vit(model)
        if "aut" in enable_modality:
            model = pre_compression_thinker_aut(model)
        if "text" in enable_modality:
            model = pre_compression_thinker_text(model)
        # if RANK_OTHER:
        #     state_text, recipe_text, model = pre_compression_thinker_text(model)
        # else:
        #     state_text, recipe_text, model = pre_compression_thinker_text_sequential(
        #         model
        #     )
        # model.thinker.apply(enable_quantization)
        state, recipe_, model = pre_compression_thinker(model)

        module = model.thinker

        ori_forward = module.forward.__func__

        def forward(module, *args, **kwargs):
            if isinstance(args[0], Mapping):
                kwargs = kwargs.copy()
                kwargs.update(args[0])
                ret = ori_forward.__get__(module, type(module))(*args[1:], **kwargs)
            else:
                ret = ori_forward.__get__(module, type(module))(*args, **kwargs)
            return ret.logits

        module.forward = forward.__get__(module, type(module))

        dispatch_for_generation(model)

        results = graphwise_error_analyse(
            model.thinker,  # TODO
            dataloader,
            method="sqnr",
            steps=8,
            verbose=True,
        )
        results = layerwise_error_analyse(
            model.thinker,
            dataloader,
            method="sqnr",
            steps=8,
            verbose=True,
        )
