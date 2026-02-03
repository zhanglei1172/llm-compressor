import base64
import contextlib
import os
from dataclasses import dataclass
from glob import glob
from io import BytesIO
from typing import Generator, Mapping, Tuple

import numpy as np
import soundfile as sf
import torch
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStatus,
    apply_quantization_config,
    enable_quantization,
)
from compressed_tensors.transform import apply_transform_config
from datasets import concatenate_datasets, load_dataset, load_from_disk
from qwen_omni_utils import process_mm_info
from safetensors.torch import safe_open
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from trl.trainer.sft_trainer import (
    DataCollatorForLanguageModeling,
)

from llmcompressor.core.state import State
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.utils.analysis.graphwise import graphwise_error_analyse
from llmcompressor.utils.analysis.layerwise import layerwise_error_analyse
from llmcompressor.utils.pytorch.module import (
    patch_module_non_persistent_buffers,
)

torch.fx.experimental._config.meta_nonzero_assume_all_nonzero = True
# awq_mappings.AWQ_MAPPING_REGISTRY["Qwen3OmniMoeThinkerForConditionalGeneration"] = awq_mappings._moe_default_mappings
USE_AUDIO_IN_VIDEO = True


#################### configurations ####################

NUM_CALIBRATION_SAMPLES = 256
enable_modality = {
    # "vit",
    # "aut",
    "text"
}
model_dtype = torch.float32
#################### configurations ####################


MODEL_ID = "/dataset/workspace/lim42/models/Qwen3-8B-r1r2-r4-gptq-w4-weightonly/"
SCALE_MODEL_PATH = "/dataset/workspace/lim42/models/Qwen3-8B-r1r2-r4-gptq-w4a8-static/"

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
from compressed_tensors.compressors import ModelCompressor
from transformers.utils.quantization_config import (  # noqa: F401
    CompressedTensorsConfig,
)


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
def pre_compression(model):
    state = State()
    state.update(
        model=model,
    )
    ignore = ["lm_head"]
    if "vit" not in enable_modality:
        ignore.append("re:.*visual.*")

    if "text" not in enable_modality:
        ignore.append("re:.*language_model.*")
    recipe_ = [
        SpinQuantModifier(
            do_fold=True,
            backe_mean=False,
            learnable=True,
            rotations=["R4"],
            transform_block_size_R4=256,
            transform_type="hadamard",
            skip_weights_folding=True,
        ),
        QuantizationModifier(
            ignore=ignore,
            config_groups={
                # "group_0": {
                #     "input_activations": {
                #         "observer": "minmax",
                #         "num_bits": 8,
                #         "type": "int",
                #         "symmetric": True,
                #         "strategy": "tensor",
                #         "dynamic": False,
                #     },
                #     "targets": [
                #         r"re:.*up_proj$",
                #         r"re:.*gate_proj$",
                #         r"re:.*q_proj$",
                #         r"re:.*k_proj$",
                #         r"re:.*v_proj$",
                #         r"re:.*o_proj$",
                #         r"re:.*out_proj$",
                #         r"re:.*attn\.proj$",
                #     ],
                # },
                "group_1": {
                    "input_activations": {
                        "observer": "minmax",
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "tensor",
                        "dynamic": False,
                    },
                    "targets": [
                        r"re:.*down_proj$",
                    ],
                },
            },
        ),
    ]

    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.nn.utils.parametrize.cached())
        for mod in recipe_:
            mod.on_initialize(state=state)
        # for param in model.thinker.model.parameters():
        #     param.requires_grad = False
        recipe_[0].on_start(state=state, event=None)
        recipe_[0]._fold_transforms_into_weights(state.model) # TODO
        model.apply(enable_quantization)

    return state, recipe_, model


class DataCollatorForQwen3OmniDataset(DataCollatorForLanguageModeling):
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
            # audio=audios,
            # images=images,
            # videos=videos,
            return_tensors="pt",
            padding=True,
            # use_audio_in_video=USE_AUDIO_IN_VIDEO,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(dtype=model_dtype)
        return batch


def load_scale(quant_model, sclae_hf_path, config_json_str=None):
    @dataclass
    class StateDictIterator:
        filepath: str

        def __iter__(self) -> Generator[Tuple[str, "torch.Tensor"], None, None]:
            if self.filepath.endswith(".safetensors"):
                with safe_open(self.filepath, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        yield key, f.get_tensor(key)

            else:
                state_dict = torch.load(
                    self.filepath, map_location="cpu", weights_only=True, mmap=True
                )
                for key in state_dict.keys():
                    yield key, state_dict[key]


    safetensor_files = list(glob(os.path.join(sclae_hf_path, "*.safetensors")))
    safetensor_files.sort()
    state_dict_iterators = [
        StateDictIterator(shard_file) for shard_file in safetensor_files
    ]

    new_state_dict = {}

    for state_dict_iterator in tqdm(
        state_dict_iterators, desc="Loading checkpoint shards"
    ):
        for name, tensor in state_dict_iterator:
            if name.endswith("_scale") or name.endswith("_zero_point"):
                new_state_dict[name] = tensor.to(dtype=model_dtype, device="cpu")
    if config_json_str:
        # 解析量化配置文件
        config = QuantizationConfig.model_validate_json(config_json_str)

        # set status to calibration
        config.quantization_status = QuantizationStatus.FROZEN

        # initialize quantization
        apply_quantization_config(quant_model, config)  # 增加scale和zero_point
        apply_transform_config(quant_model, config)

    # update statedict
    quant_model.load_state_dict(new_state_dict, strict=False)
    # re-enable quantization
    quant_model.apply(enable_quantization)


def load_model(model_path=MODEL_ID, load_processor=False):
    processor = None
    if load_processor:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_path, torch_dtype=model_dtype
    # )
    true_decompressed_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=model_dtype,
    )  # hadamard transformed weights not load
    return true_decompressed_model, processor


if __name__ == "__main__":
    model_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    model, processor = load_model(load_processor=True)
    state, recipe_, model = pre_compression(model)
    load_scale(
        model,
        sclae_hf_path=SCALE_MODEL_PATH,
        config_json_str=None,
    )
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

        module = model

        ori_forward = module.forward.__func__

        dispatch_for_generation(model)

        input_ids = train_processor("Hello my name is", return_tensors="pt").input_ids.to(
            model.device
        )
        output = model.generate(input_ids, max_new_tokens=100)
        print(train_processor.decode(output[0]))

        def forward(module, *args, **kwargs):
            if isinstance(args[0], Mapping):
                kwargs = kwargs.copy()
                kwargs.update(args[0])
                ret = ori_forward.__get__(module, type(module))(*args[1:], **kwargs)
            else:
                ret = ori_forward.__get__(module, type(module))(*args, **kwargs)
            return ret.logits

        module.forward = forward.__get__(module, type(module))

        results = graphwise_error_analyse(
            model,  # TODO
            dataloader,
            method="cosine",
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
