from typing import Mapping, Optional, Union
from datasets import load_dataset, load_from_disk
from accelerate.hooks import attach_align_device_hook
from accelerate.hooks import remove_hook_from_module
from compressed_tensors import get_execution_device
from llmcompressor.pipelines.sequential.helpers import SequentialTracer
from llmcompressor.utils import helpers
import contextlib

from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)
import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
    forward_quantize,
)
from qwen_vl_utils import process_vision_info
from qwen_omni_utils import process_mm_info
from llmcompressor.modifiers.awq.mappings import AWQ_MAPPING_REGISTRY, _moe_default_mappings

AWQ_MAPPING_REGISTRY["Qwen3OmniMoeForConditionalGeneration"] = _moe_default_mappings

#################### configurations ####################
calibrate_moe_context = True
# Select model and load it.
pretrain = "origin"
recipe = "examples/qwen3_omni_configs/text/gptq.yaml"
flag = "gptq"
#################### configurations ####################


if pretrain == "origin":

    MODEL_ID = "/code/omni_ostq/transformed_model/"
else:
    MODEL_ID = "/dataset/workspace/zhangl98/models/Qwen3-Omni-30B-A3B-Instruct/"

if calibrate_moe_context:
    flag += "-calmoe"


# Select calibration dataset.
# DATASET_ID = "/dataset/workspace/zhangl98/dataset/ultrachat_200k"
# DATASET_SPLIT = "train_sft"
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test[:512]"
# DATASET_SPLIT = "test"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048
# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
# NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
# ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
# ds = load_dataset("parquet", data_files={DATASET_SPLIT: "/dataset/workspace/zhangl98/dataset/peoples_speech/test/*"}, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)

USE_AUDIO_IN_VIDEO = True

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
dtype = model.dtype
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
# def preprocess(example):
#     # return {
#     #     "text": tokenizer.apply_chat_template(
#     #         example["messages"],
#     #         tokenize=False,
#     #     )
#     # }
#     text = processor.apply_chat_template(
#         example["messages"], tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, video_inputs = process_vision_info(example["messages"])
#     return processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=False,
#         max_length=MAX_SEQUENCE_LENGTH,
#         truncation=True,
#     )

def preprocess(example):
    # conversation = example["messages"]
    conversation = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": example["image"],
            },
            {
                "type": "text",
                # "text": example["text"].capitalize(),
                "text": "What does the image show?"
            },
        ]
    }]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    return processor(text=text, 
                   audio=audios, 
                   images=images, 
                   videos=videos, 
                   return_tensors="pt", 
                   padding=True, 
                   use_audio_in_video=USE_AUDIO_IN_VIDEO)

ds = ds.map(preprocess, remove_columns=ds.column_names)


# Tokenize inputs.
# def tokenize(sample):
#     return tokenizer(
#         sample["text"],
#         padding=False,
#         max_length=MAX_SEQUENCE_LENGTH,
#         truncation=True,
#         add_special_tokens=False,
#     )

# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value, dtype=dtype if key== "pixel_values" else None) for key, value in batch[0].items()}


# Configure the quantization algorithm to run.
# NOTE: vllm currently does not support asym MoE, using symmetric here
# recipe = [
#     AWQModifier(
#         ignore=["re:lm_head", "re:visual.*", "re:model.visual.*", "re:audio_tower.*"],
#         # scheme="W4A16",
#         config_groups={
#             "group_0": QuantizationScheme(
#                     targets=["Linear"],
#                     weights=QuantizationArgs(
#                         num_bits=4,
#                         type=QuantizationType.INT,
#                         strategy=QuantizationStrategy.CHANNEL,
#                         symmetric=True,
#                         dynamic=False,
#                     )
#                 )
#         },
#         # targets=["Linear"],
#     ),
# ]



original_init = SequentialTracer.__init__
def my_init(
    self,
    ancestors, offloaded
):
    original_init(
        self,
        ancestors,
        offloaded,
    )
    # Force onload all modules.
    device = get_execution_device(model)
    remove_hook_from_module(model.thinker.visual.pos_embed, recurse=False)
    model.thinker.visual.pos_embed.to(device)
    self.offloaded.remove(model.thinker.visual.pos_embed)

with contextlib.ExitStack() as stack:
    stack.enter_context(
        helpers.patch_attr(SequentialTracer, "__init__", my_init)
    )
    # Apply algorithms.
    oneshot(
        model=model.thinker,
        processor=model.config._name_or_path,
        dataset=ds,
        recipe=recipe,
        tie_word_embeddings=True,
        data_collator=data_collator,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        calibrate_moe_context=calibrate_moe_context,
        sequential_targets=["Qwen3OmniMoeThinkerTextDecoderLayer"],
    )

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
messages = [
    {
        "role": "user",
        "content": [
            # {
            #     "type": "image",
            #     "image": "http://images.cocodataset.org/train2017/000000231895.jpg",
            # },
            {"type": "text", "text": "Hello my name is\n"},
        ],
    }
]
# prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[prompt],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=False,
#     max_length=MAX_SEQUENCE_LENGTH,
#     truncation=True,
#     return_tensors="pt",
# ).to(model.device)
# # input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
# #     model.device
# # )
# output = model.generate(**inputs, max_new_tokens=100)
# print(processor.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
# SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq-sym2"
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.utils.match import match_named_modules
from tqdm import tqdm

# for prefix, module in tqdm(
#     match_named_modules(
#         model,
#         ["re:.*mlp.gate$"],
#         warn_on_fail=True,
#     ),
#     desc="Compressing model",
# ):
#     try:
#         assert module.quantization_status == QuantizationStatus.FROZEN, (
#             f"{module.quantization_status}"
#         )
#         delattr(module, "quantization_status")
#         delattr(module, "quantization_enabled")
#         delattr(module, "quantization_scheme")
#         # 删除scale和zero_point nn parameter，节省存储空间
#         for key in list(module._parameters.keys()):
#             if key.endswith("_scale") or key.endswith("_zero_point"):
#                 delattr(module, key)
#     except:
#         print(f"Quantization status is not frozen for {prefix}")

SAVE_DIR = "/tmp/" + MODEL_ID.rstrip("/").split("/")[-1] + f"-{pretrain}-{flag}-sym-com-text"
# from llmcompressor.transformers.compression.compressed_tensors_utils import modify_save_pretrained
# modify_save_pretrained(model)
# model.save_pretrained(SAVE_DIR, save_compressed=True)
# processor.save_pretrained(SAVE_DIR)

# SAVE_DIR = "/tmp/" + MODEL_ID.rstrip("/").split("/")[-1] + "awq-sym-realq-audio"
# model.save_pretrained(SAVE_DIR+"-trans") # trans
# processor.save_pretrained(SAVE_DIR+"-trans")
# modify_save_pretrained(model)
from llmcompressor.recipe import Recipe
recipe = Recipe.create_instance(
                path_or_modifiers=recipe, target_stage=None
            )
for _, module in match_named_modules(model, recipe.modifiers[0].resolved_targets, recipe.modifiers[0].ignore):
    if hasattr(module, "quantization_status"):
        assert module.quantization_status == QuantizationStatus.FROZEN, (
            f"{module.quantization_status}"
        )
        scheme = getattr(module, "quantization_scheme", None)
        module.weight.data = forward_quantize(
                module, module.weight, "weight", scheme.weights
            )
        delattr(module, "quantization_status")
        delattr(module, "quantization_enabled")
        delattr(module, "quantization_scheme")
        for key in list(module._parameters.keys()):
            if key.endswith("_scale") or key.endswith("_zero_point"):
                delattr(module, key)
model.save_pretrained(SAVE_DIR+"-fq")#, save_compressed=True) # fakequant
processor.save_pretrained(SAVE_DIR+"-fq")
