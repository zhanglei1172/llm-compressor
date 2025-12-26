import contextlib
import copy
from typing import Mapping, Optional, Union
from unittest.mock import patch

import torch
from accelerate.hooks import attach_align_device_hook, remove_hook_from_module
from compressed_tensors import get_execution_device
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
    forward_quantize,
)
from datasets import load_dataset, load_from_disk
from qwen_omni_utils import process_mm_info
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq import mappings as awq_mappings
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings
from llmcompressor.pipelines.sequential.helpers import SequentialTracer
from llmcompressor.utils import dispatch_for_generation, helpers

# awq_mappings.AWQ_MAPPING_REGISTRY["Qwen3OmniMoeThinkerForConditionalGeneration"] = awq_mappings._moe_default_mappings

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

#################### configurations ####################
# Select model and load it.
pretrain = "ostq"
# recipe = "examples/qwen3_omni_configs/text/gptq.yaml"
recipe = "examples/qwen3_omni_configs/text/mse_w4a8.yaml"
# flag = "gptq-moeall"
flag = "mse_w4a8"
fq = False
realq = True
NUM_CALIBRATION_SAMPLES = 256
from llmcompressor.modeling.moe_context import MoECalibrationModule
@MoECalibrationModule.register("Qwen3OmniMoeThinkerTextSparseMoeBlock")
class CalibrationQwen3MoeSparseMoeBlock(MoECalibrationModule):
    """
    Calibration version of Qwen3MoeSparseMoeBlock that sends all tokens to all experts.
    """

    is_permanent = False

    def __init__(
        self,
        original,
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.calibrate_all_experts = calibrate_all_experts
        self.gate = original.gate
        self.experts = original.experts

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=1, dtype=torch.float
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            if self.calibrate_all_experts:
                expert_out = expert_layer(hidden_states)[top_x]
            else:
                expert_out = expert_layer(hidden_states[top_x])

            # TODO: double check
            if len(top_x) > 0:
                current_hidden_states = expert_out * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original

#################### configurations ####################


if pretrain == "ostq":
    MODEL_ID = "/tmp/Qwen2.5-VL-7B-Instruct-origin-ostquant(text|)-trans-gptq-fq-autoround-fq"
else:
    MODEL_ID = "/dataset/workspace/zhangl98/models/Qwen3-Omni-30B-A3B-Instruct/"



# Select calibration dataset.
# DATASET_ID = "/dataset/workspace/zhangl98/dataset/ultrachat_200k"
# DATASET_SPLIT = "train_sft"
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test[:512]"
# DATASET_SPLIT = "test"
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

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype="auto"
)
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
    conversation = [
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
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
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
    return {
        key: torch.tensor(value, dtype=dtype if key == "pixel_values" else None)
        for key, value in batch[0].items()
    }


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


def my_init(self, ancestors, offloaded):
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


_tmp_config = copy.deepcopy(model.thinker.config)
_tmp_config.update(model.thinker.config.text_config.to_dict())

with contextlib.ExitStack() as stack:
    stack.enter_context(helpers.patch_attr(SequentialTracer, "__init__", my_init))
    stack.enter_context(helpers.patch_attr(model.thinker, "config", _tmp_config))
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
        sequential_targets=["Qwen3OmniMoeThinkerTextDecoderLayer"],
    )

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

SAVE_DIR = (
    "/tmp/"
    + MODEL_ID.rstrip("/").split("/")[-1]
    + f"-{pretrain}-{flag}-sym-com-text"
    + ("-realq" if realq else ("-fq" if fq else "-trans"))
)
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)

if realq:
    modify_save_pretrained(model)
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    processor.save_pretrained(SAVE_DIR)
    print(SAVE_DIR)
    exit(0)

# SAVE_DIR = "/tmp/" + MODEL_ID.rstrip("/").split("/")[-1] + "awq-sym-realq-audio"
# model.save_pretrained(SAVE_DIR+"-trans") # trans
# processor.save_pretrained(SAVE_DIR+"-trans")
# modify_save_pretrained(model)
from llmcompressor.recipe import Recipe

recipe = Recipe.create_instance(path_or_modifiers=recipe, target_stage=None)

quantized_name_set = set()
import re

for _, module in match_named_modules(
    model, recipe.modifiers[-1].resolved_targets, recipe.modifiers[-1].ignore
):
    if hasattr(module, "quantization_status"):
        assert (
            module.quantization_status == QuantizationStatus.FROZEN
        ), f"{module.quantization_status}"
        quantized_name_set.add(re.sub(r"\d+", "X", _))
        scheme = getattr(module, "quantization_scheme", None)
        if fq:
            module.weight.data = forward_quantize(
                module, module.weight, "weight", scheme.weights
            )
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
