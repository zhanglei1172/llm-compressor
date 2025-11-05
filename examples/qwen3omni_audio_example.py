import contextlib
import copy

import torch
from accelerate.hooks import remove_hook_from_module
from compressed_tensors import get_execution_device
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
    forward_quantize,
)
from datasets import load_dataset
from qwen_omni_utils import process_mm_info
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.modeling.qwen3_omni_moe import (
    get_audio_wrap_functions,
    replace_audio_embedding,
)
from llmcompressor.modifiers.awq import mappings as awq_mappings
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings
from llmcompressor.pipelines.sequential.helpers import SequentialTracer
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)
from llmcompressor.utils import dispatch_for_generation, helpers

mappings.SPINQUANT_MAPPING_REGISTRY["Qwen3OmniMoeAudioEncoder"] = (
    mappings.SpinQuantMapping(
        mm_proj=["conv_out"],
        embedding="re:.*positional_embedding$",
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
recipe = "examples/qwen3_omni_configs/audio/quarot.yaml"
# recipe = "examples/qwen3_omni_configs/audio/awq.yaml"
flag = "quarot"
# flag = "awq"
fq = False
#################### configurations ####################

# Select model and load it.
MODEL_ID = "/dataset/workspace/zhangl98/models/Qwen3-Omni-30B-A3B-Instruct/"

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype="auto"
)
dtype = model.dtype
replace_audio_embedding(model.thinker.audio_tower)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "MLCommons/peoples_speech"
DATASET_ID = "/dataset/workspace/zhangl98/dataset/peoples_speech/test"
DATASET_SUBSET = "test"
DATASET_SPLIT = "test"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID,
    # DATASET_SUBSET,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
)
ds = ds.shuffle(seed=42)

USE_AUDIO_IN_VIDEO = True


def preprocess(example):
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio": example["audio"]["array"],
                },
                {
                    "type": "text",
                    "text": example["text"].capitalize(),
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
    ret = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    # ret = {
    #     k: v.astype(dtype) if torch.is_floating_point(v) else v
    #     for k, v in ret.items()
    #     }
    del ret["input_ids"], ret["attention_mask"]
    return ret


ds = ds.map(preprocess, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    ret = {
        key: torch.tensor(value, dtype=dtype if key == "input_features" else None)
        for key, value in batch[0].items()
    }
    audio_feature_lengths = torch.sum(ret["feature_attention_mask"], dim=1)
    input_features = (
        ret["input_features"]
        .permute(0, 2, 1)[ret["feature_attention_mask"].bool()]
        .permute(1, 0)
    )
    feature_lens = (
        audio_feature_lengths
        if audio_feature_lengths is not None
        else ret["feature_attention_mask"].sum(-1)
    )
    return {"input_features": input_features, "feature_lens": feature_lens}


# Configure the quantization algorithm to run.
# NOTE: vllm currently does not support asym MoE, using symmetric here
# recipe = [
#     AWQModifier(
#         # ignore=["re:lm_head", "re:.*mlp.shared_expert_gate$", "re:visual.*", "re:model.visual.*", "re:model.layers.*"],
#         ignore=["re:lm_head", "re:visual.*", "re:model.visual.*"]+
#         ["re:conv.*", r"re:proj[\d].*", "re:positional_embedding*"],
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
#         mappings=[
#             AWQMapping(
#                 "re:.*self_attn_layer_norm$",
#                 ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
#             ),
#             AWQMapping("re:.*v_proj$", ["re:.*out_proj$"]),
#             AWQMapping(
#                 "re:.*final_layer_norm$",
#                 ["re:.*fc1$"],
#             ),
#             # AWQMapping(
#             #     "re:.*up_proj$",
#             #     ["re:.*down_proj$"],
#             # ),
#         ],
#         # targets=["Linear"],
#     ),
# ]


import sys

audio_wrap_funcs = get_audio_wrap_functions()

sys.modules[model.thinker.audio_tower.__class__.__module__].__dict__.update(
    audio_wrap_funcs
)

_tmp_config = copy.deepcopy(model.thinker.audio_tower.config)
_tmp_config.update(
    {"head_dim": _tmp_config.d_model // _tmp_config.encoder_attention_heads}
)

original_init = SequentialTracer.__init__


def my_init(self, ancestors, offloaded):
    original_init(
        self,
        ancestors,
        offloaded,
    )
    # Force onload all modules.
    device = get_execution_device(model)
    remove_hook_from_module(
        model.thinker.audio_tower.positional_embedding, recurse=False
    )
    model.thinker.audio_tower.positional_embedding.to(device)
    self.offloaded.remove(model.thinker.audio_tower.positional_embedding)


with contextlib.ExitStack() as stack:
    stack.enter_context(helpers.patch_attr(SequentialTracer, "__init__", my_init))
    stack.enter_context(
        helpers.patch_attr(
            model.thinker.audio_tower,
            "forward",
            audio_wrap_funcs["forward"].__get__(model.thinker.audio_tower),
        )
    )
    stack.enter_context(
        helpers.patch_attr(model.thinker.audio_tower, "config", _tmp_config)
    )
    # Apply algorithms.
    oneshot(
        model=model.thinker.audio_tower,
        processor=model.config._name_or_path,
        dataset=ds,
        recipe=recipe,
        tracing_ignore=[
            "_update_causal_mask",
            "create_causal_mask",
            "_update_mamba_mask",
            "make_causal_mask",
            "get_causal_mask",
            "mask_interface",
            "mask_function",
            "_prepare_4d_causal_attention_mask",
            "_prepare_fsmt_decoder_inputs",
            "_prepare_4d_causal_attention_mask_with_cache_position",
            "_update_linear_attn_mask",
            "pad_sequence",
            "my_wrap_0",
            "my_wrap_1",
            "my_wrap",
            # "tolist",
        ],
        tie_word_embeddings=True,
        data_collator=data_collator,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        calibrate_moe_context=True,
        sequential_targets=["Qwen3OmniMoeAudioEncoderLayer"],
    )

sys.modules[model.thinker.audio_tower.__class__.__module__].__dict__.update(
    {
        "forward": model.thinker.audio_tower.forward,
    }
)

from collections import OrderedDict

_h = set()
transform_state_dict = OrderedDict()
from compressed_tensors.transform.factory.base import TransformBase

for name, module in model.thinker.visual.named_modules():
    if isinstance(module, TransformBase):
        if module in _h or id(module.scheme) in _h:
            continue
        _h.add((module if module.scheme.block_wise else id(module.scheme)))
        print(f"{name}: {module}")
        transform_state_dict.update({name: module.state_dict()})

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
#     except:
#         print(f"Quantization status is not frozen for {prefix}")
#     delattr(module, "quantization_status")
#     delattr(module, "quantization_enabled")
#     delattr(module, "quantization_scheme")
#     # 删除scale和zero_point nn parameter，节省存储空间
#     for key in list(module._parameters.keys()):
#         if key.endswith("_scale") or key.endswith("_zero_point"):
#             delattr(module, key)

SAVE_DIR = (
    "/tmp/"
    + MODEL_ID.rstrip("/").split("/")[-1]
    + f"-{flag}-sym-com-audio"
    + ("-fq" if fq else "-trans")
)
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
            if isinstance(module, torch.nn.Linear):
                module.weight.data = forward_quantize(
                    module, module.weight, "weight", scheme.weights
                )
            elif isinstance(module, torch.nn.Conv2d):
                module.weight_scale.data = module.weight_scale.data.unsqueeze(
                    -1
                ).unsqueeze(-1)
                module.weight_zero_point.data = module.weight_zero_point.data.unsqueeze(
                    -1
                ).unsqueeze(-1)
                module.weight.data = forward_quantize(
                    module, module.weight, "weight", scheme.weights
                )
            else:
                raise NotImplementedError(f"Unsupported module type {type(module)}")
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
