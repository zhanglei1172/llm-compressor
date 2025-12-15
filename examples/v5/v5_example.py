import torch
from auto_round.calib_dataset import get_dataset
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
    forward_quantize,
)
from compressed_tensors.transform.factory.base import TransformBase
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings
from llmcompressor.utils import dispatch_for_generation

#################### configurations ####################
# Select model and load it.
MODEL_ID = "/tmp/Qwen2.5-VL-7B-Instruct-quarot-trans"

recipe = "examples/v5/configs/quarot.yaml"
fq = False  # True
realq = False
flag = "quarot"
model_dtype = torch.bfloat16
# MODEL_ID = "/dataset/workspace/zhangl98/v5-1010/w4a8/ostq_noSele/transformed_model"
#################### configurations ####################

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
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    # torch_dtype="auto",
    torch_dtype=model_dtype,
    # device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    # use_cache=True
)
# model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# Select calibration dataset.

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 128
if flag == "quarot":
    NUM_CALIBRATION_SAMPLES = 1

MAX_SEQUENCE_LENGTH = 2048

ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)

# config_groups = {
#     "group_0": {
#         "targets": ["Linear"],
#         "input_activations": None,
#         "output_activations": None,
#         "weights": {
#             "num_bits": 4,
#             "type": "int",
#             "strategy": "channel",
#             "dynamic": False,
#             "symmetric": True,
#         },
#     }
# }


# # Configure the quantization algorithm to run.
# #   * quantize the weights to 4 bit with AutoRound with a group size 128
# recipe = AutoRoundModifier(
#     ignore=["lm_head"],
#     iters=200,
#     enable_torch_compile=False,
#     config_groups=config_groups,
# )


ori_save_pretrained = model.save_pretrained
# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    # disable shuffling to get slightly better mmlu score
    shuffle_calibration_samples=False,
)
model.save_pretrained = ori_save_pretrained

print("==========================================\n\n")

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

SAVE_DIR = (
    "/tmp/"
    + MODEL_ID.rstrip("/").split("/")[-1]
    + f"-{flag}"
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


import re

from compressed_tensors.utils.match import match_named_modules

from llmcompressor.recipe import Recipe

recipe = Recipe.create_instance(path_or_modifiers=recipe, target_stage=None)
quantized_name_set = set()

for _, module in match_named_modules(
    model, recipe.modifiers[-1].resolved_targets, recipe.modifiers[-1].ignore
):
    if hasattr(module, "quantization_status"):
        assert module.quantization_status == QuantizationStatus.FROZEN, (
            f"{module.quantization_status}"
        )
        quantized_name_set.add(re.sub(r"\d+", "X", _))
        scheme = getattr(module, "quantization_scheme", None)
        if fq:
            if isinstance(module, torch.nn.Linear):
                module.weight_scale.data = module.weight_scale.data.abs()
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
