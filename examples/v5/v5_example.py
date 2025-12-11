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
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
MODEL_ID = "/dataset/workspace/zhangl98/v5-1010/vit_final_model/ostq_noSele_vit_merge/"

fq = True  # True
realq = False
flag = "autoround-fp16"
# MODEL_ID = "/dataset/workspace/zhangl98/v5-1010/w4a8/ostq_noSele/transformed_model"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    # torch_dtype="auto",
    torch_dtype=torch.float16,
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

MAX_SEQUENCE_LENGTH = 2048

ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)

config_groups = {
    "group_0": {
        "targets": ["Linear"],
        "input_activations": None,
        "output_activations": None,
        "weights": {
            "num_bits": 4,
            "type": "int",
            "strategy": "channel",
            "dynamic": False,
            "symmetric": True,
        },
    }
}


# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with AutoRound with a group size 128
recipe = AutoRoundModifier(
    ignore=["lm_head"],
    iters=200,
    enable_torch_compile=False,
    config_groups=config_groups,
)


ori_save_pretrained = model.language_model.save_pretrained
# Apply algorithms.
oneshot(
    model=model.language_model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    # disable shuffling to get slightly better mmlu score
    shuffle_calibration_samples=False,
)
model.language_model.save_pretrained = ori_save_pretrained

print("==========================================\n\n")


SAVE_DIR = (
    "/tmp/"
    + MODEL_ID.rstrip("/").split("/")[-1]
    + f"-{flag}-sym-text"
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

quantized_name_set = set()

for _, module in match_named_modules(model, recipe.resolved_targets, recipe.ignore):
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
# torch.save(transform_state_dict, f"{SAVE_DIR}/transform_state_dict.pt")
print(SAVE_DIR)
