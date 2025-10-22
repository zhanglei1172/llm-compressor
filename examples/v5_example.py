from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
MODEL_ID = "/dataset/v5_2_model/v5_2_model_release/checkpoint-14160/"
# MODEL_ID = "/dataset/workspace/zhangl98/v5-1010/w4a8/ostq_noSele/transformed_model"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    # device_map="cuda:0",
    # attn_implementation="eager",
    # use_cache=True
)
# model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "/dataset/workspace/zhangl98/dataset/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
# recipe = [
#     AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
# ]

recipe = """
quant_stage:
    quant_modifiers:
        AWQModifier:
            ignore: ["lm_head", "re:visual.*", "re:model.visual.*"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        type: int
                        strategy: channel
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
"""
# recipe = """
# quant_stage:
#     quant_modifiers:
#         QuantizationModifier:
#             ignore: ["lm_head", "re:visual.*", "re:model.visual.*"]
#             config_groups:
#                 group_0:
#                     weights:
#                         observer: mse
#                         observer_kwargs:
#                             maxshrink: 0.1
#                             patience: 10
#                             averaging_constant: 0.05
#                             grid: 128.0
#                             norm: 2.0
#                         num_bits: 4
#                         type: int
#                         strategy: channel
#                         dynamic: false
#                         symmetric: true
#                     targets: ["Linear"]
# """

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    # pipeline='basic'
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq-w4-sym"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

# cp /dataset/v5_2_model/v5_2_model_release/checkpoint-14160/preprocessor_config.json ./
