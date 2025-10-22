from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
MODEL_ID = "/workspace/lim42@xiaopeng.com/binary_data/pytorch_models/Qwen3-30B-A3B"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "/dataset/workspace/zhangl98/dataset/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# Load dataset and preprocess.
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


# Configure the quantization algorithm to run.
# NOTE: vllm currently does not support asym MoE, using symmetric here
recipe = [
    AWQModifier(
        ignore=["lm_head", "re:.*mlp.shared_expert_gate$"],
        scheme="W4A16",
        targets=["Linear"],
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
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
# SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq-sym2"
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.utils.match import match_named_modules
from tqdm import tqdm

for prefix, module in tqdm(
    match_named_modules(
        model,
        ["re:.*mlp.gate$"],
        warn_on_fail=True,
    ),
    desc="Compressing model",
):
    try:
        assert module.quantization_status == QuantizationStatus.FROZEN, (
            f"{module.quantization_status}"
        )
    except:
        print(f"Quantization status is not frozen for {module}")
    delattr(module, "quantization_status")
    delattr(module, "quantization_enabled")
    delattr(module, "quantization_scheme")
    # 删除scale和zero_point nn parameter，节省存储空间
    for key in list(module._parameters.keys()):
        if key.endswith("_scale") or key.endswith("_zero_point"):
            delattr(module, key)
SAVE_DIR = "/tmp/" + "awq-sym3-realq"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
