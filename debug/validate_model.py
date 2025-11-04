import torch
from datasets import load_dataset
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    HfArgumentParser,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.datasets import get_calibration_dataloader
from llmcompressor.modeling.qwen3_omni_moe import (
    replace_vit_attention,
)
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
REF_MODEL_ID = "/dataset/workspace/zhangl98/models/Qwen3-Omni-30B-A3B-Instruct/"
MODEL_ID = "/tmp/Qwen3-Omni-30B-A3B-Instruct-origin-quarot-calmoe-sym-com-text-trans/"

ref_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    REF_MODEL_ID, torch_dtype="auto"
)

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype="auto"
)
dtype = model.dtype
model_device = torch.device("cuda")
processor = AutoProcessor.from_pretrained(REF_MODEL_ID, trust_remote_code=True)

# replace_vit_attention(model.thinker.visual)

# Select calibration dataset.
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test[:512]"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 1
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)

USE_AUDIO_IN_VIDEO = True


def preprocess(example):
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
    # del ret["input_ids"], ret["attention_mask"]
    return ret


ds = ds.map(preprocess, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {
        key: torch.tensor(value, dtype=dtype if key == "pixel_values" else None)
        for key, value in batch[0].items()
    }


dataset_kwargs = dict(
    dataset=ds,
    data_collator=data_collator,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

parser = HfArgumentParser(DatasetArguments)
dataset_args = parser.parse_dict(dataset_kwargs)


dataloader = get_calibration_dataloader(dataset_args[0], processor)


def flatten_obj(obj):
    if isinstance(obj, dict):
        ret = []
        for v in obj.values():
            ret.extend(flatten_obj(v))
        return ret
    elif isinstance(obj, (list, tuple)):
        ret = []
        for v in obj:
            ret.extend(flatten_obj(v))
        return ret
    else:
        return [obj]


activations = IntermediatesCache.from_dataloader(dataloader, model_device)
with torch.no_grad():
    _rets = []
    _ref_rets = []
    for batch_idx in tqdm(range(len(dataloader))):
        # inputs = {k: v.to(model_device) for k, v in inputs.items()}
        inputs = activations.fetch(batch_idx)
        dispatch_for_generation(model.thinker)
        ret = flatten_obj(model.thinker(**inputs))
        _rets.append([x.cpu() for x in ret if isinstance(x, torch.Tensor)])
    rets = []
    for ret in zip(*_rets):
        rets.append(torch.cat(ret, dim=0))

    model.thinker.cpu()
    del model

    for batch_idx in tqdm(range(len(dataloader))):
        # inputs = {k: v.to(model_device) for k, v in inputs.items()}
        inputs = activations.fetch(batch_idx)
        dispatch_for_generation(ref_model.thinker)
        ref_ret = flatten_obj(ref_model.thinker(**inputs))
        _ref_rets.append([x.cpu() for x in ref_ret if isinstance(x, torch.Tensor)])

    ref_rets = []
    for ref_ret in zip(*_ref_rets):
        ref_rets.append(torch.cat(ref_ret, dim=0))

    # ref_model.thinker.cpu()
    del ref_model

    for i, (ret, ref_ret) in enumerate(zip(rets, ref_rets)):
        print("=" * 10 + f"{MODEL_ID} -Result {i}- {REF_MODEL_ID}" + "=" * 10)
        mse = torch.mean((ret - ref_ret) ** 2).item()
        print(f"MSE: {mse}")
        snr = (
            torch.mean(((ret - ref_ret) ** 2).sum(-1))
            / (torch.mean((ref_ret**2).sum(-1)) + 1e-7)
        ).item()
        print(f"SNR: {snr}")
        rel_error = torch.mean(
            torch.abs(ret - ref_ret) / (torch.abs(ref_ret) + 1e-8)
        ).item()
        print(f"Relative Error: {rel_error}")
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(ret, dim=-1),
            torch.nn.functional.softmax(ref_ret, dim=-1),
            reduction="batchmean",
        ).item()
        print(f"KL Divergence: {kl_div}")
        cos_sim = (
            torch.nn.functional.cosine_similarity(ret, ref_ret, dim=-1).mean().item()
        )
        print(f"Cosine Similarity: {cos_sim}")
        print("\n")
