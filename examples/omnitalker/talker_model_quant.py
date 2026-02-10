import contextlib
import copy
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import datasets
import torch
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_module
from compressed_tensors import get_execution_device
from compressed_tensors.quantization import (
    forward_quantize,
)
from datasets import load_dataset
from qwen_omni_utils import process_mm_info
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate
from transformers import AutoConfig, AutoProcessor

from llmcompressor.transformers.data import TextGenerationDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from qwen3_omni_moe_utils.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings
from llmcompressor.pipelines.sequential.helpers import SequentialTracer
from llmcompressor.utils import dispatch_for_generation, helpers

#################### configurations ####################
# Select model and load it.
# MODEL_ID = "/dataset/model_engine/qwen3omni_hf_v3_01000-liuding"
MODEL_ID = "/tmp/qwen3omni_hf_v3_01000-liuding-origin-quarot-sym-com-text-trans"

# recipe = "examples/omnitalker/configs/quarot.yaml"
recipe = "examples/omnitalker/configs/gptq.yaml"
# recipe = "examples/qwen3_omni_configs/text/mse_w4a8.yaml"
# flag = "quarot"
flag = "gptq"
# flag = "mse_w4a8"
fq = True  # False
realq = False
NUM_CALIBRATION_SAMPLES = 1 if flag == "quarot" else 512
bs = 1 if flag == "quarot" else 128

dtype = torch.float32

#################### configurations ####################


@TextGenerationDataset.register(name="list_of_dict", alias=["lod"])
class CustomDataset(TextGenerationDataset):

    def __call__(self, add_labels: bool = True):
        zh_ds = torch.load("/dataset/workspace/zhangl98/qwenomni-exp-talker/zh_all_inputs.pt", map_location="cpu")
        en_ds = torch.load("/dataset/workspace/zhangl98/qwenomni-exp-talker/en_all_inputs.pt", map_location="cpu")
        zh_ds = datasets.Dataset.from_list(zh_ds)
        en_ds = datasets.Dataset.from_list(en_ds)
        ds = datasets.concatenate_datasets([zh_ds, en_ds])
        ds = ds.shuffle(seed=42)
        return ds


from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("Qwen3OmniMoeTalkerTextSparseMoeBlock")
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
        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate

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

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )
        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return original
    

# DATASET_ID = "lmms-lab/flickr30k"
# DATASET_SPLIT = "test[:512]"
# DATASET_SPLIT = "test"
MAX_SEQUENCE_LENGTH = 2048

USE_AUDIO_IN_VIDEO = True

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config.enable_audio_output = True
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_ID, config=config, torch_dtype="auto"
)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)




IGNORE_INDEX = -100


@dataclass
class DataCollatorWithPadding:
    """
    Data collator with padding.
    """

    def __call__(self, features):
        batch = defaultdict(list)

        # batching features
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])
        keys = set(batch.keys())
        for key in keys:
            # process padding features
            if key in [
                "inputs_embeds",
                "position_embeddings",
                "attention_mask",
                "input_ids",
            ]:
                s = pad_sequence(
                    [torch.tensor(x).squeeze(0) for x in batch[key]],
                    batch_first=True,
                    padding_value=0,
                    padding_side="left",
                )
                if key == "inputs_embeds":
                    mask = [torch.ones(len(x[0]), dtype=torch.long) for x in batch[key]]
                    batch[key] = s.to(dtype)
                    batch["attention_mask"] = pad_sequence(
                        mask,
                        batch_first=True,
                        padding_value=0,
                        padding_side="left",
                    )
                else:
                    batch[key] = s
                    
            elif key == "position_ids":
                batch[key] = pad_sequence(
                    [torch.tensor(x).squeeze(1).T for x in batch[key]],
                    batch_first=False,
                    padding_value=0,
                    padding_side="left",
                ).transpose(0, 2)
            elif key in ["labels", "labels_image"]:
                batch[key] = pad_sequence(
                    [torch.tensor(x).squeeze(0) for x in batch[key]],
                    batch_first=True,
                    padding_value=IGNORE_INDEX,
                    padding_side="left",
                )
            else:
                batch[key] = default_collate(batch[key])

        return batch


data_collator = DataCollatorWithPadding()


@contextlib.contextmanager
def patch_model_name(model):
    model.model.codec_head = model.codec_head

    def hack_method(mod, *args, **kwargs):
        return mod.model.codec_head(*args, **kwargs)

    del model.codec_head
    model.codec_head = hack_method.__get__(model, type(model))

    yield

    del model.codec_head
    model.codec_head = model.model.codec_head
    del model.model.codec_head


@contextlib.contextmanager
def patch_model_thinker_emb(model):
    model.talker.thinker_embeddings = model.thinker.get_input_embeddings().to(dtype)

    yield
    model.thinker.model.embed_tokens = model.talker.thinker_embeddings

    del model.talker.thinker_embeddings


@contextlib.contextmanager
def patch_model_config(model):
    model.talker.config.im_start_token_id = model.config.im_start_token_id
    model.talker.config.tts_bos_token_id = model.config.tts_bos_token_id
    model.talker.config.tts_eos_token_id = model.config.tts_eos_token_id
    model.talker.config.tts_pad_token_id = model.config.tts_pad_token_id
    model.talker.config.user_token_id = model.config.user_token_id
    model.talker.config.assistant_token_id = model.config.assistant_token_id

    yield
    del model.talker.config.im_start_token_id
    del model.talker.config.tts_bos_token_id
    del model.talker.config.tts_eos_token_id
    del model.talker.config.tts_pad_token_id
    del model.talker.config.user_token_id
    del model.talker.config.assistant_token_id


# _tmp_config = copy.deepcopy(model.thinker.config)
# _tmp_config.update(model.thinker.config.text_config.to_dict())

for param in model.talker.parameters():
    param.requires_grad_(False)

model.talker.to(dtype)
_use_cache = model.talker.model.config.use_cache
model.talker.model.config.use_cache = False

with contextlib.ExitStack() as stack:
    if flag == "quarot":
        stack.enter_context(patch_model_name(model.talker))
    stack.enter_context(patch_model_thinker_emb(model))
    stack.enter_context(patch_model_config(model))
    stack.enter_context(torch.no_grad())
    # stack.enter_context(helpers.patch_attr(model.thinker, "config", _tmp_config))
    # Apply algorithms.
    oneshot(
        model=model.talker.model,  # TODO
        processor=model.config._name_or_path,
        dataset='lod',  # TODO
        recipe=recipe,
        tie_word_embeddings=True,
        data_collator=data_collator,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        sequential_targets=["Qwen3OmniMoeTalkerDecoderLayer"],
        batch_size=bs,
    )

model.talker.model.config.use_cache = _use_cache

_h = set()
transform_state_dict = OrderedDict()
from compressed_tensors.transform.factory.base import TransformBase

for name, module in model.talker.named_modules():
    if isinstance(module, TransformBase):
        if module in _h or id(module.scheme) in _h:
            continue
        _h.add((module if module.scheme.block_wise else id(module.scheme)))
        print(f"{name}: {module}")
        transform_state_dict.update({name: module.state_dict()})

to_removes = []
for name, module in model.talker.named_modules():
    for child_name, child_module in module.named_children():
        if isinstance(child_module, TransformBase):
            to_removes.append((module, child_name))
for module, child_name in to_removes:
    delattr(module, child_name)
# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
# dispatch_for_generation(model)
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
print("==========================================\n\n")

from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.utils.match import match_named_modules

SAVE_DIR = (
    "/tmp/"
    + MODEL_ID.rstrip("/").split("/")[-1]
    + f"-{flag}-sym-com-text"
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


from llmcompressor.recipe import Recipe

recipe = Recipe.create_instance(path_or_modifiers=recipe, target_stage=None)

quantized_name_set = set()
import re

for _, module in match_named_modules(
    model.talker.model, recipe.modifiers[-1].resolved_targets, recipe.modifiers[-1].ignore
):
    if hasattr(module, "quantization_status"):
        assert module.quantization_status == QuantizationStatus.FROZEN, (
            f"{module.quantization_status}"
        )
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
