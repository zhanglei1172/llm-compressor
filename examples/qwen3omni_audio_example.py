import contextlib
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
    _get_feat_extract_output_lengths,
)
import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping
from llmcompressor.utils import dispatch_for_generation, helpers
from llmcompressor.transformers.compression.compressed_tensors_utils import modify_save_pretrained

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
recipe = "examples/qwen3_omni_configs/audio/gptq.yaml"
# recipe = "examples/qwen3_omni_configs/audio/awq.yaml"
flag = "gptq"
# flag = "awq"
#################### configurations ####################

# Select model and load it.
MODEL_ID = "/dataset/workspace/zhangl98/models/Qwen3-Omni-30B-A3B-Instruct/"

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
dtype = model.dtype
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
    conversation = [{
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
        ]
    }]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    ret = processor(text=text, 
                   audio=audios, 
                   images=images, 
                   videos=videos, 
                   return_tensors="pt", 
                   padding=True, 
                   use_audio_in_video=USE_AUDIO_IN_VIDEO)
    # ret = {
    #     k: v.astype(dtype) if torch.is_floating_point(v) else v
    #     for k, v in ret.items()
    #     }
    del ret['input_ids'], ret['attention_mask']
    return ret


ds = ds.map(preprocess, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    ret = {key: torch.tensor(value, dtype=dtype if key== "input_features" else None) for key, value in batch[0].items()}
    audio_feature_lengths = torch.sum(ret['feature_attention_mask'], dim=1)
    input_features = ret['input_features'].permute(0, 2, 1)[ret['feature_attention_mask'].bool()].permute(1, 0)
    feature_lens = audio_feature_lengths if audio_feature_lengths is not None else ret['feature_attention_mask'].sum(-1)
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


def my_wrap(self, feature_lens, input_features):
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

    chunk_lengths = torch.tensor(
        [self.n_window * 2] * chunk_num.sum(),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
    chunk_lengths[chunk_lengths == 0] = self.n_window * 2

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
        [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
        batch_first=True,
    )
    padded_feature = padded_feature.unsqueeze(1)
    return aftercnn_lens, padded_feature, padded_mask_after_cnn

# @torch.fx.wrap
def my_wrap_0(self, padded_feature):
    padded_embeds = []
    for chunk in padded_feature.split(self.conv_chunksize, dim=0):
    # padded_embed = F.gelu(self.conv2d1(padded_feature))
        padded_embed = F.gelu(self.conv2d1(chunk))
        padded_embed = F.gelu(self.conv2d2(padded_embed))
        padded_embed = F.gelu(self.conv2d3(padded_embed))
        padded_embeds.append(padded_embed)
    return torch.cat(padded_embeds, dim=0)

# @torch.fx.wrap
def my_wrap_1(self, aftercnn_lens, padded_mask_after_cnn):
    cu_chunk_lens = [0]
    window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
    for cnn_len in aftercnn_lens:
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]
    cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)
    return cu_seqlens

from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F
import torch.nn as nn

def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
):
        aftercnn_lens, padded_feature, padded_mask_after_cnn = my_wrap(self, feature_lens, input_features)
        # Split to chunk to avoid OOM during convolution
        # padded_embeds = []
        # for chunk in padded_feature.split(self.conv_chunksize, dim=0):
        #     padded_embed = F.gelu(self.conv2d1(chunk))
        #     padded_embed = F.gelu(self.conv2d2(padded_embed))
        #     padded_embed = F.gelu(self.conv2d3(padded_embed))
        #     padded_embeds.append(padded_embed)
        # padded_embed = torch.cat(padded_embeds, dim=0)
        padded_embed = my_wrap_0(self, padded_feature)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = my_wrap_1(self, aftercnn_lens, padded_mask_after_cnn)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)
import sys
sys.modules[model.thinker.audio_tower.__class__.__module__].__dict__.update({
    "forward": forward,
    "my_wrap_0": my_wrap_0,
    "my_wrap_1": my_wrap_1,
    "my_wrap": my_wrap,
})

with contextlib.ExitStack() as stack:
    stack.enter_context(
        helpers.patch_attr(model.thinker.audio_tower, "forward", forward)
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

sys.modules[model.thinker.audio_tower.__class__.__module__].__dict__.update({
    "forward": model.thinker.audio_tower.forward,
})

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

SAVE_DIR = "/tmp/" + MODEL_ID.rstrip("/").split("/")[-1] + f"-{flag}-sym-com-audio"
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
        if isinstance(module, torch.nn.Linear):
            module.weight.data = forward_quantize(
                    module, module.weight, "weight", scheme.weights
                )
        elif isinstance(module, torch.nn.Conv2d):
            module.weight_scale.data = module.weight_scale.data.unsqueeze(-1).unsqueeze(-1)
            module.weight_zero_point.data = module.weight_zero_point.data.unsqueeze(-1).unsqueeze(-1)
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
model.save_pretrained(SAVE_DIR+"-fq")#, save_compressed=True) # fakequant
processor.save_pretrained(SAVE_DIR+"-fq")
