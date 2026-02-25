import argparse
import base64
import contextlib
import copy
import datetime
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from accelerate import init_empty_weights
from accelerate.hooks import remove_hook_from_module
from compressed_tensors import get_execution_device, match_modules_set
from compressed_tensors.quantization import (
    enable_quantization,
)
from easydict import EasyDict
from loguru import logger
from qwen_omni_utils import process_mm_info
from torch.distributed.fsdp import (
    FullStateDictConfig,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as PT_FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoFeatureExtractor, AutoProcessor, MimiModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from qwen3_omni_moe_utils.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)
from qwen3_omni_moe_utils.qwen3_omni_codec_decode import load_model_from_config
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from llmcompressor import oneshot
from llmcompressor.core.state import State
from llmcompressor.modeling.qwen3_omni_moe import replace_vit_attention_inv
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings
from llmcompressor.pipelines.sequential.helpers import SequentialTracer
from llmcompressor.train.fsdp_trainer import MyTrainer
from llmcompressor.train.train_utils import LLMCTrainingArguments, TeacherModel
from llmcompressor.utils import helpers
from llmcompressor.utils.pytorch.module import (
    build_weight_tied_map_with_unionfind,
    patch_module_non_persistent_buffers,
)

torch.fx.experimental._config.meta_nonzero_assume_all_nonzero = True
USE_AUDIO_IN_VIDEO = True


mappings.SPINQUANT_MAPPING_REGISTRY["Qwen3OmniMoeTalkerForConditionalGeneration"] = (
    mappings.SpinQuantMapping(
        mm_proj=[
            r"re:.*text_projection\.linear_fc2$",
            r"re:.*hidden_projection\.linear_fc2$",
        ],
        embedding=r"re:(model\.codec_embedding|.*codec_embedding\.\d+)$",
        attn="re:.*self_attn$",
        attn_q="re:.*model.*q_proj$",
        attn_k="re:.*model.*k_proj$",
        attn_v="re:.*model.*v_proj$",
        attn_o="re:.*model.*o_proj$",
        mlp_in=[r"re:.*up_proj", r"re:.*mlp.*gate.*"],
        mlp_out=[r"re:.*mlp.*down_proj$"],
        lm_head=r"re:.*(lm_head\.\d+|codec_head)$",
    )  # 2+16emb + 16head + (4*20+4*5)qkvo + 129*20+5 u + 131*20+5 g + (129)*20 + 5 d
)
norm_mappings.NORM_MAPPING_REGISTRY["Qwen3OmniMoeTalkerForConditionalGeneration"] = [
    norm_mappings.NormMapping(
        norm="re:.*model.*input_layernorm$",
        linears=["re:.*model.*q_proj$", "re:.*model.*k_proj$", "re:.*model.*v_proj$"],
    ),
    norm_mappings.NormMapping(
        norm="re:.*model.*post_attention_layernorm$",
        linears=[r"re:.*mlp.*gate.*", r"re:.*mlp.*up_proj"],
    ),
    norm_mappings.NormMapping(
        norm=r"re:.*model\.norm",
        linears=[r"re:.*(lm_head\.\d+|codec_head)$"],
        # model.model.norm - model.codec_head
        # model.code_predictor.model.norm - model.code_predictor.lm_head.[0-14]
    ),
]


#################### configurations ####################
# Select model and load it.
ENABLE_SMOOTH = False
pretrain = "origin"
flag = "spinquant"
NUM_CALIBRATION_SAMPLES = 150 * 2 * 4 + 16
enable_modality = {"talker"}
model_dtype = torch.bfloat16
jsonl_path = Path("/dataset/workspace/zhangl98/dataset/talker/processed/train.jsonl")
MIMI_REPO_ID = "/dataset/workspace/zhangl98/models/mimi"
NUM_CODE_GROUPS = 16
code2wav_model_path = "/dataset/model_engine/Qwen3-codec-decode/ckpt/0905_ckpt/"
#################### configurations ####################

SPEAKER = "f245"

if pretrain == "ostq":
    MODEL_ID = ""
else:
    # MODEL_ID = "/dataset/workspace/zhangl98/models/Qwen3-Omni-30B-A3B-Instruct/"
    MODEL_ID = "/dataset/model_engine/qwen3omni_hf_v3_01000-liuding"

flag += str(tuple(enable_modality)).replace("'", "")

SAVE_DIR = (
    "/tmp/" + MODEL_ID.rstrip("/").split("/")[-1] + f"-{pretrain}-{flag}" + "-trans"
)

MAX_SEQUENCE_LENGTH = 2048


def encode_base64_img(img) -> str:
    with BytesIO() as buffer:
        img.save(buffer, format="PNG")
        data = buffer.getvalue()

    return base64.b64encode(data).decode("utf-8")


def encode_base64_audio(audio_array: np.ndarray, sampling_rate: int) -> str:
    with BytesIO() as buffer:
        sf.write(buffer, audio_array, samplerate=sampling_rate, format="WAV")
        data = buffer.getvalue()

    return base64.b64encode(data).decode("utf-8")


class TTSDataset(Dataset):
    """Dataset for TTS training"""

    def __init__(self, jsonl_path: Path, max_samples: int | None = None):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data = json.loads(line)

                if "messages" in data and "audios" in data:
                    self.samples.append(
                        {
                            "messages": data["messages"],
                            "audio_path": Path(data["audios"][0]),
                            "speaker": data.get("speaker", SPEAKER),
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if "talker" in enable_modality:
    ds = TTSDataset(jsonl_path=jsonl_path)


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


@contextlib.contextmanager
def patch_model_thinker_emb(model):
    model.talker.thinker_embeddings = model.thinker.get_input_embeddings().to(
        model_dtype
    )

    yield
    model.thinker.model.embed_tokens = model.talker.thinker_embeddings

    del model.talker.thinker_embeddings


@torch.no_grad()
def pre_trans_talker(model):
    # session = active_session()
    # session.reset()
    state = State()
    state.update(
        model=model.talker,
    )
    recipe_ = [
        SpinQuantModifier(
            backe_mean=False,
            learnable=True,
            rotations=["R1", "R2"],
            transform_block_size_R1=1024,
            transform_type="random-hadamard",
            sequential_onload=not RANK_OTHER,
        )
    ]

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch_model_name(model.talker))
        stack.enter_context(patch_model_thinker_emb(model))
        stack.enter_context(torch.no_grad())
        for mod in recipe_:
            mod.on_initialize(state=state)
        # for param in model.talker.model.parameters():
        #     param.requires_grad = False
        recipe_[0].on_start(state=state, event=None)

    class SmoothTransform(torch.nn.Module):
        def __init__(self, dim, trans=False):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(dim) * 1.0)
            self.trans = trans

        def forward(self, x, inverse=False):
            if inverse:
                scale = 1 / self.scale.to(x.device)
            else:
                scale = self.scale.to(x.device)
            if not self.trans and x.dim() > 1:
                scale = scale.view(-1, 1)
            return (x.to(self.scale.dtype) * scale).to(x.dtype)

        def right_inverse(self, x):
            return self.forward(x, inverse=True)

    if ENABLE_SMOOTH:
        import torch.nn.utils.parametrize as P
        from compressed_tensors.utils import (
            align_module_device,
            update_offload_parameter,
        )
        from torch.nn.utils.parametrize import is_parametrized

        for up_projs, down_projs in match_modules_set(
            model.talker, (r"re:.*up_proj$", r"re:.*down_proj$")
        ):
            assert len(up_projs) == 1
            assert len(down_projs) == 1
            up_proj = up_projs[0]
            down_proj = down_projs[0]
            transform = SmoothTransform(up_proj.out_features).to(
                torch.cuda.current_device()
            )
            transform_inv = SmoothTransform(down_proj.in_features, trans=True).to(
                torch.cuda.current_device()
            )
            transform_inv.scale = transform.scale
            with (
                torch.no_grad(),
                align_module_device(up_proj),
                align_module_device(down_proj),
            ):
                if not is_parametrized(up_proj, "weight"):
                    update_offload_parameter(
                        up_proj, "weight", transform(up_proj.weight)
                    )
                P.register_parametrization(up_proj, "weight", transform)
                if hasattr(up_proj, "bias") and up_proj.bias is not None:
                    if not is_parametrized(up_proj, "bias"):
                        update_offload_parameter(
                            up_proj, "bias", transform(up_proj.bias)
                        )
                    P.register_parametrization(up_proj, "bias", transform)
                if not is_parametrized(down_proj, "weight"):
                    update_offload_parameter(
                        down_proj, "weight", transform_inv(down_proj.weight)
                    )
                P.register_parametrization(down_proj, "weight", transform_inv)

    return state, recipe_, model


@torch.no_grad()
def pre_compression_talker(model):
    state = State()
    state.update(
        model=model.talker,
    )
    ignore = ["lm_head"]
    recipe_ = [
        QuantizationModifier(
            ignore=ignore,
            config_groups={
                "group_0": {
                    "weights": {
                        "observer": "minmax",
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "dynamic": True,
                    },
                    "input_activations": {
                        "observer": "minmax",
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "tensor",
                        "dynamic": True,
                    },
                    "targets": [
                        r"re:^model.*up_proj$",
                        r"re:^model.*gate_proj$",
                        r"re:^model.*q_proj$",
                        r"re:^model.*k_proj$",
                        r"re:^model.*v_proj$",
                        r"re:^model.*o_proj$",
                        r"re:^model.*out_proj$",
                        # r"re:^model.*gate$",
                    ],
                    "ste": True,
                },
                "group_1": {
                    "weights": {
                        "observer": "minmax",
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "dynamic": True,
                    },
                    "input_activations": {
                        "observer": "minmax",
                        "num_bits": 16,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "tensor",
                        "dynamic": True,
                    },
                    "targets": [
                        r"re:^model.*down_proj$",
                    ],
                    "ste": True,
                },
                "group_2": {
                    "weights": {
                        "observer": "minmax",
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "dynamic": True,
                    },
                    "input_activations": {
                        "observer": "minmax",
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "tensor",
                        "dynamic": True,
                    },
                    "targets": [
                        r"re:^code_predictor.*up_proj$",
                        r"re:^code_predictor.*gate_proj$",
                        r"re:^code_predictor.*q_proj$",
                        r"re:^code_predictor.*k_proj$",
                        r"re:^code_predictor.*v_proj$",
                        r"re:^code_predictor.*o_proj$",
                        r"re:^code_predictor.*out_proj$",
                    ],
                    "ste": True,
                },
                "group_3": {
                    "weights": {
                        "observer": "minmax",
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "dynamic": True,
                    },
                    "input_activations": {
                        "observer": "minmax",
                        "num_bits": 16,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "tensor",
                        "dynamic": True,
                    },
                    "targets": [
                        r"re:^code_predictor.*down_proj$",
                    ],
                    "ste": True,
                },
            },
        )
    ]

    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.nn.utils.parametrize.cached())
        for mod in recipe_:
            mod.on_initialize(state=state)
        # for param in model.talker.model.parameters():
        #     param.requires_grad = False
        model.talker.apply(enable_quantization)

    return state, recipe_, model


def encode_audio_to_codes(audio_path: Path, feature_extractor, mimi_model, device):
    """Convert audio file to codec codes"""
    # Read audio and resample to 24kHz if needed
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 24kHz if needed (Mimi requires 24kHz)
    target_sr = 24000
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    audio_inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
    audio_tensor = torch.as_tensor(audio_inputs["input_values"], dtype=mimi_model.dtype)

    if audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.unsqueeze(1)

    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        codes = mimi_model.encode(audio_tensor).audio_codes

    return codes.to(device)


def collate_fn(
    batch_samples, processor, feature_extractor, mimi_model, device, code2wav
):
    """Collate multiple samples into a batch - returns CPU tensors for DataLoader"""

    # Prepare text inputs
    formatted_texts = []
    speakers = []
    audio_paths = []
    conversations = []

    for sample in batch_samples:
        formatted_text = processor.apply_chat_template(
            sample["messages"], add_generation_prompt=False, tokenize=False
        )
        formatted_texts.append(formatted_text)
        speakers.append(sample["speaker"])
        audio_paths.append(sample["audio_path"])
        conversations.append(sample["messages"])

    # Batch process texts (returns CPU tensors)
    batch = processor(
        text=formatted_texts,
        audio=None,
        images=None,
        videos=None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )

    # Encode all audios and move back to CPU
    target_codes_list = []
    for audio_path in audio_paths:
        codes = encode_audio_to_codes(audio_path, feature_extractor, mimi_model, device)
        codes = align_codebook_dim(codes, len(code2wav.code_embeddings))
        # Move to CPU for DataLoader
        target_codes_list.append(codes.cpu())

    # Find max audio length
    max_audio_len = max(codes.shape[2] for codes in target_codes_list)

    # Pad audio codes to same length (on CPU)
    padded_codes = []
    audio_lengths = []
    for codes in target_codes_list:
        audio_lengths.append(codes.shape[2])
        if codes.shape[2] < max_audio_len:
            padding = torch.zeros(
                (codes.shape[0], codes.shape[1], max_audio_len - codes.shape[2]),
                dtype=codes.dtype,
            )
            codes = torch.cat([codes, padding], dim=2)
        padded_codes.append(codes)

    # Stack into batch (on CPU)
    target_codes = torch.cat(padded_codes, dim=0)

    return {
        "batch": batch,
        "target_codes": target_codes,
        "speakers": speakers,
        "conversations": conversations,
        "audio_lengths": audio_lengths,
    }


def pt_fsdp_state_dict(model: torch.nn.Module):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with PT_FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        return model.state_dict()


def setup():
    # initialize the process group
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def align_codebook_dim(codes: torch.Tensor, target_quantizers: int):
    """Align codec codes to target quantizer dimension"""
    current = codes.shape[1]
    if current == target_quantizers:
        return codes
    if current < target_quantizers:
        raise ValueError(
            f"Codes have {current} quantizers but need {target_quantizers}"
        )
    return codes[:, :target_quantizers, :]


def build_talker_prefix_tts(
    model, thinker_outputs, input_ids, speaker_name, device, batch_idx=0
):
    """Build Talker prefix for TTS - works with batched inputs"""
    config = model.config

    # Extract for specific batch index
    thinker_embed = thinker_outputs.hidden_states[0][batch_idx : batch_idx + 1].to(
        device
    )
    accept_layer = config.talker_config.accept_hidden_layer
    thinker_hidden = thinker_outputs.hidden_states[accept_layer][
        batch_idx : batch_idx + 1
    ].to(device)
    sample_input_ids = input_ids[batch_idx : batch_idx + 1]

    im_start_positions = torch.nonzero(
        sample_input_ids[0] == config.im_start_token_id
    ).view(-1)
    im_start_indexes = torch.cat(
        (im_start_positions, torch.tensor([sample_input_ids.shape[1]], device=device)),
        dim=0,
    )

    # Multimodal mask (all False for text-only TTS)
    multimodal_mask = torch.zeros_like(sample_input_ids, dtype=torch.bool).to(device)

    # Special tokens for Talker
    talker_special_tokens = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device=device,
        dtype=sample_input_ids.dtype,
    )

    thinker_embeddings = model.thinker.get_input_embeddings()
    if hasattr(thinker_embeddings, "base_layer"):
        thinker_embeddings = thinker_embeddings.base_layer

    tts_bos_embed, tts_eos_embed, tts_pad_embed = (
        model.talker.text_projection(thinker_embeddings(talker_special_tokens))
        .to(device)
        .chunk(3, dim=1)
    )

    speaker_id = config.talker_config.spk_id.get(speaker_name.lower())
    if speaker_id is None:
        raise ValueError(f"Speaker {speaker_name} not found")

    talker_input_embeds, talker_input_ids = [], []
    trailing_text_hidden = None

    for i in range(len(im_start_indexes) - 1):
        im_start_index = im_start_indexes[i]
        segment_end_index = im_start_indexes[i + 1]
        role_token = sample_input_ids[0][im_start_index + 1]

        if role_token == config.user_token_id:
            user_part = model._get_talker_user_parts(
                im_start_index,
                segment_end_index,
                multimodal_mask,
                thinker_hidden,
                thinker_embed,
            )
            talker_input_embeds.append(user_part)
            talker_input_ids.append(
                sample_input_ids[:, im_start_index:segment_end_index]
            )

        elif role_token == config.assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_embeds, assistant_ids, trailing_text_hidden = (
                model._get_talker_assistant_parts(
                    im_start_index,
                    segment_end_index,
                    speaker_id,
                    thinker_embed,
                    tts_pad_embed,
                    tts_bos_embed,
                    tts_eos_embed,
                )
            )
            talker_input_embeds.append(assistant_embeds)
            talker_input_ids.append(assistant_ids)

    if trailing_text_hidden is None:
        raise RuntimeError("Failed to build trailing_text_hidden")

    talker_input_embed = torch.cat(
        [embed.to(device) for embed in talker_input_embeds], dim=1
    )
    talker_input_id = torch.cat([ids.to(device) for ids in talker_input_ids], dim=1)

    return (
        talker_input_embed,
        talker_input_id,
        trailing_text_hidden.to(device),
        tts_pad_embed,
    )


def fsdp_main(model, config):
    feature_extractor = AutoFeatureExtractor.from_pretrained(MIMI_REPO_ID)
    mimi_model = MimiModel.from_pretrained(MIMI_REPO_ID, torch_dtype=model_dtype).to(
        torch.cuda.current_device()
    )
    config_path = code2wav_model_path + "/config.yaml"
    checkpoint_path = code2wav_model_path + "/model_weights.pt"
    code2wav = load_model_from_config(config_path, checkpoint_path, device="cuda")
    training_params_cnt = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            training_params_cnt += 1
    weight_tied_name_map = build_weight_tied_map_with_unionfind(model)

    state_dict = model.state_dict()
    model.to("meta")
    model_to_train = copy.deepcopy(model)
    if not RANK_OTHER:
        model_to_train.load_state_dict(state_dict, assign=True)
    del state_dict

    # fsdp
    train_processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        model_max_length=MAX_SEQUENCE_LENGTH,
        padding_side="left",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    train_args = LLMCTrainingArguments(**config.train_args)
    need_teacher = train_args.special.get("loss_type", "origin") not in (
        "origin",
        "DFT",
    )
    model_to_train.train()
    if need_teacher:
        model_path = train_args.special.get("teacher_path", MODEL_ID)

        teacher_model, _ = dist_load_model(model_path)
        teacher_model = teacher_model.talker
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.text_config.use_cache = False
        model_to_train.teacher = TeacherModel(teacher_model)
    # Now you can train the model
    # model_to_train.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model_to_train.thinker.config.text_config.use_cache = False  # make activation ckpt
    assert len(set(weight_tied_name_map.values())) == training_params_cnt

    def collate_wrapper(batch):
        return collate_fn(
            batch,
            train_processor,
            feature_extractor,
            mimi_model,
            torch.cuda.current_device(),
            code2wav,
        )

    class TalkerTrainer(MyTrainer):
        @torch.compile(fullgraph=False, disable=True)
        def compute_loss(self, model, inputs, **kwargs):
            device = torch.cuda.current_device()
            batch = inputs["batch"]
            target_codes = inputs["target_codes"]
            speakers = inputs["speakers"]
            audio_lengths = inputs["audio_lengths"]
            """Compute loss for a single sample in the batch"""

            # Forward Thinker (frozen)
            with torch.no_grad():
                thinker_outputs = model.thinker(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    output_hidden_states=True,
                    return_dict=True,
                )
            model.thinker._is_root = False
            # Process each sample in batch
            batch_size_actual = batch["input_ids"].shape[0]
            batch_talker_loss = 0.0
            batch_mtp_loss = 0.0
            # with FSDP.summon_full_params(model, writeback=True, recurse=False):
            for i in range(batch_size_actual):
                # Build Talker prefix
                (
                    talker_input_embed,
                    talker_input_ids,
                    trailing_text_hidden,
                    tts_pad_embed,
                ) = build_talker_prefix_tts(
                    model=model,
                    thinker_outputs=thinker_outputs,
                    input_ids=batch["input_ids"],
                    speaker_name=speakers[i],
                    device=device,
                    batch_idx=i,
                )

                # Extract this sample's codes
                sample_codes = target_codes[i : i + 1, :, : audio_lengths[i]]

                # Prepare Talker training (Layer 0)
                layer0_codes = sample_codes[:, 0, :]
                num_codec_tokens = layer0_codes.shape[1]

                layer0_embeds = model.talker.get_input_embeddings()(
                    layer0_codes.to(device)
                )
                predictor_embeds = (
                    model.talker.code_predictor.get_input_embeddings()
                )
                if hasattr(predictor_embeds, "_orig_mod"):
                    predictor_embeds = predictor_embeds._orig_mod

                # Sum all layer embeddings
                all_layer_embeds_sum = layer0_embeds.clone()
                for j in range(len(predictor_embeds)):
                    layer_j_codes = sample_codes[:, j + 1, :]
                    emb = predictor_embeds[j](layer_j_codes.to(device))
                    all_layer_embeds_sum = all_layer_embeds_sum + emb

                # Build shifted inputs (teacher forcing)
                text_len = trailing_text_hidden.shape[1]
                codec_input_embeds_list = []

                for pos in range(num_codec_tokens):
                    if pos == 0:
                        continue
                    prev_pos = pos - 1
                    text_hidden = (
                        trailing_text_hidden[:, prev_pos : prev_pos + 1, :]
                        if prev_pos < text_len
                        else tts_pad_embed
                    )
                    pos_embed = (
                        all_layer_embeds_sum[:, prev_pos : prev_pos + 1, :]
                        + text_hidden
                    )
                    codec_input_embeds_list.append(pos_embed)

                # EOS input
                last_pos = num_codec_tokens - 1
                eos_text_hidden = (
                    trailing_text_hidden[:, last_pos : last_pos + 1, :]
                    if last_pos < text_len
                    else tts_pad_embed
                )
                eos_input_embed = (
                    all_layer_embeds_sum[:, last_pos : last_pos + 1, :]
                    + eos_text_hidden
                )
                codec_input_embeds_list.append(eos_input_embed)

                # Concatenate
                if codec_input_embeds_list:
                    codec_input_embeds = torch.cat(
                        codec_input_embeds_list, dim=1
                    ).to(model_dtype)
                    full_inputs_embeds = torch.cat(
                        [talker_input_embed, codec_input_embeds], dim=1
                    )
                else:
                    full_inputs_embeds = talker_input_embed

                # Labels
                prefix_len = talker_input_embed.shape[1]
                labels_prefix = torch.full(
                    (1, prefix_len - 1), -100, dtype=torch.long, device=device
                )
                labels_code = layer0_codes.to(device)
                codec_eos_id = model.config.talker_config.codec_eos_token_id
                labels_eos = torch.tensor([[codec_eos_id]], device=device)
                labels = torch.cat([labels_prefix, labels_code, labels_eos], dim=1)

                # Attention mask
                seq_len = full_inputs_embeds.shape[1]
                attention_mask = torch.ones(
                    (1, seq_len), dtype=torch.long, device=device
                )

                # Prefill
                if (
                    full_inputs_embeds is not None
                    and full_inputs_embeds.shape[1] > 1
                ):
                    generation_step = -1
                    residual_codes = None
                if attention_mask is not None:
                    delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                    position_ids, rope_deltas = model.talker.get_rope_index(
                        talker_input_ids,
                        None,
                        None,
                        attention_mask,
                        None,
                        None,
                        None,
                    )
                    rope_deltas = rope_deltas - delta0
                    model.talker.rope_deltas = rope_deltas

                outputs = model.talker.model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=full_inputs_embeds,
                    use_cache=False,
                    output_router_logits=None,
                    cache_position=None,
                    output_hidden_states=True,
                )
                hidden_states = outputs.last_hidden_state
                logits = model.talker.codec_head(hidden_states)
                # Forward Talker
                # talker_outputs = model.talker(
                #     inputs_embeds=full_inputs_embeds,
                #     attention_mask=attention_mask,
                #     trailing_text_hidden=trailing_text_hidden,
                #     tts_pad_embed=tts_pad_embed,
                #     output_hidden_states=True,
                #     return_dict=True,
                # )
                hidden_states = (
                    outputs.hidden_states,
                    residual_codes,
                )

                # Compute Talker loss
                talker_logits = logits

                shift_logits = talker_logits[:, :, :].contiguous()
                shift_labels = labels[:, :].contiguous()

                talker_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                # MTP Training (Layers 1-15)
                # talker_hidden = (
                #     hidden_states[0][-1]
                #     if isinstance(hidden_states, tuple)
                #     else hidden_states[-1]
                # )

                # codec_hidden_start = prefix_len - 1
                # codec_hidden_end = prefix_len - 1 + num_codec_tokens
                # codec_hidden = talker_hidden[
                #     :, codec_hidden_start:codec_hidden_end, :
                # ]

                # mtp_total_loss = 0.0
                # code_predictor = model.talker.code_predictor
                # hidden_dim = codec_hidden.shape[2]
                # num_mtp_layers = NUM_CODE_GROUPS - 1

                # layer0_embed_for_mtp = model.talker.get_input_embeddings()(
                #     layer0_codes.to(device)
                # )
                # hidden_flat = codec_hidden.reshape(-1, 1, hidden_dim)
                # layer0_flat = layer0_embed_for_mtp.reshape(-1, 1, hidden_dim)

                # for mtp_layer_idx in range(num_mtp_layers):
                #     embed_list = [hidden_flat, layer0_flat]

                #     for prev_layer in range(mtp_layer_idx):
                #         prev_codes = sample_codes[:, prev_layer + 1, :].to(device)
                #         prev_embed = predictor_embeds[prev_layer](prev_codes)
                #         prev_embed_flat = prev_embed.reshape(-1, 1, hidden_dim)
                #         embed_list.append(prev_embed_flat)

                #     mtp_inputs = torch.cat(embed_list, dim=1).to(model_dtype)
                #     target_layer_codes = sample_codes[:, mtp_layer_idx + 1, :].to(
                #         device
                #     )
                #     target_labels = target_layer_codes.reshape(-1)

                #     mtp_outputs = code_predictor(
                #         inputs_embeds=mtp_inputs,
                #         generation_steps=mtp_layer_idx,
                #         use_cache=False,
                #     )

                #     mtp_logits = mtp_outputs.logits[:, -1, :]
                #     mtp_layer_loss = F.cross_entropy(mtp_logits, target_labels)
                #     mtp_total_loss += mtp_layer_loss

                # mtp_avg_loss = mtp_total_loss / num_mtp_layers
                batch_talker_loss += talker_loss
                # batch_mtp_loss += mtp_avg_loss
            avg_talker_loss = batch_talker_loss / batch_size_actual
            # avg_mtp_loss = batch_mtp_loss / batch_size_actual
            return (avg_talker_loss ) #+ 2.0 * avg_mtp_loss)

    trainer = TalkerTrainer(
        model=model_to_train,
        # tokenizer=train_processor.tokenizer,
        args=train_args,
        train_dataset=ds,
        eval_dataset=None,
        # data_collator=default_data_collator,
        data_collator=collate_wrapper,
        # optimizers=(optimizer, None),
        # optimizers=(None, None),
        # ignored_modules=ignored_modules,
        weight_tied_name_map=weight_tied_name_map,
        ignored_modules=[
            model_to_train.thinker.get_input_embeddings(),
            model_to_train.talker.get_input_embeddings(),
            # model_to_train.talker.code_predictor.get_input_embeddings(),
            model_to_train.talker.codec_head,
            model_to_train.talker.code_predictor,
            model_to_train.talker.text_projection,
            # model.talker.model,
            # model.thinker
            ],
    )
    with patch_model_thinker_emb(model_to_train):
        trainer.train()
    dist.barrier()
    if hasattr(trainer.model, "_orig_mod"):
        unwrapped_model = trainer.model._orig_mod
    else:
        unwrapped_model = trainer.model
    state_dict = trainer.accelerator.get_state_dict(unwrapped_model)
    if not RANK_OTHER:
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("teacher")
        }
        model.load_state_dict(state_dict, assign=True)
        trainer.register_tied_parameters(model, weight_tied_name_map)


@torch.no_grad()
def post_trans_talker(model):
    pass


@torch.no_grad()
def post_compression_talker(state, recipe_, model, processor):
    # recipe_[0].on_end(state=state, event=None)
    from collections import OrderedDict

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

    print("==========================================\n\n")

    from compressed_tensors.utils.match import match_named_modules

    quantized_name_set = set()
    import re

    for _, module in match_named_modules(
        model, recipe_[-1].resolved_targets, recipe_[-1].ignore
    ):
        if hasattr(module, "quantization_status"):
            quantized_name_set.add(re.sub(r"\d+", "X", _))
            scheme = getattr(module, "quantization_scheme", None)

            delattr(module, "quantization_status")
            delattr(module, "quantization_enabled")
            delattr(module, "quantization_scheme")
            for key in list(module._parameters.keys()):
                if key.endswith("_scale") or key.endswith("_zero_point"):
                    delattr(module, key)
    print(f"Total quantized modules: {quantized_name_set}")
    post_compression_talker(state, recipe_, model, processor)

    model.save_pretrained(SAVE_DIR)  # , save_compressed=True) # fakequant
    processor.save_pretrained(SAVE_DIR)
    torch.save(transform_state_dict, f"{SAVE_DIR}/transform_state_dict.pt")
    print(SAVE_DIR)


def dist_load_model(model_path=MODEL_ID, load_processor=False):
    processor = None
    if RANK_OTHER:
        with init_empty_weights():
            model = Qwen3OmniMoeForConditionalGeneration._from_config(
                model_config,
                # trust_remote_code=True,
                dtype=model_dtype,
                # low_cpu_mem_usage=True,
                # attn_implementation=ATTN_IMPL,
            )
    else:
        if load_processor:
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, config=model_config, torch_dtype=model_dtype
        )
    return model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        config = EasyDict(config)
    setup()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    RANK_OTHER = dist.is_initialized() and dist.get_rank() != 0
    model_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    # model_config.enable_audio_output = True
    if RANK_OTHER:
        logger.remove()
    model, processor = dist_load_model(load_processor=True)
    # dist.barrier()
    with patch_module_non_persistent_buffers(model):
        model.eval()
        # no_grad for compression
        for param in model.parameters():
            param.requires_grad = False
        state_text, recipe_trans, model = pre_trans_talker(model)

        state, recipe_, model = pre_compression_talker(model)
        dist.barrier()
        fsdp_main(model, config)

    if not RANK_OTHER:
        recipe_trans[0]._fold_transforms_into_weights(state_text.model)

        post_compression_talker(state, recipe_, model, processor)
    cleanup()
