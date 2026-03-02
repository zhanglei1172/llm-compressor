import contextlib
import functools
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import soundfile as sf
import torch
from compressed_tensors.offload.dispatch import (
    offload_model,
)
from compressed_tensors.transform import TransformLocation
from compressed_tensors.transform.factory.base import TransformBase
from compressed_tensors.transform.utils.hadamard import random_hadamard_matrix
from compressed_tensors.transform.utils.matrix import apply_transform_weight
from compressed_tensors.utils import (
    remove_dispatch,
    update_offload_parameter,
)
from torch.nn.utils.parametrize import is_parametrized
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoFeatureExtractor, AutoProcessor, MimiModel

from llmcompressor.train.fsdp_trainer import MyTrainer
from llmcompressor.utils import dispatch_for_generation

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from qwen3_omni_moe_utils.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)
from qwen3_omni_moe_utils.qwen3_omni_codec_decode import load_model_from_config

from llmcompressor.core.state import State
from llmcompressor.modeling.qwen3_omni_moe import replace_vit_attention_inv
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.modifiers.transform.spinquant import mappings, norm_mappings
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
pretrain = "origin"
flag = "dartquant"
NUM_CALIBRATION_SAMPLES = 256
enable_modality = {"talker"}
model_dtype = torch.bfloat16
jsonl_path = Path("/dataset/workspace/zhangl98/dataset/talker/processed/train.jsonl")
MIMI_REPO_ID = "/dataset/workspace/zhangl98/models/mimi"
NUM_CODE_GROUPS = 16
code2wav_model_path = "/dataset/model_engine/Qwen3-codec-decode/ckpt/0905_ckpt/"
#################### configurations ####################

SPEAKER = "f245"

MODEL_ID = "/dataset/model_engine/qwen3omni_hf_v3_01000-liuding"


flag += str(tuple(enable_modality)).replace("'", "").replace(",", "|")

SAVE_DIR = (
    "/tmp/" + MODEL_ID.rstrip("/").split("/")[-1] + f"-{pretrain}-{flag}" + "-trans"
)

MAX_SEQUENCE_LENGTH = 2048


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
        random.shuffle(self.samples)
        self.samples = self.samples[:NUM_CALIBRATION_SAMPLES]

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
            sequential_onload=True,
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


@torch.no_grad()
def post_compression_talker(model, processor):
    # recipe_[0].on_end(state=state, event=None)
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
    # Confirm generations of the quantized model look sane.
    print("\n\n")

    model.save_pretrained(SAVE_DIR)  # , save_compressed=True) # fakequant
    processor.save_pretrained(SAVE_DIR)
    torch.save(transform_state_dict, f"{SAVE_DIR}/transform_state_dict.pt")
    print(SAVE_DIR)


def dist_load_model(model_path=MODEL_ID, load_processor=False):
    processor = None
    if load_processor:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, config=model_config, torch_dtype=model_dtype
    )
    return model, processor


def regist_hook(model, stat_tensors, weight_tied_name_map):
    def stat_transform_input_hook(m, x, y):
        if isinstance(x, tuple):  # TransformLocation.INPUT
            x = x[0]
        ori_shape = x.shape
        x = x.view(-1, ori_shape[-1])
        # Deduplicate by original tensor data pointer and store CPU copy
        src_id = (
            float(x.max().item()),
            float(x.min().item()),
            float(x.mean().item()),
        )  # x.data_ptr()
        d = stat_tensors[id(m.weight)]
        if src_id not in d:
            d[src_id] = x.detach().cpu()

    def stat_linear_input_hook(m, x, y, subm_name):
        if isinstance(x, tuple):
            x = x[0]
        ori_shape = x.shape
        x = x.view(-1, ori_shape[-1])

        # Deduplicate by original tensor data pointer and store CPU copy
        src_id = (
            float(x.max().item()),
            float(x.min().item()),
            float(x.mean().item()),
        )  # x.data_ptr()
        d = stat_tensors[subm_name]
        if src_id not in d:
            d[src_id] = x.detach().cpu()

    hooks = []
    for name, m in model.named_modules():
        if is_parametrized(m):
            for _name, subm in m.named_children():
                if _name.endswith("_weight_input"):  # TransformLocation.WEIGHT_OUTPUT
                    break
            else:
                continue
            key = weight_tied_name_map[(f"{name}.{_name}", "weight")][0]
            hooks.append(
                m.register_forward_hook(
                    functools.partial(
                        stat_linear_input_hook,
                        subm_name=key,
                    )
                )
            )
        # elif (
        #     isinstance(m, TransformBase) and m.args.location == TransformLocation.INPUT
        # ):
        #     hooks.append(m.register_forward_hook(stat_transform_input_hook))
    return hooks


def model_forward(model, inputs, norm_weight=None):
    device = torch.cuda.current_device()
    batch = inputs["batch"]
    target_codes = inputs["target_codes"]
    speakers = inputs["speakers"]
    audio_lengths = inputs["audio_lengths"]
    """Compute loss for a single sample in the batch"""
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    target_codes = target_codes.to(device)

    # Forward Thinker (frozen)
    with torch.no_grad():
        thinker_outputs = model.thinker(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    # Process each sample in batch
    batch_size_actual = batch["input_ids"].shape[0]
    batch_talker_loss = 0.0
    batch_mtp_loss = 0.0
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

        layer0_embeds = model.talker.get_input_embeddings()(layer0_codes.to(device))
        predictor_embeds = model.talker.code_predictor.get_input_embeddings()
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
                all_layer_embeds_sum[:, prev_pos : prev_pos + 1, :] + text_hidden
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
            all_layer_embeds_sum[:, last_pos : last_pos + 1, :] + eos_text_hidden
        )
        codec_input_embeds_list.append(eos_input_embed)

        # Concatenate
        if codec_input_embeds_list:
            codec_input_embeds = torch.cat(codec_input_embeds_list, dim=1).to(
                model_dtype
            )
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
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

        # Prefill
        if full_inputs_embeds is not None and full_inputs_embeds.shape[1] > 1:
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

        # talker_loss = F.cross_entropy(
        #     shift_logits.view(-1, shift_logits.size(-1)),
        #     shift_labels.view(-1),
        #     ignore_index=-100,
        # )

        # MTP Training (Layers 1-15)
        talker_hidden = (
            hidden_states[0][-1]
            if isinstance(hidden_states, tuple)
            else hidden_states[-1]
        )
        if norm_weight is not None:
            dtype = norm_weight.dtype
            R1 = model.talker.model.codec_embedding.R1_weight_output.weight.float()
            talker_hidden = talker_hidden @ (
                (R1.T * norm_weight.float()) @ R1
            ).to(dtype)

        codec_hidden_start = prefix_len - 1
        codec_hidden_end = prefix_len - 1 + num_codec_tokens
        codec_hidden = talker_hidden[:, codec_hidden_start:codec_hidden_end, :]

        mtp_total_loss = 0.0
        code_predictor = model.talker.code_predictor
        hidden_dim = codec_hidden.shape[2]
        num_mtp_layers = NUM_CODE_GROUPS - 1

        layer0_embed_for_mtp = model.talker.get_input_embeddings()(
            layer0_codes.to(device)
        )
        hidden_flat = codec_hidden.reshape(-1, 1, hidden_dim)
        layer0_flat = layer0_embed_for_mtp.reshape(-1, 1, hidden_dim)

        for mtp_layer_idx in range(num_mtp_layers):
            embed_list = [hidden_flat, layer0_flat]

            for prev_layer in range(mtp_layer_idx):
                prev_codes = sample_codes[:, prev_layer + 1, :].to(device)
                prev_embed = predictor_embeds[prev_layer](prev_codes)
                prev_embed_flat = prev_embed.reshape(-1, 1, hidden_dim)
                embed_list.append(prev_embed_flat)

            mtp_inputs = torch.cat(embed_list, dim=1).to(model_dtype)
            target_layer_codes = sample_codes[:, mtp_layer_idx + 1, :].to(device)
            target_labels = target_layer_codes.reshape(-1)

            mtp_outputs = code_predictor(
                inputs_embeds=mtp_inputs,
                generation_steps=mtp_layer_idx,
                use_cache=False,
            )

            mtp_logits = mtp_outputs.logits[:, -1, :]
            # mtp_layer_loss = F.cross_entropy(mtp_logits, target_labels)
            # mtp_total_loss += mtp_layer_loss

        # mtp_avg_loss = mtp_total_loss / num_mtp_layers
        # batch_talker_loss += talker_loss
        # batch_mtp_loss += mtp_avg_loss
    # avg_talker_loss = batch_talker_loss / batch_size_actual
    # avg_mtp_loss = batch_mtp_loss / batch_size_actual
    # return (avg_talker_loss ) #+ 2.0 * avg_mtp_loss)
    return None


def inference_model(model, norm_weight=None):
    feature_extractor = AutoFeatureExtractor.from_pretrained(MIMI_REPO_ID)
    mimi_model = MimiModel.from_pretrained(MIMI_REPO_ID, torch_dtype=model_dtype).to(
        torch.cuda.current_device()
    )
    config_path = code2wav_model_path + "/config.yaml"
    checkpoint_path = code2wav_model_path + "/model_weights.pt"
    code2wav = load_model_from_config(config_path, checkpoint_path, device="cuda")
    dispatch_for_generation(model, extra_memory=10 * 256 * 2048 * 2)
    train_processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        model_max_length=MAX_SEQUENCE_LENGTH,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )

    def collate_wrapper(batch):
        return collate_fn(
            batch,
            train_processor,
            feature_extractor,
            mimi_model,
            torch.cuda.current_device(),
            code2wav,
        )

    dataloader = DataLoader(
        ds,
        batch_size=1,
        collate_fn=collate_wrapper,
        pin_memory=True,
        persistent_workers=False,
    )

    for batch in tqdm(dataloader):
        with torch.no_grad():
            # batch = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in batch.items()}
            _ = model_forward(model, batch, norm_weight)
            # stat_tensors holds CPU tensors already; nothing to cast per-batch.
    remove_dispatch(model)


def train_rotate(
    name,
    transform_module,
    train_datas,
    optim="sgd",
    lr=1.5e-3,
    mom=0.9,
    cos_lr=False,
    ep=10,
    bsz=64,
    accumulation_steps=2,
    val_ratio: float = 0.1,
):
    class R_QR(torch.nn.Module):
        def __init__(self, size: int, location, module_type):
            super(R_QR, self).__init__()
            self.size = size
            self.matrix = torch.nn.Parameter(torch.eye(size))
            self.location = location
            self.module_type = module_type

        def forward(self, x):
            self.rotate, _ = torch.linalg.qr(self.matrix, mode="complete")
            # o_x = torch.matmul(x, self.rotate)
            o_x = apply_transform_weight(
                self.rotate.to(device=x.device, dtype=torch.float64),
                x.to(dtype=torch.float64),
                self.location,
                self.module_type,
            )
            return o_x

    device = "cuda" if torch.cuda.is_available() else "cpu"
    construct_device = device
    precision = transform_module.weight.dtype
    size = transform_module.weight.shape[0]
    location = TransformLocation.INPUT
    module_type = torch.nn.Linear
    data = random_hadamard_matrix(size, precision, construct_device)
    data = data.to(device=device)
    _scale = torch.tensor(size, dtype=precision, device=device).sqrt()
    data = data / _scale

    R = R_QR(size=size, location=location, module_type=module_type).to(device)
    R.matrix.data = data

    if optim == "sgd":
        optimizer = torch.optim.SGD(R.parameters(), lr=lr, momentum=mom)
    elif optim == "adam":
        optimizer = torch.optim.Adam(
            R.parameters(),
            lr=lr,
        )
    else:
        raise NotImplementedError

    if cos_lr:  # 设置余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ep, eta_min=0
        )

    R.train()

    print(f"---> start training R of layer {name} ")

    def pad_collate(batch):
        # batch: list of tensors [L_i, hidden]
        lengths = [t.shape[0] for t in batch]
        max_len = max(lengths)
        hidden = batch[0].shape[-1]
        out = torch.zeros((len(batch), max_len, hidden), dtype=batch[0].dtype)
        mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, t in enumerate(batch):
            out[i, : t.shape[0], :] = t
            mask[i, : t.shape[0]] = 1
        return out, mask

    train_dataloader = DataLoader(
        train_datas,
        batch_size=bsz,
        shuffle=True,
        pin_memory=True,
        collate_fn=pad_collate,
    )
    # Prepare validation dataloader if a validation split is requested
    val_dataloader = None
    if val_ratio and 0.0 < val_ratio < 1.0:
        n = len(train_datas)
        if n > 0:
            val_size = max(1, int(n * val_ratio))
            indices = list(range(n))
            random.shuffle(indices)
            val_indices = set(indices[:val_size])
            train_indices = indices[val_size:]
            train_list = [train_datas[i] for i in train_indices]
            val_list = [train_datas[i] for i in sorted(val_indices)]
            train_dataloader = DataLoader(
                train_list,
                batch_size=bsz,
                shuffle=True,
                pin_memory=True,
                collate_fn=pad_collate,
            )
            val_dataloader = DataLoader(
                val_list,
                batch_size=bsz,
                shuffle=False,
                pin_memory=True,
                collate_fn=pad_collate,
            )
    for epoch in range(ep):
        loss_log = []

        for batch_idx, batch in enumerate(train_dataloader):
            batch_samples, mask = batch
            batch_samples = batch_samples.to(device)
            mask = mask.to(device)
            outputs = R(batch_samples)
            # outputs shape: (B, L, hidden) -> per-token score: (B, L)
            per_token = torch.sum(torch.exp((-outputs.abs())), dim=-1)
            # mask: bool -> float
            mask_f = mask.to(dtype=per_token.dtype)
            # avoid division by zero
            denom = mask_f.sum()
            if denom.item() == 0:
                denom = torch.tensor(
                    1.0, device=per_token.device, dtype=per_token.dtype
                )
            loss = (per_token * mask_f).sum() / denom / accumulation_steps
            loss.backward()

            # 如果达到指定的累计步数，更新梯度并清零
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_log.append(loss.detach())

        if cos_lr:
            scheduler.step()

        # 计算平均损失 并打印日志
        if len(loss_log) == 0:
            mean_loss = torch.tensor(0.0, device=device)
        else:
            mean_loss = torch.stack(loss_log).mean()
        # If we have a validation set evaluate it once per epoch
        val_mean = None
        if val_dataloader is not None:
            R.eval()
            val_loss_log = []
            with torch.no_grad():
                for batch in val_dataloader:
                    batch_samples, mask = batch
                    batch_samples = batch_samples.to(device)
                    mask = mask.to(device)
                    outputs = R(batch_samples)
                    per_token = torch.sum(torch.exp((-outputs.abs())), dim=-1)
                    mask_f = mask.to(dtype=per_token.dtype)
                    denom = mask_f.sum()
                    if denom.item() == 0:
                        denom = torch.tensor(
                            1.0, device=per_token.device, dtype=per_token.dtype
                        )
                    loss = (per_token * mask_f).sum() / denom
                    val_loss_log.append(loss.detach())
            if len(val_loss_log) > 0:
                val_mean = torch.stack(val_loss_log).mean()
            R.train()

        log_message = f"Epoch [{epoch + 1}/{ep}], Train Loss: {mean_loss.item():.4f}"
        if val_mean is not None:
            log_message += f", Val Loss: {val_mean.item():.4f}"
        log_message += ", "
        if cos_lr:
            log_message += f", LR: {scheduler.get_last_lr()[0]:.4e}"
        print(log_message)

    print(f"---> R of layer {name} training done ")

    return R.rotate.data.detach().clone()


if __name__ == "__main__":
    model_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model, processor = dist_load_model(load_processor=True)
    # dist.barrier()
    with patch_module_non_persistent_buffers(model):
        model.eval()
        # no_grad for compression
        for param in model.parameters():
            param.requires_grad = False
        norm_weight = model.talker.model.norm.weight.data.clone()
        state, recipe_, model = pre_trans_talker(model)

        weight_tied_name_map = build_weight_tied_map_with_unionfind(
            model, remove_duplicate=False
        )
        # Map from weight_id -> {src_data_ptr: cpu_tensor}
        stat_tensors = defaultdict(dict)
        hooks = regist_hook(model, stat_tensors, weight_tied_name_map)
        inference_model(model, norm_weight.cuda())
        model.cpu()
        # offload_model(model, "cuda:0", "cpu")
        MyTrainer.register_tied_parameters(model, weight_tied_name_map)
        for param in model.parameters():
            param.requires_grad = False
        torch.cuda.empty_cache()
        for h in hooks:
            h.remove()
        for module_name, param_name in set(weight_tied_name_map.values()):
            module = model.get_submodule(module_name)
            param = model.get_parameter(f"{module_name}.{param_name}")
            R = train_rotate(
                name=f"{module_name}.{param_name}",
                transform_module=module,
                train_datas=list(stat_tensors[module_name].values()),
                optim="adam",
                lr=1.5e-3,
                mom=0.9,
                cos_lr=False,
                ep=10,
                bsz=32,
                accumulation_steps=2,
                val_ratio=0.1,
            )
            update_offload_parameter(
                module,
                "weight",
                R.to(device=module.weight.device, dtype=module.weight.dtype),
            )
            # module.weight.data.copy_(
            #     R.to(device=module.weight.device, dtype=module.weight.dtype)
            # )

        if "talker" in enable_modality:
            recipe_[0]._fold_transforms_into_weights(state.model)

    post_compression_talker(model, processor)
