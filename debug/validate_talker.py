import contextlib
import json
import os
import random
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch
from datasets import load_dataset
from qwen_omni_utils import process_mm_info
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoProcessor,
    HfArgumentParser,
    MimiModel,
)

sys.path.insert(0, os.path.abspath("./examples/omnitalker/"))
from qwen3_omni_moe_utils.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)
from qwen3_omni_moe_utils.qwen3_omni_codec_decode import load_model_from_config

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.datasets import get_calibration_dataloader
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.utils import dispatch_for_generation, helpers

# Select model and load it.
REF_MODEL_ID = "/dataset/model_engine/qwen3omni_hf_v3_01000-liuding"
MODEL_ID = "/tmp/qwen3omni_hf_v3_01000-liuding-origin-dartquant(talker|)-trans"
jsonl_path = Path("/dataset/workspace/zhangl98/dataset/talker/processed/train.jsonl")
MIMI_REPO_ID = "/dataset/workspace/zhangl98/models/mimi"
NUM_CODE_GROUPS = 16
code2wav_model_path = "/dataset/model_engine/Qwen3-codec-decode/ckpt/0905_ckpt/"
SPEAKER = "f245"
USE_AUDIO_IN_VIDEO = True
model_config = AutoConfig.from_pretrained(REF_MODEL_ID, trust_remote_code=True)
ref_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    REF_MODEL_ID, config=model_config, torch_dtype="auto"
)

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_ID, config=model_config, torch_dtype="auto"
)
model_dtype = model.dtype
model_device = torch.device("cuda")
processor = AutoProcessor.from_pretrained(REF_MODEL_ID, trust_remote_code=True)


# Select calibration dataset.
DATASET_ID = "/dataset/workspace/zhangl98/dataset/peoples_speech/test"
DATASET_SPLIT = "test[:512]"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 1
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)


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


# helpers.patch_attr(model.thinker.audio_tower, "forward", audio_forward))


def model_forward(model, inputs):
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

            # mtp_logits = mtp_outputs.logits[:, -1, :]
            # mtp_layer_loss = F.cross_entropy(mtp_logits, target_labels)
            # mtp_total_loss += mtp_layer_loss

        # mtp_avg_loss = mtp_total_loss / num_mtp_layers
        # batch_talker_loss += talker_loss
        # batch_mtp_loss += mtp_avg_loss
    # avg_talker_loss = batch_talker_loss / batch_size_actual
    # avg_mtp_loss = batch_mtp_loss / batch_size_actual
    # return (avg_talker_loss ) #+ 2.0 * avg_mtp_loss)
    return shift_logits, mtp_outputs.logits


feature_extractor = AutoFeatureExtractor.from_pretrained(MIMI_REPO_ID)
mimi_model = MimiModel.from_pretrained(MIMI_REPO_ID, torch_dtype=model_dtype).to(
    torch.cuda.current_device()
)
config_path = code2wav_model_path + "/config.yaml"
checkpoint_path = code2wav_model_path + "/model_weights.pt"
code2wav = load_model_from_config(config_path, checkpoint_path, device="cuda")
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

# for batch in tqdm(dataloader):
#     with torch.no_grad():
#         # batch = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in batch.items()}
#         _ = model_forward(model, batch)
#         # stat_tensors holds CPU tensors already; nothing to cast per-batch.
# # remove_dispatch(model)


with contextlib.ExitStack() as stack:
    stack.enter_context(torch.no_grad())
    # stack.enter_context(
    #     helpers.patch_attr(
    #         model.thinker.audio_tower,
    #         "forward",
    #         audio_wrap_funcs["forward"].__get__(model.thinker.audio_tower),
    #     )
    # )
    _rets = []
    _ref_rets = []
    for inputs in tqdm(dataloader):
        dispatch_for_generation(model, extra_memory=10 * 256 * 2048 * 2)
        ret = flatten_obj(model_forward(model, inputs))
        _rets.append([x.cpu() for x in ret])
    rets = []
    for ret in zip(*_rets):
        rets.append(torch.cat(ret, dim=0))

    # model.thinker.audio_tower.cpu()
    del model

    for inputs in tqdm(dataloader):
        dispatch_for_generation(ref_model, extra_memory=10 * 256 * 2048 * 2)
        ref_ret = flatten_obj(model_forward(ref_model, inputs))
        _ref_rets.append([x.cpu() for x in ref_ret])

    ref_rets = []
    for ref_ret in zip(*_ref_rets):
        ref_rets.append(torch.cat(ref_ret, dim=0))

    # ref_model.thinker.audio_tower.cpu()
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
