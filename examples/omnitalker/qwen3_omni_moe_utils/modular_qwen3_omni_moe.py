# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2.5Omni model (Audio, Image, Video)."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeModel,
    Qwen3MoeForCausalLM,
    Qwen3MoePreTrainedModel,
    eager_attention_forward,
    load_balancing_loss_func,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeDecoderLayer,
    Qwen3MoeMLP,
    Qwen3MoeSparseMoeBlock,
)
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3MLP, Qwen3ForCausalLM
from transformers.models.qwen3_omni.configuration_qwen3_omni import (
    Qwen3OmniConfig,
    Qwen3OmniTalkerConfig,
    Qwen3OmniThinkerConfig,
    Qwen3OmniToken2WavConfig,
)
from transformers.models.qwen3_omni.modeling_qwen3_omni import (
    Qwen3OmniForConditionalGeneration,
    Qwen3OmniPreTrainedModel,
    Qwen3OmniPreTrainedModelForConditionalGeneration,
    Qwen3OmniRotaryEmbedding,
    Qwen3OmniTalkerForConditionalGeneration,
    Qwen3OmniThinkerForConditionalGeneration,
    apply_multimodal_rotary_pos_emb,
    Qwen2_5_VLMLP,
    _get_feat_extract_output_lengths,
)
from transformers.models.qwen3_omni.processing_qwen3_omni import Qwen3OmniProcessor

from ...cache_utils import Cache, DynamicCache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import ModelOutput, BaseModelOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...configuration_utils import PretrainedConfig
from ...activations import ACT2FN
from ...utils import logging
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask


logger = logging.get_logger(__name__)


class Qwen3OmniMoeTextConfig(Qwen3MoeConfig):
    def __init__(
        self,
        vocab_size=3584,
        hidden_size=2048,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=32768,
        max_window_layers=28,
        attention_dropout=0,
        decoder_sparse_step=1,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            use_sliding_window,
            sliding_window,
            max_window_layers,
            attention_dropout,
            decoder_sparse_step,
            moe_intermediate_size,
            num_experts_per_tok,
            num_experts,
            norm_topk_prob,
            output_router_logits,
            router_aux_loss_coef,
            mlp_only_layers,
            **kwargs,
        )


class Qwen3OmniMoeThinkerTextConfig(Qwen3OmniMoeTextConfig):
    pass


class Qwen3OmniMoeThinkerConfig(Qwen3OmniThinkerConfig):
    pass

class Qwen3OmniMoeTalkerCodePredictorConfig(Qwen3Config):
    def __init__(
        self,
        vocab_size=2048,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=5,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0,
        num_code_groups=32,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            use_sliding_window,
            sliding_window,
            max_window_layers,
            layer_types,
            attention_dropout,
            **kwargs,
        )
        self.num_code_groups = num_code_groups

class Qwen3OmniMoeTalkerConfig(Qwen3MoeConfig):
    sub_configs = {"code_predictor_config": Qwen3OmniMoeTalkerCodePredictorConfig}

    def __init__(
        self,
        code_predictor_config=None,
        vocab_size=3072,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        attention_dropout=0,
        decoder_sparse_step=1,
        moe_intermediate_size=384,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        num_code_groups=32,
        thinker_hidden_size=2048,
        codec_eos_token_id=4198,
        accept_hidden_layer=18,
        codec_nothink_id=4203,
        codec_think_bos_id=4204,
        codec_think_eos_id=4205,
        codec_pad_id=4196,
        codec_bos_id=4197,
        debug_spk_id=4348,
        audio_token_index=151646,
        image_token_index=151655,
        video_token_index=151656,
        position_id_per_seconds=25,
        seconds_per_chunk=2,
        audio_start_token_id=151647,
        audio_end_token_id=151648,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            use_sliding_window,
            sliding_window,
            attention_dropout,
            decoder_sparse_step,
            moe_intermediate_size,
            num_experts_per_tok,
            num_experts,
            norm_topk_prob,
            output_router_logits,
            router_aux_loss_coef,
            mlp_only_layers,
            **kwargs,
        )
        if code_predictor_config is None:
            code_predictor_config = {}
            self.code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig()
            logger.info("thinker_config is None. Initializing thinker model with default values")
        elif isinstance(code_predictor_config, Qwen3OmniMoeTalkerCodePredictorConfig):
            self.code_predictor_config = code_predictor_config
        else:
            self.code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig(**code_predictor_config)
        self.num_code_groups = num_code_groups
        self.thinker_hidden_size = thinker_hidden_size
        self.codec_eos_token_id = codec_eos_token_id
        self.accept_hidden_layer = accept_hidden_layer
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
        self.debug_spk_id = debug_spk_id
        self.audio_token_index = audio_token_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.position_id_per_seconds = position_id_per_seconds
        self.seconds_per_chunk = seconds_per_chunk
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id

class Qwen3OmniMoeConfig(Qwen3OmniConfig, PretrainedConfig):
    model_type = "qwen3_omni"
    sub_configs = {
        "thinker_config": Qwen3OmniMoeThinkerConfig,
        "talker_config": Qwen3OmniMoeTalkerConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        talker_config=None,
        enable_audio_output=True,
        im_start_token_id=151644,
        im_end_token_id=151645,
        tts_pad_token_id=151671,
        tts_bos_token_id=151672,
        tts_eos_token_id=151673,
        system_token_id=8948,
        user_token_id=872,
        assistant_token_id=77091,
        **kwargs,
    ):
        PretrainedConfig.__init__(**kwargs)
        if thinker_config is None:
            thinker_config = {}
            logger.info("thinker_config is None. Initializing thinker model with default values")

        if talker_config is None:
            talker_config = {}
            logger.info("talker_config is None. Initializing talker model with default values")

        self.thinker_config = Qwen3OmniMoeThinkerConfig(**thinker_config)
        self.talker_config = Qwen3OmniMoeTalkerConfig(**talker_config)
        self.enable_audio_output = enable_audio_output
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id
        self.system_token_id = system_token_id
        self.user_token_id = user_token_id
        self.assistant_token_id = assistant_token_id

    def get_text_config(self, decoder=False) -> "PretrainedConfig":
        """
        Returns the config that is meant to be used with text IO. On most models, it is the original config instance
        itself. On specific composite models, it is under a set of valid names.

        Args:
            decoder (`Optional[bool]`, *optional*, defaults to `False`):
                If set to `True`, then only search for decoder config names.
        """
        # Overridden for deeply nested config like Qwen2-Omni. We don't have any omni model
        # except for Qwen yet. This has to be generalized if more deeply nested configs are
        # added. NOTE: currently method used only by vLLM
        return self.thinker_config.get_text_config()


class Qwen3OmniMoePreTrainedModel(Qwen3OmniPreTrainedModel):
    pass


class Qwen3OmniMoePreTrainedModelForConditionalGeneration(Qwen3OmniPreTrainedModelForConditionalGeneration):
    pass


class Qwen3OmniMoeThinkerTextRotaryEmbedding(Qwen3OmniRotaryEmbedding):
    pass


class Qwen3OmniMoeThinkerTextAttention(Qwen3MoeAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.rope_scaling = config.rope_scaling

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"], self.rope_scaling["interleaved"]
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3OmniMoeThinkerTextPreTrainedModel(Qwen3MoePreTrainedModel):
    config_class = Qwen3OmniMoeTextConfig

class Qwen3OmniMoeThinkerTextDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3OmniMoeThinkerTextAttention(config, layer_idx)

class Qwen3OmniMoeThinkerTextModel(Qwen3MoeModel):
    config_class = Qwen3OmniMoeTextConfig

    def __init__(self, config: Qwen3OmniMoeTextConfig):
        super().__init__(config)
        self.deepstack_multiscale_layer_start = 1

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds_this):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds_this = visual_embeds_this.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks].clone() + visual_embeds_this.view(-1)
        hidden_states[visual_pos_masks] = local_this
        return hidden_states

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_router_logits=None,
        cache_position=None,
        deepstack_visual_embeds_multiscale=None,
        visual_pos_masks=None,
        **flash_attn_kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # FIXME:lc: check the process with output_hidden_states=True
            if deepstack_visual_embeds_multiscale is not None and layer_idx in range(
                self.deepstack_multiscale_layer_start,
                self.deepstack_multiscale_layer_start + len(deepstack_visual_embeds_multiscale),
            ):
                deepstack_visual_embeds_multiscale_this = deepstack_visual_embeds_multiscale[
                    layer_idx - self.deepstack_multiscale_layer_start
                ]
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds_multiscale_this
                )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

@dataclass
class Qwen3OmniMoeThinkerCausalLMOutputWithPast(ModelOutput):
    r"""
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
                aux_loss for the sparse modules.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

                Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
                loss for Mixture of Experts models.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

class Qwen3OmniMoeThinkerForConditionalGeneration(Qwen3OmniThinkerForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.text_config.num_experts
        self.num_experts_per_tok = config.text_config.num_experts_per_tok

    def forward(
        self,
        input_ids=None,
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_router_logits: Optional[bool] = None,
        return_dict=None,
        use_audio_in_video=None,
        cache_position=None,
        video_second_per_grid=None,
    ) -> Union[tuple, Qwen3OmniMoeThinkerCausalLMOutputWithPast]:
        r"""
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, feature_sequence_length)`):
                Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
                loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a `numpy.ndarray`, *e.g.* via
                the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
                [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
                tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size), *optional*):
                The tensors corresponding to the input videos. Pixel values can be obtained using
                [`AutoImageProcessor`]. See [`SiglipImageProcessor.__call__`] for details ([]`NewTaskModelProcessor`] uses
                [`SiglipImageProcessor`] for processing videos).
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
                The rope index difference between sequence length and multimodal rope.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_audio_in_video (`bool`, *optional*):
                Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
            video_second_per_grid (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                Number of seconds per grid for each video, used for temporal feature mapping.

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from qwen_vl_utils import process_vision_info
        >>> from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

        >>> thinker = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        >>> processor = Qwen3OmniMoeProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        >>> conversations = [
        >>>         {'role': 'system', 'content': 'You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.'},
        >>>         {"role": "user", "content": [
        >>>             {"type": "image", "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
        >>>             {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
        >>>         ]},
        >>> ]

        >>> text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        >>> audios = [ librosa.load(BytesIO(urlopen( conversations[1]['content'][1]['audio_url'] ).read()), sr=self.processor.feature_extractor.sampling_rate) ]
        >>> images, videos = process_vision_info(conversations)
        >>> inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)

        >>> # Generate
        >>> inputs['use_audio_in_video'] = `True` or `False`
        >>> generation = thinker.generate(**inputs, max_new_tokens=2048)
        >>> generate_ids = generation[:, inputs.input_ids.size(1):]

        >>> response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        visual_embeds_multiscale = None
        visual_pos_masks = None
        # 2. Merge text , audios , image and video
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            if input_ids is None:
                audio_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                audio_mask = audio_mask.all(-1)
            else:
                audio_mask = input_ids == self.config.audio_token_id

            audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if pixel_values is not None:
            image_embeds, image_embeds_multiscale = self.get_image_features(pixel_values, image_grid_thw)
            if input_ids is None:
                image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                image_mask = image_mask.all(-1)
            else:
                image_mask = input_ids == self.config.image_token_id

            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            visual_pos_masks = image_mask
            visual_embeds_multiscale = image_embeds_multiscale

        if pixel_values_videos is not None:
            video_embeds, video_embeds_multiscale = self.get_video_features(pixel_values_videos, video_grid_thw)
            if input_ids is None:
                video_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                video_mask = video_mask.all(-1)
            else:
                video_mask = input_ids == self.config.video_token_id

            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if visual_embeds_multiscale is None:
                visual_embeds_multiscale = video_embeds_multiscale
                visual_pos_masks = video_mask
            else:
                visual_pos_masks = video_mask | image_mask
                visual_embeds_multiscale_joint = ()
                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]
                for img_embed, vid_embed in zip(visual_embeds_multiscale, video_embeds_multiscale):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1])
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    visual_embeds_multiscale_joint = visual_embeds_multiscale_joint + (embed_joint,)
                visual_embeds_multiscale = visual_embeds_multiscale_joint

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            deepstack_visual_embeds_multiscale=visual_embeds_multiscale,
            visual_pos_masks=visual_pos_masks,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return Qwen3OmniMoeThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            rope_deltas=self.rope_deltas,
        )

class Qwen3OmniMoeTalkerResizeMLP(Qwen2_5_VLMLP, nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int, act: str, bias=False):
        nn.Module().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = ACT2FN[act]



@dataclass
class Qwen3OmniMoeTalkerCodePredictorOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    generation_steps: Optional[int] = None


class Qwen3OmniMoeTalkerCodePredictorModel(Qwen3Model):
    config_class = Qwen3OmniMoeTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor.model"

    def __init__(self, config: Qwen3OmniMoeTalkerCodePredictorConfig):
        super().__init__(config)
        del self.embed_tokens
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, config.hidden_size) for _ in range(config.num_code_groups - 1)]
        )

    def get_input_embeddings(self):
        return self.codec_embedding

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        generation_steps=None,
        **flash_attn_kwargs,
    ):
        if input_ids is not None:
            raise ValueError("`input_ids` is expected to be `None`")
        return super().forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            cache_position,
            **flash_attn_kwargs,
        )


class Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(Qwen3ForCausalLM):
    config_class = Qwen3OmniMoeTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor"

    def __init__(self, config: Qwen3OmniMoeTalkerCodePredictorConfig):
        super().__init__(config)
        self.model = Qwen3OmniMoeTalkerCodePredictorModel(config)
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        generation_steps=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Prefill stage
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = inputs_embeds.shape[1] - 2  # hidden & layer 0
        # Generation stage
        else:
            inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head[generation_steps](hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return Qwen3OmniMoeTalkerCodePredictorOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            generation_steps=generation_steps + 1,
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["generation_steps"] = outputs.generation_steps
        return model_kwargs


@dataclass
class Qwen3OmniMoeTalkerOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None
    past_hidden: Optional[torch.FloatTensor] = None
    generation_step: Optional[int] = None
    trailing_text_hidden: Optional[torch.FloatTensor] = None
    tts_pad_embed: Optional[torch.FloatTensor] = None


class Qwen3OmniMoeTalkerRotaryEmbedding(Qwen3OmniMoeThinkerTextRotaryEmbedding):
    pass


class Qwen3OmniMoeTalkerAttention(Qwen3OmniMoeThinkerTextAttention):
    pass

class Qwen3OmniMoeTalkerTextMLP(Qwen3MoeMLP):
    pass

class Qwen3OmniMoeTalkerTextSparseMoeBlock(Qwen2MoeSparseMoeBlock):
    pass

class Qwen3OmniMoeTalkerDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3OmniMoeTalkerAttention(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3OmniMoeTalkerTextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3OmniMoeTalkerTextMLP(config, intermediate_size=config.intermediate_size)


class Qwen3OmniMoeTalkerModel(Qwen3MoeModel):
    config_class = Qwen3OmniMoeTalkerConfig
    base_model_prefix = "talker.model"

    def __init__(self, config):
        super().__init__(config)
        del self.embed_tokens
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3OmniMoeTalkerDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Qwen3OmniMoeTalkerRotaryEmbedding(config)

    def get_input_embeddings(self):
        return self.codec_embedding


class Qwen3OmniMoeTalkerForConditionalGeneration(Qwen3MoeForCausalLM):
    config_class = Qwen3OmniMoeTalkerConfig
    base_model_prefix = "talker"

    def __init__(self, config: Qwen3OmniMoeTalkerConfig):
        super().__init__(config)
        del self.lm_head
        self.model = Qwen3OmniMoeTalkerModel(config)
        self.text_projection = Qwen3OmniMoeTalkerResizeMLP(
            config.thinker_hidden_size, config.intermediate_size, config.hidden_size, config.hidden_act, bias=True
        )
        self.hidden_projection = Qwen3OmniMoeTalkerResizeMLP(
            config.thinker_hidden_size, config.intermediate_size, config.hidden_size, config.hidden_act, bias=True
        )
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.code_predictor = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(
            config=config.code_predictor_config
        )
        self.rope_deltas = None
        self.spatial_merge_size = self.config.spatial_merge_size

        # TODO: hack, modular cannot inherit multiple classes

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            use_audio_in_video (`bool`, *optional*):
                 If set to `True`, use the audio in video.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.zeros(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=torch.float,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = torch.sum(input_ids == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (
                    (vision_tokens == audio_start_token_id).sum()
                    if use_audio_in_video
                    else (vision_tokens == video_token_id).sum()
                )
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
                multimodal_nums = (
                    image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
                )
                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                        remain_videos > 0 or remain_images > 0
                    ):
                        ed_vision_start = input_tokens.index(vision_start_token_id, st)
                    else:
                        ed_vision_start = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio_start = input_tokens.index(audio_start_token_id, st)
                    else:
                        ed_audio_start = len(input_tokens) + 1
                    min_ed = min(ed_vision_start, ed_audio_start)
                    # Audio Only
                    if min_ed == ed_audio_start:
                        text_len = min_ed - st
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                        llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + audio_len + eos_len
                        audio_idx += 1
                        remain_audios -= 1

                    # Image Only
                    elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == image_token_id:
                        text_len = min_ed - st
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).float()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + image_len + eos_len
                        image_idx += 1
                        remain_images -= 1

                    # Video Only
                    elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == video_token_id:
                        text_len = min_ed - st
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        shift = (
                            (self.config.vision_config.temporal_patch_size - 1)
                            / self.config.vision_config.temporal_patch_size
                            / 2
                        )
                        t_index = (
                            (torch.arange(grid_t) + shift)
                            * second_per_grids[video_idx].cpu().float()
                            * position_id_per_seconds
                        ).float()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + video_len + eos_len
                        video_idx += 1
                        remain_videos -= 1

                    # Audio in Video
                    elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                        text_len = min_ed - st
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        shift = (
                            (self.config.vision_config.temporal_patch_size - 1)
                            / self.config.vision_config.temporal_patch_size
                            / 2
                        )
                        t_index = (
                            (torch.arange(grid_t) + shift)
                            * second_per_grids[video_idx].cpu().float()
                            * position_id_per_seconds
                        ).float()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_data_index, audio_data_index = 0, 0
                        while (
                            video_data_index < video_llm_pos_ids.shape[-1]
                            and audio_data_index < audio_llm_pos_ids.shape[-1]
                        ):
                            if video_llm_pos_ids[0][video_data_index] <= audio_llm_pos_ids[0][audio_data_index]:
                                llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_data_index + 1])
                                video_data_index += 1
                            else:
                                llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1])
                                audio_data_index += 1
                        if video_data_index < video_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                video_llm_pos_ids[:, video_data_index : video_llm_pos_ids.shape[-1]]
                            )
                        if audio_data_index < audio_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                audio_llm_pos_ids[:, audio_data_index : audio_llm_pos_ids.shape[-1]]
                            )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                        st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max().ceil() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat([item.float() for item in llm_pos_ids_list], dim=1).reshape(3, -1)

                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.float().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

            return position_ids, mrope_position_deltas

    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[int],
        grid_hs: list[int],
        grid_ws: list[int],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
        t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
        _llm_pos_ids = torch.stack([t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)  # + 1 ) # 12.09 by malinhan
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        use_audio_in_video=None,
        audio_feature_lengths=None,
        video_second_per_grid=None,
        image_grid_thw=None,
        video_grid_thw=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_router_logits=None,
        cache_position=None,
        past_hidden=None,
        trailing_text_hidden=None,
        tts_pad_embed=None,
        generation_step=None,
        talker_input_ids=None,
        **kwargs,
    ):
        # Prefill
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_step = -1
            codec_ids = None
        # Generate
        else:
            last_id_hidden = self.get_input_embeddings()(input_ids)
            predictor_result = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                max_new_tokens=self.config.num_code_groups - 1,
                do_sample=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)
            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [hid[0] for hid in predictor_result.hidden_states[1:]]
                + [self.code_predictor.get_input_embeddings()[-1](predictor_result.sequences[..., -1:])],
                dim=1,
            )
            inputs_embeds = codec_hiddens.sum(1, keepdim=True)

            if generation_step < trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed
        if attention_mask is not None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    talker_input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return Qwen3OmniMoeTalkerOutputWithPast(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            past_key_values=outputs.past_key_values,
            hidden_states=(outputs.hidden_states, codec_ids),
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            past_hidden=hidden_states[:, -1:, :],
            generation_step=generation_step + 1,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["past_hidden"] = outputs.past_hidden
        model_kwargs["generation_step"] = outputs.generation_step
        model_kwargs["trailing_text_hidden"] = outputs.trailing_text_hidden
        model_kwargs["tts_pad_embed"] = outputs.tts_pad_embed
        return model_kwargs


class Qwen3OmniMoeForConditionalGeneration(Qwen3OmniMoePreTrainedModel, GenerationMixin):
    config_class = Qwen3OmniMoeConfig

    def __init__(self, config: Qwen3OmniMoeConfig):
        super().__init__(config)

        self.thinker = Qwen3OmniMoeThinkerForConditionalGeneration(config.thinker_config)
        self.has_talker = config.enable_audio_output
        if self.has_talker:
            self.enable_talker()
        self.post_init()

    def enable_talker(self):
        self.talker = Qwen3OmniMoeTalkerForConditionalGeneration(self.config.talker_config)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        speaker: str = "debug_spk",
        use_audio_in_video: bool = False,
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        thinker_eos_token_id: int = 151645,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 40,
        talker_top_p: float = 0.8,
        talker_temperature: float = 0.9,
        talker_eos_token_id: Optional[int] = None,
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initialized. Use `enable_talker` method or set enable_talker in config to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker
        spk_id = getattr(self.config.talker_config, f"{speaker.lower()}_id", None)
        if return_audio and spk_id is None:
            raise NotImplementedError(f"Speaker {speaker} not implemented")
        if input_ids.shape[0] != 1 and return_audio:
            raise NotImplementedError("Qwen3-Omni currently does not support batched inference with audio output")

        shared_kwargs = {}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
            "eos_token_id": thinker_eos_token_id,
            "use_audio_in_video": use_audio_in_video,
        }
        talker_kwargs = {
            "max_new_tokens": talker_max_new_tokens,
            "do_sample": talker_do_sample,
            "top_k": talker_top_k,
            "top_p": talker_top_p,
            "temperature": talker_temperature,
            "eos_token_id": talker_eos_token_id
            if talker_eos_token_id is not None
            else self.config.talker_config.codec_eos_token_id,
            "repetition_penalty": talker_repetition_penalty,
            "suppress_tokens": [
                i
                for i in range(self.config.talker_config.vocab_size - 1024, self.config.talker_config.vocab_size)
                if i not in (self.config.talker_config.codec_eos_token_id,)
            ],
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        }
        token2wav_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            # Process special input values
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key in ("input_features", "attention_mask", "use_audio_in_video"):
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value

        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value

        # 1. Generate from thinker module
        generate_audio = return_audio and self.has_talker
        if generate_audio:
            thinker_kwargs["output_hidden_states"] = True
            thinker_kwargs["return_dict_in_generate"] = True

        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)

        if not generate_audio:
            return thinker_result, None

        # 2. Prepare talker input
        thinker_embed = torch.cat(
            [thinker_result.hidden_states[i][0] for i in range(len(thinker_result.hidden_states))], dim=1
        )  # [1 t d]
        thinker_hidden = torch.cat(
            [
                thinker_result.hidden_states[i][self.config.talker_config.accept_hidden_layer]
                for i in range(len(thinker_result.hidden_states))
            ],
            dim=1,
        )  # [1 t d]
        im_start_indexes = torch.cat(
            (
                torch.nonzero(input_ids[0] == self.config.im_start_token_id).squeeze(),
                torch.tensor([thinker_result.sequences.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
            ),
            dim=-1,
        )  # Shape [n_starts + 1]; Take batch 0 since batched inference is not supported here.
        multimodal_mask = (
            (thinker_result.sequences == self.config.thinker_config.audio_token_index) |
            (thinker_result.sequences == self.config.thinker_config.image_token_index) |
            (thinker_result.sequences == self.config.thinker_config.video_token_index)
        )  # [1 t] # fmt: skip
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
            self.thinker.get_input_embeddings()(
                torch.tensor(
                    [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                    device=self.thinker.device,
                    dtype=input_ids.dtype,
                )
            )
        ).chunk(3, dim=1)  # 3 * [1 1 d]

        talker_input_embeds = []  # [1 t d]
        talker_input_ids = []
        # For every chatml parts
        for i in range(len(im_start_indexes) - 1):
            im_start_index = im_start_indexes[i]
            segment_end_index = im_start_indexes[i + 1]
            role_token = input_ids[0][im_start_index + 1]
            # Talker should ignore thinker system prompt
            if role_token == self.config.system_token_id:
                continue
            # Talker takes word embeddings for tokens and hidden state from `accept_hidden_layer` for multimodal inputs
            elif role_token == self.config.user_token_id:
                user_talker_part = torch.empty(
                    (1, segment_end_index - im_start_index, self.config.talker_config.hidden_size),
                    device=self.talker.device,
                    dtype=self.talker.dtype,
                )

                user_mm_mask = multimodal_mask[:, im_start_index:segment_end_index]

                # Multimodal data exists
                if user_mm_mask.any():
                    user_thinker_hidden_mm = thinker_hidden[im_start_index:segment_end_index][user_mm_mask]
                    mm_hidden = self.talker.hidden_projection(user_thinker_hidden_mm)
                    user_talker_part[user_mm_mask] = mm_hidden
                user_thinker_embed = thinker_embed[:, im_start_index:segment_end_index][~user_mm_mask]
                user_text_hidden = self.talker.text_projection(user_thinker_embed)
                user_talker_part[~user_mm_mask] = user_text_hidden
                talker_input_embeds.append(user_talker_part)
                talker_input_ids.append(thinker_result.sequences[:, im_start_index:segment_end_index])
            # Take assistant output (for now)
            elif role_token == self.config.assistant_token_id and i == len(im_start_indexes) - 2:
                assistant_hidden = self.talker.text_projection(
                    thinker_embed[:, im_start_index:segment_end_index]
                )  # [1 t d]
                assistant_text_hidden = torch.cat(
                    (
                        assistant_hidden[:, :3],
                        tts_pad_embed.expand(-1, 4, -1),
                        tts_bos_embed,
                        assistant_hidden[:, 3:4],  # First text
                    ),
                    dim=1,
                )
                assistant_codec_hidden = torch.cat(
                    (
                        torch.zeros(
                            (1, 3, self.config.talker_config.hidden_size),
                            device=self.talker.device,
                            dtype=self.talker.dtype,
                        ),
                        self.talker.get_input_embeddings()(
                            torch.tensor(
                                [
                                    [
                                        self.config.talker_config.codec_nothink_id,
                                        self.config.talker_config.codec_think_bos_id,
                                        self.config.talker_config.codec_think_eos_id,
                                        # self.config.talker_config.codec_spk_id,
                                        spk_id,
                                        self.config.talker_config.codec_pad_id,
                                        self.config.talker_config.codec_bos_id,
                                    ]
                                ],
                                device=self.talker.device,
                                dtype=input_ids.dtype,
                            )
                        ),
                    ),
                    dim=1,
                )
                trailing_text_hidden = torch.cat(
                    (
                        assistant_hidden[:, 4:],
                        tts_eos_embed,
                    ),
                    dim=1,
                )
                talker_input_embeds.append(assistant_text_hidden + assistant_codec_hidden)
                talker_input_ids.append(
                    torch.empty(
                        (1, assistant_text_hidden.shape[1]), dtype=input_ids.dtype, device=assistant_text_hidden.device
                    ).fill_(self.config.tts_pad_token_id)
                )
            # History assistant output (ignore for now)
            elif role_token == self.config.assistant_token_id and i == len(im_start_indexes) - 2:
                continue
            else:
                raise AssertionError("Unreachable")
        talker_input_embed = torch.cat(talker_input_embeds, dim=1)
        talker_input_id = torch.cat(talker_input_ids, dim=1)
        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embed,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            talker_input_ids=talker_input_id,  # Not use input_ids to prevent repetation penalty out of bound
            **talker_kwargs,
        )
        talker_codes = torch.stack([hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1)

        return thinker_result, talker_codes


class Qwen3OmniMoeProcessor(Qwen3OmniProcessor):
    pass


__all__ = [
    "Qwen3OmniMoeConfig",
    "Qwen3OmniMoeThinkerConfig",
    "Qwen3OmniMoeTalkerConfig",
    "Qwen3OmniMoeForConditionalGeneration",
    "Qwen3OmniMoeThinkerTextModel",
    "Qwen3OmniMoeThinkerForConditionalGeneration",
    "Qwen3OmniMoeTalkerForConditionalGeneration",
    "Qwen3OmniMoePreTrainedModel",
    "Qwen3OmniMoePreTrainedModelForConditionalGeneration",
    "Qwen3OmniMoeTalkerModel",
    "Qwen3OmniMoeThinkerTextPreTrainedModel",
    "Qwen3OmniMoeProcessor",
]
