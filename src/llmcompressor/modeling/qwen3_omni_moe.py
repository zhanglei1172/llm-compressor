from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3OmniMoeVisionAttention,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)


def forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        hidden_states: (batch_size * sequence_length, hidden_dim)
        selected_experts: (batch_size * sequence_length, top_k)
        routing_weights: (batch_size * sequence_length, top_k)
    Returns:
        (batch_size * sequence_length, hidden_dim)
    """
    final_hidden_states = torch.zeros_like(hidden_states)
    expert_mask = torch.nn.functional.one_hot(
        top_k_index, num_classes=self.num_experts
    ).permute(2, 1, 0)

    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        current_hidden_states = (
            self[expert_idx](hidden_states)[None, top_x].reshape(
                -1, hidden_states.shape[-1]
            )
            * top_k_weights[top_x, idx, None]
        )
        final_hidden_states.index_add_(
            0, top_x, current_hidden_states.to(hidden_states.dtype)
        )
    return final_hidden_states


class ReformQwen3OmniMoeVisionAttention(nn.Module):
    def __init__(self, ori_module) -> None:
        super().__init__()
        self.dim = ori_module.config.hidden_size
        self.num_heads = ori_module.config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        # self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.q_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = ori_module.config
        self.attention_dropout = 0.0
        self.is_causal = False

        self.q_proj.weight.data = ori_module.qkv.weight.data[: self.dim, :]
        self.q_proj.bias.data = ori_module.qkv.bias.data[: self.dim]
        self.k_proj.weight.data = ori_module.qkv.weight.data[self.dim : 2 * self.dim, :]
        self.k_proj.bias.data = ori_module.qkv.bias.data[self.dim : 2 * self.dim]
        self.v_proj.weight.data = ori_module.qkv.weight.data[2 * self.dim :, :]
        self.v_proj.bias.data = ori_module.qkv.bias.data[2 * self.dim :]

        self.proj.weight.data = ori_module.proj.weight.data
        self.proj.bias.data = ori_module.proj.bias.data

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        # query_states, key_states, value_states = (
        #     self.qkv(hidden_states)
        #     .reshape(seq_length, 3, self.num_heads, -1)
        #     .permute(1, 0, 2, 3)
        #     .unbind(0)
        # )
        query_states = self.q_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1
        )
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


def replace_vit_attention(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, Qwen3OmniMoeVisionAttention):
            replaced = ReformQwen3OmniMoeVisionAttention(child)
            setattr(module, name, replaced)
        else:
            replace_vit_attention(child)


def replace_vit_attention_inv(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, ReformQwen3OmniMoeVisionAttention):
            # Revert back to original Qwen3OmniMoeVisionAttention
            original = Qwen3OmniMoeVisionAttention(child.config)
            # Copy weights back
            original.qkv.weight.data[: child.dim, :] = child.q_proj.weight.data
            original.qkv.bias.data[: child.dim] = child.q_proj.bias.data
            original.qkv.weight.data[child.dim : 2 * child.dim, :] = (
                child.k_proj.weight.data
            )
            original.qkv.bias.data[child.dim : 2 * child.dim] = child.k_proj.bias.data
            original.qkv.weight.data[2 * child.dim :, :] = child.v_proj.weight.data
            original.qkv.bias.data[2 * child.dim :] = child.v_proj.bias.data
            original.proj.weight.data = child.proj.weight.data
            original.proj.bias.data = child.proj.bias.data

            setattr(module, name, original)
        else:
            replace_vit_attention_inv(child)


def replace():
    return forward
