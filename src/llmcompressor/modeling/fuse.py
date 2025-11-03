from typing import Iterable

import torch
from compressed_tensors import (
    align_module_device,
    get_execution_device,
    update_offload_parameter,
)

__all__ = ["center_embeddings", "fuse_norm_linears", "back_mean_into_fc"]


PRECISION = torch.float64


def center_embeddings(embedding: torch.nn.Module):
    """
    Shift each embedding to have a mean of zero

    :param embedding: embedding module containing embeddings to center
    """
    if not hasattr(embedding, "weight"):
        raise ValueError(f"Cannot fuse norm of type {type(embedding)}")

    with align_module_device(embedding):
        weight_dtype = embedding.weight.dtype
        weight = embedding.weight.to(PRECISION)
        new_weight = weight - weight.mean(dim=-1, keepdim=True)
        new_weight = new_weight.to(weight_dtype)

    update_offload_parameter(embedding, "weight", new_weight)


def back_mean_into_fc(linear: torch.nn.Linear):
    exec_device = get_execution_device(linear)
    with align_module_device(linear, exec_device):
        weight_dtype = linear.weight.dtype
        new_weight = linear.weight.to(PRECISION)
        new_weight = new_weight - new_weight.mean(dim=-2, keepdim=True)
        new_weight = new_weight.to(weight_dtype)
        if hasattr(linear, "bias") and linear.bias is not None:
            new_bias = linear.bias.to(PRECISION)
            new_bias = new_bias - new_bias.mean()
            new_bias = new_bias.to(weight_dtype)

    update_offload_parameter(linear, "weight", new_weight)
    if hasattr(linear, "bias") and linear.bias is not None:
        update_offload_parameter(linear, "bias", new_bias)


def fuse_norm_linears(norm: torch.nn.Module, linears: Iterable[torch.nn.Linear]):
    """
    Fuse the scaling operation of norm layer into subsequent linear layers.
    This useful for ensuring transform invariance between norm and linear layers.

    Note that unitary transforms (rotation) commute with normalization, but not scaling

    :param norm: norm layer whose weight will be fused into subsequent linears
    :param linears: linear layers which directly follow the norm layer
    """
    if not hasattr(norm, "weight"):
        raise ValueError(f"Cannot fuse norm of type {type(norm)}")

    for linear in linears:
        new_bias = None
        # NOTE: spinquant does this op in float64
        exec_device = get_execution_device(norm)
        with align_module_device(norm, exec_device), align_module_device(
            linear, exec_device
        ):
            fc_hidden_size = linear.weight.shape[-1]
            ln_hidden_size = norm.weight.shape[-1]
            weight_dtype = linear.weight.dtype
            if fc_hidden_size > ln_hidden_size:
                norm_weight = norm.weight.data.repeat(fc_hidden_size // ln_hidden_size)
            else:
                norm_weight = norm.weight
            new_weight = linear.weight.to(PRECISION) * norm_weight.to(PRECISION)
            new_weight = new_weight.to(weight_dtype)
            if hasattr(norm, "bias") and norm.bias is not None:
                if fc_hidden_size > ln_hidden_size:
                    norm_bias = norm.bias.data.repeat(fc_hidden_size // ln_hidden_size)
                else:
                    norm_bias = norm.bias
                new_bias = linear.bias.to(PRECISION) + linear.weight.to(
                    PRECISION
                ).matmul(norm_bias.to(PRECISION))
                new_bias = new_bias.to(weight_dtype)
                norm.register_parameter("bias", None)

        update_offload_parameter(linear, "weight", new_weight)
        if new_bias is not None:
            update_offload_parameter(linear, "bias", new_bias)

    new_norm_weight = torch.ones_like(norm.weight, device="cpu")
    update_offload_parameter(norm, "weight", new_norm_weight)
