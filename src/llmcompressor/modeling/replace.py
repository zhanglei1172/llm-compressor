import torch
from compressed_tensors import (
    align_module_device,
    get_execution_device,
    update_offload_parameter,
)
from compressed_tensors.utils.offload import add_hook_to_module
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations


def replace_ln_to_rmsnorm(name: str, module: torch.nn.Module, model: torch.nn.Module):
    """
    Replace LayerNorm with RMSNorm in the model for better quantization performance.
    This is useful for ensuring transform invariance between norm and linear layers.

    :param norm: norm layer to be replaced
    """
    if not isinstance(module, torch.nn.LayerNorm):
        return

    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=module.normalized_shape,
        eps=module.eps,
        elementwise_affine=module.elementwise_affine,
    )

    if hasattr(module, "_hf_hook"):
        add_hook_to_module(rmsnorm, module._hf_hook)

    exec_device = get_execution_device(module)
    with align_module_device(module, exec_device):
        weight = module.weight

    parent_module = None
    attr_name = None
    if "." in name:
        *parent_parts, attr_name = name.rsplit(".", 1)
        parent_name = ".".join(parent_parts)
        parent_module = model.get_submodule(parent_name)
    else:
        attr_name = name
        parent_module = model

    setattr(parent_module, attr_name, rmsnorm)
    update_offload_parameter(rmsnorm, "weight", weight)
    if (
        f"{rmsnorm._hf_hook.weights_map.prefix}bias"
        in rmsnorm._hf_hook.weights_map.dataset.state_dict
    ):
        del rmsnorm._hf_hook.weights_map.dataset.state_dict[
            f"{rmsnorm._hf_hook.weights_map.prefix}bias"
        ]


def replace_parametrizations_to_weights(model: torch.nn.Module):
    """
    Fold the parametrizations into the weight of the module.

    :param module: target module to apply parametrizations to
    """
    for _, module in model.named_modules():
        if is_parametrized(module):
            change_keys = list(module.parametrizations.keys())
            for key in change_keys:
                remove_parametrizations(module, key)
