"""
Utility / helper functions
"""

import contextlib
import copy
import warnings
from typing import Dict, List, Union

import torch
from compressed_tensors.quantization.utils import is_module_quantized
from compressed_tensors.utils import match_named_modules
from loguru import logger
from torch.nn import Module
from transformers import PreTrainedModel

from llmcompressor.core import ModelParameterizedLayer

__all__ = [
    "expand_special_targets",
    "build_parameterized_layers",
    "qat_active",
    "get_no_split_params",
]

ALL_TARGET = "__ALL__"
ALL_PRUNABLE_TARGET = "__ALL_PRUNABLE__"
ALL_QUANTIZABLE_TARGET = "__ALL_QUANTIZABLE__"


def expand_special_targets(targets: Union[str, List[str]]) -> List[str]:
    """
    Expand special target constants to explicit class names with backward compatibility.

    Special constants like __ALL_PRUNABLE__ and __ALL_QUANTIZABLE__ are deprecated
    in favor of explicit class name lists. This function provides backward compatibility
    by expanding these constants while issuing deprecation warnings.

    :param targets: Target strings which may include special constants
    :return: List of expanded target strings
    :raises ValueError: If __ALL__ constant is used (no longer supported)
    """
    if isinstance(targets, str):
        targets = [targets]

    expanded = []
    for target in targets:
        if target == ALL_PRUNABLE_TARGET:
            warnings.warn(
                f"{ALL_PRUNABLE_TARGET} is deprecated. "
                "Use explicit targets: ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']",
                DeprecationWarning,
                stacklevel=3,
            )
            expanded.extend(["Linear", "Conv1d", "Conv2d", "Conv3d"])
        elif target == ALL_QUANTIZABLE_TARGET:
            warnings.warn(
                f"{ALL_QUANTIZABLE_TARGET} is deprecated. "
                "Use explicit targets: ['Linear', 'Conv2d', 'Conv3d']",
                DeprecationWarning,
                stacklevel=3,
            )
            expanded.extend(["Linear", "Conv2d", "Conv3d"])
        elif target == ALL_TARGET:
            raise ValueError(
                f"{ALL_TARGET} is no longer supported. "
                "Use explicit layer types or patterns instead."
            )
        else:
            expanded.append(target)

    return expanded


def build_parameterized_layers(
    model: Module,
    targets: Union[str, List[str]],
    param_name: str = "weight",
) -> Dict[str, ModelParameterizedLayer]:
    """
    Build ModelParameterizedLayer objects for modules matching the given targets.

    This function replaces get_layers_params() by using compressed-tensors'
    match_named_modules() to find matching modules and their parameters,
    then constructing ModelParameterizedLayer objects.

    :param model: The model to search for matching modules
    :param targets: Target patterns to match (supports class names, regex with "re:",
                    and special constants for backward compatibility)
    :param param_name: Name of the parameter to extract from each layer
        (default: "weight")
    :return: Dictionary mapping layer names to ModelParameterizedLayer objects
    """
    # Expand special constants if present
    targets = expand_special_targets(targets)

    parameterized_layers = {}
    for layer_name, module in match_named_modules(model, targets):
        # Get the parameter from the module
        param = getattr(module, param_name, None)
        if param is None:
            continue

        # Avoid duplicate entries (same layer can be matched multiple times)
        if layer_name not in parameterized_layers:
            parameterized_layers[layer_name] = ModelParameterizedLayer(
                layer_name=layer_name,
                layer=module,
                param_name=f"{layer_name}.{param_name}",
                param=param,
            )

    return parameterized_layers


def qat_active(module: Module) -> bool:
    """
    Determines if any layers in the model have quantization enabled by checking for
    weight_fake_quant attributes

    :param module: PyTorch model to check for quantization
    :return: True if quantization is active anywhere in the model, False otherwise
    """
    for _, layer in module.named_modules():
        if isinstance(layer, torch.quantization.FakeQuantize):
            return True
        if is_module_quantized(layer):
            return True

    return False


def get_no_split_params(model: PreTrainedModel) -> Union[str, List[str]]:
    """
    Get list of module classes that shouldn't be split when sharding. For
    Hugging Face Transformer models, this is the decoder layer type. For other
    types of models, this just returns all module names.

    :return: list of class names that shouldn't be split
    """
    no_split_modules = model._get_no_split_modules("auto")
    if len(no_split_modules) <= 0:
        return ALL_TARGET

    return no_split_modules


# https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8


def get_module_name(model, module):
    """
    Get the name of the module in the model.
    """
    for name, mod in model.named_modules():
        if mod is module:
            return name
    return None


@contextlib.contextmanager
def patch_module_non_persistent_buffers(model: torch.nn.Module):
    change_records = []
    for _, module in model.named_modules():
        # register_buffer as persistent to avoid issues in load_state_dict
        change_name_list = list(module._non_persistent_buffers_set)
        for buffer_name in change_name_list:
            module._non_persistent_buffers_set.discard(buffer_name)
            change_records.append((module, buffer_name))
    try:
        yield
    finally:
        for module, buffer_name in change_records:
            module._non_persistent_buffers_set.add(buffer_name)


def get_non_persistent_buffers(
    module: torch.nn.Module, recurse: bool = False, fqns: bool = False
):
    """
    Gather all non persistent buffers of a given modules into a set

    Args:
        module (`nn.Module`):
            The module we want the non persistent buffers on.
        recurse (`bool`, *optional*, defaults to `False`):
            Whether or not to go look in every submodule or just return the direct non persistent buffers.
        fqns (`bool`, *optional*, defaults to `False`):
            Whether or not to return the fully-qualified names of the non persistent buffers.
    """

    non_persistent_buffers_set = module._non_persistent_buffers_set
    if recurse:
        for n, m in module.named_modules():
            if fqns:
                non_persistent_buffers_set |= {
                    n + "." + b for b in m._non_persistent_buffers_set
                }
            else:
                non_persistent_buffers_set |= m._non_persistent_buffers_set

    return non_persistent_buffers_set


@contextlib.contextmanager
def patch_tensor_to_cuda(base: object = torch.Tensor):
    """
    Patch the value of an object attribute. Original value is restored upon exit

    :param base: object which has the attribute to patch
    :param attr: name of the the attribute to patch
    :param value: used to replace original value

    Usage:
    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace()
    >>> with patch_attr(obj, "attribute", "value"):
    ...     assert obj.attribute == "value"
    >>> assert not hasattr(obj, "attribute")
    """
    # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    attr = "cuda"
    _sentinel = object()
    original_value = getattr(base, attr, _sentinel)

    # setattr(base, attr, module_to_cuda.__get__(base))
    setattr(base, attr, tensor_to_cuda)
    try:
        yield
    finally:
        if original_value is not _sentinel:
            setattr(base, attr, original_value)
        else:
            delattr(base, attr)


@torch.no_grad()
def tensor_to_cuda(self: torch.Tensor, device="cuda"):
    """
    Move a module to CUDA
    :param module: module to move
    """
    if device == "cuda":
        device = f"cuda:{torch.cuda.current_device()}"
    if not isinstance(self, torch.Tensor):
        return self.to(device)
    src_rank = 0

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    if rank == src_rank:
        self = self.to(device)
    else:
        self = torch.empty_like(self, device=device)
    torch.distributed.broadcast(self, src=src_rank)
    return self


@contextlib.contextmanager
def patch_module_to_cuda(base: object = torch.nn.Module):
    """
    Patch the value of an object attribute. Original value is restored upon exit

    :param base: object which has the attribute to patch
    :param attr: name of the the attribute to patch
    :param value: used to replace original value

    Usage:
    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace()
    >>> with patch_attr(obj, "attribute", "value"):
    ...     assert obj.attribute == "value"
    >>> assert not hasattr(obj, "attribute")
    """
    # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    attr = "cuda"
    _sentinel = object()
    original_value = getattr(base, attr, _sentinel)

    # setattr(base, attr, module_to_cuda.__get__(base))
    setattr(base, attr, module_to_cuda)
    try:
        yield
    finally:
        if original_value is not _sentinel:
            setattr(base, attr, original_value)
        else:
            delattr(base, attr)


@torch.no_grad()
def module_to_cuda(self: torch.nn.Module, device="cuda"):
    """
    Move a module to CUDA
    :param module: module to move
    """
    if not isinstance(self, torch.nn.Module):
        return self.to("cuda")
    src_rank = 0
    if not device:
        device = "cuda"
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    non_persistent_buffer_fqns = get_non_persistent_buffers(
        self, recurse=True, fqns=True
    )

    _sd = {}
    meta_sd = self.state_dict()

    for name, sd_param in meta_sd.items():
        if rank == src_rank:
            send_tensor = sd_param.to(device)
            torch.distributed.broadcast(send_tensor, src=src_rank)
            _sd[name] = send_tensor
        else:
            recv_tensor = torch.empty_like(sd_param, device=device)
            torch.distributed.broadcast(recv_tensor, src=src_rank)
            _sd[name] = recv_tensor
    self.load_state_dict(_sd, assign=True)

    del _sd

    original_non_persistent_buffers = copy.deepcopy(
        {k: v for k, v in self.named_buffers() if k in non_persistent_buffer_fqns}
    )

    for fqn, buffer_tensor in original_non_persistent_buffers.items():
        if rank == src_rank:
            buffer_tensor = buffer_tensor.to(device)
        else:
            buffer_tensor = torch.empty_like(buffer_tensor, device=device)

        if "." in fqn:
            parent_fqn, local_buffer_name = fqn.rsplit(".", 1)
            parent_module = self.get_submodule(parent_fqn)
        else:
            local_buffer_name = fqn
            parent_module = self

        parent_module.register_buffer(
            local_buffer_name, buffer_tensor, persistent=False
        )

    return self


class UnionFind:
    """并查集数据结构，用于管理共享参数的映射关系"""

    def __init__(self):
        self.parent = {}  # 存储每个元素的父节点
        self.rank = {}  # 存储每个集合的秩（用于优化）

    def find(self, x):
        """查找元素 x 的根节点（代表元素），路径压缩优化"""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x

        if self.parent[x] != x:
            # 路径压缩：将查找路径上的所有节点直接连接到根节点
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y):
        """合并 x 和 y 所在的集合"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return  # 已在同一集合

        # 按秩合并：将秩小的树连接到秩大的树下
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


def build_weight_tied_map_with_unionfind(model: torch.nn.Module, remove_duplicate=True):
    weight_tied_map = {}
    weight_tied_name_map = {}
    uf = UnionFind()
    param_id_to_first_name = {}

    # 第一遍：收集所有参数
    for module_name, module in model.named_modules(remove_duplicate=remove_duplicate):
        for param_name, param in module.named_parameters(
            recurse=False, remove_duplicate=remove_duplicate
        ):
            if not param.requires_grad:
                continue

            param_id = id(param)

            if param_id not in param_id_to_first_name:
                # 第一次遇到这个参数
                param_id_to_first_name[param_id] = (module_name, param_name)
                weight_tied_map[param_id] = param
            else:
                # 共享参数：绑定到第一个参数
                first_name = param_id_to_first_name[param_id]
                uf.union(first_name, (module_name, param_name))
                param.data = weight_tied_map[param_id].data

            weight_tied_name_map[(module_name, param_name)] = param_id

    # 构建规范名称映射
    canonical_name_map = {}
    for name in weight_tied_name_map.keys():
        canonical_name_map[name] = uf.find(name)

    return canonical_name_map


def get_module_to_name_dict(model: Module) -> dict[Module, str]:
    module_to_name = {}
    for name, module in model.named_modules():
        if module in module_to_name:
            logger.warning(
                f"Warning, {name} and {module_to_name[module]} both "
                "share the same module, which can result in unexpected behavior"
            )
        module_to_name[module] = name
    return module_to_name
