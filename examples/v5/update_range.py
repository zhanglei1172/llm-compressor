import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file, save_file


def load_input_scales_from_safetensors(model_dir: str) -> Dict[str, torch.Tensor]:
    """
    从HF模型的safetensors文件中读取所有input_scale参数

    Args:
        model_dir: HF模型目录路径，应包含 model.safetensors.index.json

    Returns:
        字典，键为参数名（如 'model.layers.11.mlp.up_proj.input_scale'），
        值为对应的torch.Tensor（通常为标量tensor）

    Example:
        >>> scales = load_input_scales_from_safetensors('/path/to/model')
        >>> print(scales)
        {
            'model.layers.0.mlp.up_proj.input_scale': 123.456,
            'model.layers.0.mlp.down_proj.input_scale': 234.567,
            ...
        }
    """
    model_path = Path(model_dir)
    index_file = model_path / "model.safetensors.index.json"

    if not index_file.exists():
        raise FileNotFoundError(f"未找到索引文件: {index_file}")

    # 读取索引文件
    with open(index_file, "r") as f:
        index_data = json.load(f)

    # weights 字段包含参数名到文件的映射
    weights_map = index_data.get("weight_map", {})

    # 收集所有input_scale参数和对应的文件
    input_scales_files: Dict[str, str] = {}
    for param_name, file_name in weights_map.items():
        if "input_scale" in param_name:
            input_scales_files[param_name] = file_name

    if not input_scales_files:
        print(f"警告: 未在 {index_file} 中找到任何input_scale参数")
        return {}

    # 按文件分组，减少文件读取次数
    files_to_params: Dict[str, list] = {}
    for param_name, file_name in input_scales_files.items():
        if file_name not in files_to_params:
            files_to_params[file_name] = []
        files_to_params[file_name].append(param_name)

    # 从各个safetensors文件中读取权重
    input_scales = {}
    for file_name, param_names in files_to_params.items():
        file_path = model_path / file_name

        if not file_path.exists():
            print(f"警告: 文件不存在 {file_path}")
            continue

        print(f"读取文件: {file_name}")

        try:
            # 只读取需要的参数
            tensors = load_file(file_path, device="cpu")

            for param_name in param_names:
                if param_name in tensors:
                    # 将tensor转换为Python float
                    tensor_value = tensors[param_name]
                    input_scales[param_name] = tensor_value
                    # if tensor_value.numel() == 1:
                    #     input_scales[param_name] = float(tensor_value.item())
                    # else:
                    #     # 如果是数组，取第一个值
                    #     print(f"警告: {param_name} 不是标量，使用第一个元素")
                    #     input_scales[param_name] = float(
                    #         tensor_value.flatten()[0].item()
                    #     )
                else:
                    print(f"警告: 参数 {param_name} 不在文件 {file_name} 中")

        except Exception as e:
            print(f"错误: 读取文件 {file_path} 时出错: {e}")
            continue

    print(f"\n成功读取 {len(input_scales)} 个input_scale参数")
    return input_scales


def update_range(
    input_scales: Dict[str, torch.Tensor],
    keys: list,
    mapping: Dict[str, str],
    new_scale: Optional[Dict[str, Any]] = None,
):
    if new_scale is None:
        raise ValueError("new_scale 不能为空")
    mapping_keys = copy.deepcopy(mapping)
    mapping_keys = set(mapping.keys())
    for name in keys:
        for module_type in mapping_keys:
            if module_type in name:
                input_scales[name] = (
                    new_scale[
                        (
                            f"{mapping[module_type]}.activation_quantizer.quantizer._delta"
                        ).format(name.split(".")[2])
                    ]
                    .detach()
                    .to(
                        dtype=input_scales[name].dtype,
                        device=input_scales[name].device,
                    )
                )


def update_range_to_model(
    model_dir: str, input_scales: Dict[str, torch.Tensor]
) -> None:
    """
    将更新后的input_scale参数写回到模型的safetensors文件中。

    Args:
        model_dir: HF模型目录路径，应包含 model.safetensors.index.json
        input_scales: 包含要更新的input_scale参数及其新值的字典
    """
    model_path = Path(model_dir)
    index_file = model_path / "model.safetensors.index.json"

    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    # 读取索引文件
    with open(index_file, "r") as f:
        index_data = json.load(f)

    # weights 字段包含参数名到文件的映射
    weights_map = index_data.get("weight_map", {})

    # 先按文件分组所有待更新参数，避免重复IO
    files_to_updates: Dict[str, Dict[str, torch.Tensor]] = {}
    for param_name, file_name in weights_map.items():
        if param_name in input_scales:
            if file_name not in files_to_updates:
                files_to_updates[file_name] = {}
            files_to_updates[file_name][param_name] = input_scales[param_name]

    # 对每个文件只读取一次并统一写回
    for file_name, updates in files_to_updates.items():
        safetensor_file = model_path / file_name
        if not safetensor_file.exists():
            raise FileNotFoundError(f"Safetensor file not found: {safetensor_file}")
        update_safetensors_file_batch(safetensor_file, updates)

    # 可能需要重新保存索引文件或其他操作
    # ...其他操作...


def update_safetensor_file(
    safetensor_file: Path, param_name: str, new_value: torch.Tensor
) -> None:
    """
    更新safetensors文件中的指定参数。

    Args:
        safetensor_file: safetensors文件的路径
        param_name: 要更新的参数名
        new_value: 新的参数值
    """
    # 这里假设我们有一个方法可以读取和更新safetensors文件
    # 读取safetensors文件内容
    data = load_file(safetensor_file)

    # 更新指定参数
    if param_name in data:
        # 确保类型/设备一致
        existing = data[param_name]
        if isinstance(new_value, torch.Tensor):
            data[param_name] = new_value.to(
                dtype=existing.dtype,
                device=existing.device,
            )
        else:
            data[param_name] = torch.tensor(
                new_value,
                dtype=existing.dtype,
                device=existing.device,
            )
    else:
        raise KeyError(f"Parameter {param_name} not found in {safetensor_file}")

    # 保存更新后的数据回safetensors文件
    # 注意save_file参数顺序: tensors在前, 文件路径在后
    save_file(data, str(safetensor_file))


def update_safetensors_file_batch(
    safetensor_file: Path, updates: Dict[str, torch.Tensor]
) -> None:
    """
    批量更新同一个safetensors文件中的多个参数，仅一次读写。

    Args:
        safetensor_file: safetensors文件的路径
        updates: 需要更新的参数名到新tensor的映射
    """
    data = load_file(safetensor_file)

    for param_name, new_value in updates.items():
        if param_name in data:
            existing = data[param_name]
            if isinstance(new_value, torch.Tensor):
                data[param_name] = new_value.to(
                    dtype=existing.dtype,
                    device=existing.device,
                )
            else:
                data[param_name] = torch.tensor(
                    new_value,
                    dtype=existing.dtype,
                    device=existing.device,
                )
        else:
            print(f"警告: 参数 {param_name} 不在文件 {safetensor_file.name} 中，跳过")

    save_file(data, str(safetensor_file))


if __name__ == "__main__":
    model_path = "/tmp/ostq-gptq-lrqat_klt-my-fp32-r4_mse_w4a8-realq/"
    new_scale = torch.load("/tmp/ostq-gptq-lrqat_klt-my-fp32/quant_params.pt")
    # 示例: 过滤掉权重量化相关的keys
    # 示例用途，如需请自行启用
    # new_scale_keys = [
    #     x for x in list(new_scale.keys())
    #     if 'weight_quantizer' not in x
    # ]

    mapping = {
        "up_proj": "model.language_model.layers.{}.mlp_ln_act_quantizer",
        # "gate_proj": "model.language_model.layers.{}.mlp_ln_act_quantizer",
        "down_proj": "model.language_model.layers.{}.mlp.mlp_mul_act_quantizer",
        # "q_proj": "model.language_model.layers.{}.attn_ln_act_quantizer",
        "k_proj": "model.language_model.layers.{}.attn_ln_act_quantizer",
        # "v_proj": "model.language_model.layers.{}.attn_ln_act_quantizer",
        "o_proj": "model.language_model.layers.{}.self_attn.attn_act_re_quantizer",
    }

    input_scales = load_input_scales_from_safetensors(model_path)
    print(f"读取到 {len(input_scales)} 个input_scale参数")
    print("\n前10个参数:")
    for i, (name, value) in enumerate(sorted(input_scales.items())[:10]):
        print(f"  {name}: {value}")
    keys = input_scales.keys()
    keys = list(sorted(keys, key=lambda x: (int(x.split(".")[2]), x)))
    num_layers = int(keys[-1].split(".")[2]) + 1
    print(f"\n模型总层数: {num_layers}")

    update_range(
        input_scales,
        keys,
        mapping=mapping,
        new_scale=new_scale,
    )

    update_range_to_model(model_path, input_scales)
