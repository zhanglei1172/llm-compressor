import contextlib
from functools import partial, wraps
from typing import Dict, Iterable, List, Tuple, Union

import torch
from tqdm import tqdm

from llmcompressor.utils import helpers

from ..measure import (
    METHOD_DISPLAY_NAMES,
    METHOD_USE_PERCENTAGE,
    SUPPORTED_METHODS,
    MeasurePrinter,
    MeasureRecorder,
)


def _normalize_methods(method: Union[str, List[str], None]) -> List[str]:
    """Normalize method parameter to a list of methods.

    Args:
        method: A single method string, list of methods, or None for all methods.

    Returns:
        List of method strings.

    Raises:
        ValueError: If an unsupported method is provided.
    """
    if method is None:
        return SUPPORTED_METHODS.copy()
    if isinstance(method, str):
        methods = [method]
    else:
        methods = list(method)

    for m in methods:
        if m not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{m}'. Supported methods: {SUPPORTED_METHODS}"
            )
    return methods


@torch.no_grad()
def layerwise_error_analyse(
    model: torch.nn.Module,
    dataloader: Iterable,
    method: Union[str, List[str], None] = "snr",
    steps: int = 8,
    verbose: bool = True,
) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Analyze quantization error at layer level.

    Args:
        model: The model to analyze.
        dataloader: DataLoader providing input batches.
        method: Measurement method(s). Can be:
            - A single method string (e.g., "snr", "cosine", "mse", "sqnr", "kl")
            - A list of methods (e.g., ["snr", "cosine"])
            - None to use all supported methods
        steps: Number of batches to analyze.
        verbose: Whether to print results.

    Returns:
        If method is a single string: Dict[str, float] mapping layer names to values.
        If method is a list or None: Dict[str, Dict[str, float]] mapping layer names
            to dicts of {method: value}.
    """
    # Determine if single method mode (for backward compatibility)
    single_method_mode = isinstance(method, str)
    methods = _normalize_methods(method)

    # find all quantable operations.
    quantable_operations: List[Tuple[str, torch.nn.Module]] = []
    for name, operation in model.named_modules():
        if hasattr(operation, "quantization_status"):
            operation.quantization_enabled = False

            quantable_operations.append((name, operation))

    # dequantize all operations, create recorder for each operation
    # recorders[name][method] = MeasureRecorder
    recorders: Dict[str, Dict[str, MeasureRecorder]] = {}
    for name, operation in quantable_operations:
        if hasattr(operation, "quantization_status"):
            recorders[name] = {m: MeasureRecorder(measurement=m) for m in methods}

    # run for each quantable operations:
    for name, operation in tqdm(
        quantable_operations, desc="Analysing Layerwise quantization error:"
    ):
        assert hasattr(operation, "quantization_status")
        layer_recorders = recorders[name]

        for idx, batch in enumerate(dataloader):
            fp_outputs = model(batch)

            # manually override quantization state
            operation.quantization_enabled = True
            qt_outputs = model(batch)

            # Update all method recorders
            for m in methods:
                layer_recorders[m].update(y_pred=qt_outputs, y_real=fp_outputs)

            # manually override quantization state
            operation.quantization_enabled = False
            if idx >= steps:
                break

    # restore quantization states
    for name, operation in model.named_modules():
        if hasattr(operation, "quantization_status"):
            operation.quantization_enabled = True

    # Collect results
    if single_method_mode:
        # Backward compatible: return Dict[str, float]
        results: Dict[str, float] = {}
        for name, operation in quantable_operations:
            if hasattr(operation, "quantization_status"):
                results[name] = recorders[name][methods[0]].measure
    else:
        # Multi-method mode: return Dict[str, Dict[str, float]]
        results: Dict[str, Dict[str, float]] = {}
        for name, operation in quantable_operations:
            if hasattr(operation, "quantization_status"):
                results[name] = {m: recorders[name][m].measure for m in methods}

    if verbose:
        if single_method_mode:
            # Single method: print once
            m = methods[0]
            method_str = METHOD_DISPLAY_NAMES.get(m, "MEASUREMENT")
            MeasurePrinter(
                results,
                order="large_to_small",
                measure=method_str,
                percentage=METHOD_USE_PERCENTAGE.get(m, False),
            ).print()
        else:
            # Multi-method: print for each method
            for m in methods:
                method_str = METHOD_DISPLAY_NAMES.get(m, "MEASUREMENT")
                # Extract single-method results for printing
                single_results = {name: vals[m] for name, vals in results.items()}
                print(f"\n{'=' * 60}")
                print(f"Method: {m.upper()}")
                print(f"{'=' * 60}")
                MeasurePrinter(
                    single_results,
                    order="large_to_small",
                    measure=method_str,
                    percentage=METHOD_USE_PERCENTAGE.get(m, False),
                ).print()

    return results


@torch.no_grad()
def layerwise_qdq_error_analyse(
    model: torch.nn.Module,
    dataloader: Iterable,
    method: Union[str, List[str], None] = "snr",
    steps: int = 8,
    verbose: bool = True,
) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
    # Determine if single method mode (for backward compatibility)
    single_method_mode = isinstance(method, str)
    methods = _normalize_methods(method)

    # find all quantable operations.
    quantable_operations: List[Tuple[str, torch.nn.Module]] = []
    for name, operation in model.named_modules():
        if hasattr(operation, "quantization_status"):
            operation.quantization_enabled = True

            quantable_operations.append((name, operation))

    with contextlib.ExitStack() as stack:
        recorders: Dict[str, Dict[str, MeasureRecorder]] = {}
        for name, operation in quantable_operations:
            if hasattr(operation, "quantization_status"):
                recorders[name] = {m: MeasureRecorder(measurement=m) for m in methods}
                layer_recorders = recorders[name]
                if hasattr(operation.forward, "__func__"):
                    forward_func_orig = operation.forward.__func__
                else:
                    forward_func_orig = operation.forward.func

                @wraps(
                    forward_func_orig
                )  # ensures docstring, names, etc are propagated
                def wrapped_forward(self, *args, layer_recorders=None, **kwargs):
                    self.quantization_enabled = True
                    qt_outputs = forward_func_orig.__get__(self, self.__class__)(
                        *args, **kwargs
                    )
                    self.quantization_enabled = False
                    fp_outputs = forward_func_orig.__get__(self, self.__class__)(
                        *args, **kwargs
                    )
                    # Update all method recorders
                    for m in methods:
                        layer_recorders[m].update(y_pred=qt_outputs, y_real=fp_outputs)
                    return fp_outputs

                bound_wrapped_forward = partial(
                    wrapped_forward.__get__(operation, operation.__class__),
                    layer_recorders=layer_recorders,
                )
                # setattr(operation, "forward", bound_wrapped_forward)
                stack.enter_context(
                    helpers.patch_attr(operation, "forward", bound_wrapped_forward)
                )

        for idx, batch in tqdm(
            enumerate(dataloader),
            desc="Analysing Layerwise Q-DQ quantization error:",
            total=(min(len(dataloader), steps)),
        ):
            fp_outputs = model(batch)
            if idx >= steps:
                break

    # restore quantization states
    for name, operation in model.named_modules():
        if hasattr(operation, "quantization_status"):
            operation.quantization_enabled = True

    # Collect results
    if single_method_mode:
        # Backward compatible: return Dict[str, float]
        results: Dict[str, float] = {}
        for name, operation in quantable_operations:
            if hasattr(operation, "quantization_status"):
                results[name] = recorders[name][methods[0]].measure
    else:
        # Multi-method mode: return Dict[str, Dict[str, float]]
        results: Dict[str, Dict[str, float]] = {}
        for name, operation in quantable_operations:
            if hasattr(operation, "quantization_status"):
                results[name] = {m: recorders[name][m].measure for m in methods}

    if verbose:
        if single_method_mode:
            # Single method: print once
            m = methods[0]
            method_str = METHOD_DISPLAY_NAMES.get(m, "MEASUREMENT")
            MeasurePrinter(
                results,
                order="large_to_small",
                measure=method_str,
                percentage=METHOD_USE_PERCENTAGE.get(m, False),
            ).print()
        else:
            # Multi-method: print for each method
            for m in methods:
                method_str = METHOD_DISPLAY_NAMES.get(m, "MEASUREMENT")
                # Extract single-method results for printing
                single_results = {name: vals[m] for name, vals in results.items()}
                print(f"\n{'=' * 60}")
                print(f"Method: {m.upper()}")
                print(f"{'=' * 60}")
                MeasurePrinter(
                    single_results,
                    order="large_to_small",
                    measure=method_str,
                    percentage=METHOD_USE_PERCENTAGE.get(m, False),
                ).print()

    return results
