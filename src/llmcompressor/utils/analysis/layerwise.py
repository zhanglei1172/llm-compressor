from typing import Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm

from ..measure import MeasurePrinter, MeasureRecorder

@torch.no_grad()
def layerwise_error_analyse(
    model: torch.nn.Module,
    dataloader: Iterable,
    method: str = "snr",
    steps: int = 8,
    verbose: bool = True,
) -> Dict[str, tuple]:
    # find all quantable operations.
    quantable_operations: List[Tuple[str, torch.nn.Module]] = []
    for name, operation in model.named_modules():
        if hasattr(operation, "quantization_status"):
            operation.quantization_enabled = False

            quantable_operations.append((name, operation))

    # dequantize all operations, create recorder for each operation
    recorders = {}
    for name, operation in quantable_operations:
        if hasattr(operation, "quantization_status"):
            recorders[name] = MeasureRecorder(measurement=method)

    # run for each quantable operations:
    for name, operation in tqdm(
        quantable_operations, desc="Analysing Layerwise quantization error:"
    ):
        assert hasattr(operation, "quantization_status")
        recorder = recorders[name]
        assert isinstance(recorder, MeasureRecorder)

        for idx, batch in enumerate(dataloader):
            fp_outputs = model(batch)

            # manually override quantization state
            operation.quantization_enabled = True
            qt_outputs = model(batch)

            recorder.update(y_pred=qt_outputs, y_real=fp_outputs)

            # manually override quantization state
            operation.quantization_enabled = False
            if idx >= steps:
                break

    # restore quantization states
    for name, operation in model.named_modules():
        if hasattr(operation, "quantization_status"):
            operation.quantization_enabled = True

    results = {}
    for name, operation in quantable_operations:
        if hasattr(operation, "quantization_status"):
            results[name] = recorders[name].measure

    if verbose:
        method_str = "MEASUREMENT"
        if method == "snr":
            method_str = "NOISE:SIGNAL POWER RATIO"
        if method == "cosine":
            method_str = "COSINE SIMILARITY"
        if method == "mse":
            method_str = "MSE LOSS(UNSCALED)"
        if method == "sqnr":
            method_str = "SIGNAL:QUANTIZATION NOISE RATIO"
        if method == "kl":
            method_str = "KL DIVERGENCE"
        MeasurePrinter(
            results,
            order="large_to_small",
            measure=method_str,
            percentage=method in {"snr", "cosine"},
        ).print()
    return results
