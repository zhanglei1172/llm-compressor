import math
from functools import partial
from typing import Dict

import torch
from numpy import dot, ndarray
from numpy.linalg import norm


def torch_cosine_similarity(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    reduction: str = "mean",
    flatten_start_dim=1,
) -> torch.Tensor:
    if y_pred.shape != y_real.shape:
        raise ValueError(
            f"Can not compute mse loss for tensors with different shape. "
            f"({y_pred.shape} and {y_real.shape})"
        )
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=flatten_start_dim).float()
    y_real = y_real.flatten(start_dim=flatten_start_dim).float()

    cosine_sim = torch.cosine_similarity(y_pred, y_real, dim=-1)

    if reduction == "mean":
        return torch.mean(cosine_sim)
    elif reduction == "sum":
        return torch.sum(cosine_sim)
    elif reduction == "none":
        return cosine_sim
    else:
        raise ValueError(f"Unsupported reduction method.")


def torch_mean_square_error(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    reduction: str = "mean",
    flatten_start_dim=1,
) -> torch.Tensor:
    """
    Compute mean square error between y_pred(tensor) and y_real(tensor)

    MSE error can be calcualted as following equation:

        MSE(x, y) = (x - y) ^ 2

    if x and y are matrixs, MSE error over matrix should be the mean value of MSE error over all elements.

        MSE(X, Y) = mean((X - Y) ^ 2)

    By this equation, we can easily tell that MSE is an symmtrical measurement:
        MSE(X, Y) == MSE(Y, X)
        MSE(0, X) == X ^ 2

    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        torch.Tensor: _description_
    """
    if y_pred.shape != y_real.shape:
        raise ValueError(
            f"Can not compute mse loss for tensors with different shape. "
            f"({y_pred.shape} and {y_real.shape})"
        )
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    diff = torch.pow(y_pred - y_real, 2).flatten(start_dim=flatten_start_dim)
    mse = torch.mean(diff, dim=-1)

    if reduction == "mean":
        return torch.mean(mse)
    elif reduction == "sum":
        return torch.sum(mse)
    elif reduction == "none":
        return mse
    else:
        raise ValueError(f"Unsupported reduction method.")


def torch_snr_error(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    reduction: str = "mean",
    flatten_start_dim=1,
) -> torch.Tensor:
    """
    Compute SNR between y_pred(tensor) and y_real(tensor)

    SNR can be calcualted as following equation:

        SNR(pred, real) = (pred - real) ^ 2 / (real) ^ 2

    if x and y are matrixs, SNR error over matrix should be the mean value of SNR error over all elements.

        SNR(pred, real) = mean((pred - real) ^ 2 / (real) ^ 2)

    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        torch.Tensor: _description_
    """
    if y_pred.shape != y_real.shape:
        raise ValueError(
            f"Can not compute snr loss for tensors with different shape. "
            f"({y_pred.shape} and {y_real.shape})"
        )
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=flatten_start_dim)
    y_real = y_real.flatten(start_dim=flatten_start_dim)

    noise_power = torch.pow(y_pred - y_real, 2).sum(dim=-1)
    signal_power = torch.pow(y_real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)

    if reduction == "mean":
        return torch.mean(snr)
    elif reduction == "sum":
        return torch.sum(snr)
    elif reduction == "none":
        return snr
    else:
        raise ValueError(f"Unsupported reduction method.")


def torch_kl_divergence(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    reduction: str = "mean",
    flatten_start_dim=1,
) -> torch.Tensor:
    """
    Compute Kullback-Leibler divergence between y_pred(tensor) and y_real(tensor)

    KL divergence can be calculated as following equation:

        KL(y_pred || y_real) = sum(y_pred * log(y_pred / y_real))

    if x and y are matrices, KL divergence over matrix should be the mean value of KL divergence over all elements.

        KL(X, Y) = mean(X * log(X / Y))

    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        torch.Tensor: _description_
    """
    if y_pred.shape != y_real.shape:
        raise ValueError(
            f"Can not compute kl divergence for tensors with different shape. "
            f"({y_pred.shape} and {y_real.shape})"
        )
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=flatten_start_dim)
    y_real = y_real.flatten(start_dim=flatten_start_dim)

    # kl = torch.sum(y_pred * torch.log(y_pred / (y_real + 1e-7)), dim=-1)
    kl = torch.nn.functional.kl_div(
        y_pred.log_softmax(dim=-1), y_real.softmax(dim=-1), reduction="none"
    )

    if reduction == "mean":
        return torch.mean(kl)
    elif reduction == "sum":
        return torch.sum(kl)
    elif reduction == "none":
        return kl
    else:
        raise ValueError(f"Unsupported reduction method.")


def torch_sqnr(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    reduction: str = "mean",
    flatten_start_dim=1,
) -> torch.Tensor:
    """
    Compute Signal to Quantization Noise Ratio (SQNR) in dB for each sample.

    SQNR = 20 * log10( ||x||_2 / ||x - y||_2 )

    Returns per-sample SQNR (or reduced value according to `reduction`).
    """
    if y_pred.shape != y_real.shape:
        raise ValueError(
            f"Can not compute sqnr for tensors with different shape. "
            f"({y_pred.shape} and {y_real.shape})"
        )
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=flatten_start_dim).float()
    y_real = y_real.flatten(start_dim=flatten_start_dim).float()

    # compute norms per sample
    signal_norm = torch.norm(y_real, p=2, dim=-1)
    noise_norm = torch.norm(y_pred - y_real, p=2, dim=-1)
    # numerical stability
    eps = 1e-12
    sqnr = 20.0 * torch.log10((signal_norm + eps) / (noise_norm + eps))

    if reduction == "mean":
        return torch.mean(sqnr)
    elif reduction == "sum":
        return torch.sum(sqnr)
    elif reduction == "none":
        return sqnr
    else:
        raise ValueError(f"Unsupported reduction method.")


def torch_sqnr_as_loss(y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Return a loss-like value (negative SQNR) so higher SQNR is lower loss."""
    return -torch_sqnr(y_pred=y_pred, y_real=y_real, reduction=reduction)


def numpy_cosine_similarity(x: ndarray, y: ndarray) -> ndarray:
    return dot(x, y) / (norm(x) * norm(y))


def torch_cosine_similarity_as_loss(
    y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    return 1 - torch_cosine_similarity(
        y_pred=y_pred, y_real=y_real, reduction=reduction
    )


class MeasureRecorder:
    """Helper class for collecting data."""

    def __init__(
        self, measurement: str = "cosine", reduce: str = "mean", flatten_start_dim=1
    ) -> None:
        self.num_of_elements = 0
        self.measure = 0
        if reduce not in {"mean", "max"}:
            raise ValueError(
                f"PPQ MeasureRecorder Only support reduce by mean or max, however {reduce} was given."
            )

        if str(measurement).lower() == "cosine":
            measure_fn = partial(
                torch_cosine_similarity,
                reduction=reduce,
                flatten_start_dim=flatten_start_dim,
            )
        elif str(measurement).lower() == "mse":
            measure_fn = partial(
                torch_mean_square_error,
                reduction=reduce,
                flatten_start_dim=flatten_start_dim,
            )
        elif str(measurement).lower() == "snr":
            measure_fn = partial(
                torch_snr_error, reduction=reduce, flatten_start_dim=flatten_start_dim
            )

        elif str(measurement).lower() == "sqnr":
            measure_fn = partial(
                torch_sqnr, reduction=reduce, flatten_start_dim=flatten_start_dim
            )
        elif str(measurement).lower() == "kl":
            measure_fn = partial(
                torch_kl_divergence,
                reduction=reduce,
                flatten_start_dim=flatten_start_dim,
            )
        else:
            raise ValueError(
                "Unsupported measurement detected. "
                    f"PPQ only support mse, snr, sqnr and cosine now, while {measurement} was given."
            )

        self.measure_fn = measure_fn
        self.reduce = reduce

    def update(self, y_pred: torch.Tensor, y_real: torch.Tensor):
        elements = y_pred.shape[0]
        if elements != y_real.shape[0]:
            raise Exception(
                "Can not update measurement, cause your input data do not share a same batchsize. "
                f"Shape of y_pred {y_pred.shape} - against shape of y_real {y_real.shape}"
            )
        result = self.measure_fn(y_pred=y_pred.float(), y_real=y_real.float()).item()

        if self.reduce == "mean":
            self.measure = self.measure * self.num_of_elements + result * elements
            self.num_of_elements += elements
            self.measure /= self.num_of_elements

        if self.reduce == "max":
            self.measure = max(self.measure, result)
            self.num_of_elements += elements

    def clear(self):
        self.num_of_elements = 0
        self.measure = 0


class MeasurePrinter:
    """Helper class for print top-k record."""

    def __init__(
        self,
        data: Dict[str, float],
        measure: str,
        label: str = "Layer",
        k: int = None,
        order: str = None,
        percentage: bool = False,
    ) -> None:
        if order not in {"large_to_small", "small_to_large", None}:
            raise ValueError(
                'Parameter "order" can only be "large_to_small" or "small_to_large"'
            )
        self.collection = [(name, value) for name, value in data.items()]
        if order is not None:
            self.collection = sorted(self.collection, key=lambda x: x[1])
            if order == "large_to_small":
                self.collection = self.collection[::-1]
        if k is not None:
            self.collection = self.collection[:k]

        if order is None:
            sorted_collection = sorted(self.collection, key=lambda x: x[1])
            largest_element, smallest_element = (
                sorted_collection[-1][1],
                sorted_collection[0][1],
            )
        elif order == "large_to_small":
            largest_element, smallest_element = (
                self.collection[0][1],
                self.collection[-1][1],
            )
        else:
            largest_element, smallest_element = (
                self.collection[-1][1],
                self.collection[0][1],
            )
        self.normalized_by = largest_element - smallest_element
        self.min = smallest_element

        max_name_length = len(label)
        for name, _ in self.collection:
            max_name_length = max(len(name), max_name_length)
        self.max_name_length = max_name_length
        self.measure_str = measure
        self.label = label
        self.percentage = percentage

    def print(self, max_blocks: int = 20):
        print(
            f'{self.label}{" " * (self.max_name_length - len(self.label))}  | {self.measure_str} '
        )
        for name, value in self.collection:
            normalized_value = (value - self.min) / (self.normalized_by + 1e-7)
            if math.isnan(value):
                print("MeasurePrinter found an NaN value in your data.")
                normalized_value = 0
            num_of_blocks = round(normalized_value * max_blocks)

            if not self.percentage:
                print(
                    f'{name}:{" " * (self.max_name_length - len(name))} | '
                    f'{"█" * num_of_blocks}{" " * (max_blocks - num_of_blocks)} | '
                    f"{value:.4f}"
                )
            else:
                print(
                    f'{name}:{" " * (self.max_name_length - len(name))} | '
                    f'{"█" * num_of_blocks}{" " * (max_blocks - num_of_blocks)} | '
                    f"{value * 100:.3f}%"
                )
            if value == 0.0:
                print()
