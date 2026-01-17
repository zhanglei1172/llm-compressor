from typing import Optional

import torch

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.moving_base import MovingAverageObserverBase

__all__ = [
    "MemorylessPercentileObserver",
    "StaticPercentileObserver",
    "PercentileObserver",
]


@Observer.register("memoryless_percentile")
class MemorylessPercentileObserver(Observer):
    """
    Compute quantization parameters by taking the percentile values of the observed value.
    This helps exclude outliers and provides more robust quantization ranges.

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization
        percentile: percentile value to use for min/max calculation (default: 99.99)
            The min value will be at (100 - percentile) percentile
            The max value will be at percentile percentile
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.percentile = observer_kwargs.get("percentile", 99.99)

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_percentile_min_max(observed, self.percentile)

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_percentile_min_max(observed, self.percentile)


@Observer.register("static_percentile")
class StaticPercentileObserver(Observer):
    """
    Compute quantization parameters by taking the percentile values of all observed values.
    This helps exclude outliers and provides more robust quantization ranges.

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization
        percentile: percentile value to use for min/max calculation (default: 99.99)
            The min value will be at (100 - percentile) percentile
            The max value will be at percentile percentile
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.percentile = observer_kwargs.get("percentile", 99.99)
        self.past_min_vals = None
        self.past_max_vals = None
        self.past_global_min_vals = None
        self.past_global_max_vals = None

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        min_vals, max_vals = _get_percentile_min_max(observed, self.percentile)

        if self.past_min_vals is not None:
            min_vals = torch.min(min_vals, self.past_min_vals)
            max_vals = torch.max(max_vals, self.past_max_vals)

        self.past_min_vals = min_vals
        self.past_max_vals = max_vals

        return min_vals, max_vals

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        min_vals, max_vals = _get_percentile_min_max(observed, self.percentile)

        if self.past_global_min_vals is not None:
            min_vals = torch.min(min_vals, self.past_global_min_vals)
            max_vals = torch.max(max_vals, self.past_global_max_vals)

        self.past_global_min_vals = min_vals
        self.past_global_max_vals = max_vals

        return min_vals, max_vals


@Observer.register("percentile")
class PercentileObserver(MovingAverageObserverBase):
    """
    Compute quantization parameters by taking the moving average of percentile values.
    This helps exclude outliers and provides more robust quantization ranges.

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization
        percentile: percentile value to use for min/max calculation (default: 99.99)
            The min value will be at (100 - percentile) percentile
            The max value will be at percentile percentile
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.percentile = observer_kwargs.get("percentile", 99.99)

    def get_current_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_percentile_min_max(observed, self.percentile)

    def get_current_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_percentile_min_max(observed, self.percentile)


def _get_percentile_min_max(observed: torch.Tensor, percentile: float) -> MinMaxTuple:
    """
    Calculate min and max values using percentile-based approach.

    :param observed: value of shape (num_observations, *qparams_shape, group_size)
    :param percentile: percentile value to use (e.g., 99.99 means use 99.99th percentile
        for max and 0.01th percentile for min)
    :return: minimum value and maximum value whose shapes are (*qparams_shape, )
    """
    # Calculate percentile values
    q_high = percentile / 100.0
    q_low = 1.0 - q_high

    # Get the shape information
    qparams_shape = observed.shape[1:-1]
    num_observations = observed.shape[0]
    group_size = observed.shape[-1]

    # Flatten the observation dimension
    # observed_flat = observed.flatten(0, -2)  # (num_observations * qparams_shape, group_size)

    # Compute min and max along the group dimension
    min_vals = torch.amin(observed, dim=-1)  # (num_observations * qparams_shape,)
    max_vals = torch.amax(observed, dim=-1)  # (num_observations * qparams_shape,)

    # Reshape to separate observations and qparams
    if len(qparams_shape) > 0:
        min_vals = min_vals.view(num_observations, *qparams_shape)
        max_vals = max_vals.view(num_observations, *qparams_shape)
    else:
        min_vals = min_vals.view(num_observations, 1)
        max_vals = max_vals.view(num_observations, 1)

    # Compute percentile across observations
    # Move to CPU for quantile computation
    min_vals = min_vals.float()
    max_vals = max_vals.float()

    min_val = torch.quantile(min_vals, q_low)
    max_val = torch.quantile(max_vals, q_high)

    # Reshape to match the expected output shape (*qparams_shape,)
    max_val = max_val.reshape(qparams_shape).to(device=observed.device, dtype=observed.dtype)
    min_val = min_val.reshape(qparams_shape).to(device=observed.device, dtype=observed.dtype)

    return min_val, max_val
