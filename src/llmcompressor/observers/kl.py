from typing import Optional

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.utils import patch_attr

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.moving_base import MovingAverageObserverBase

__all__ = ["MemorylessKLObserver", "KLObserver"]


def _compute_histogram_gpu(
    tensor: torch.Tensor, num_bins: int, min_val: float, max_val: float
) -> torch.Tensor:
    """
    Compute histogram on GPU using torch.histc or manual implementation.

    This function provides GPU-accelerated histogram computation to avoid
    CPU-GPU data transfer overhead.

    :param tensor: Input tensor
    :param num_bins: Number of histogram bins
    :param min_val: Minimum value for histogram range
    :param max_val: Maximum value for histogram range
    :return: Histogram counts
    """
    device = tensor.device

    # Try torch.histc first (works on CUDA in older PyTorch versions)
    try:
        # torch.histc uses inclusive-exclusive bins [min, max)
        hist = torch.histc(
            tensor.float(),
            bins=num_bins,
            min=min_val,
            max=max_val,
        )
        return hist.to(device)
    except Exception:
        # Fallback to manual implementation using bucketize
        # Create bin edges
        bin_edges = torch.linspace(min_val, max_val, num_bins + 1, device=device)

        # Use bucketize to find which bin each element belongs to
        # bucketize returns indices in [0, num_bins] where num_bins is for out-of-range
        indices = torch.bucketize(tensor.float(), bin_edges, right=True) - 1

        # Clamp indices to valid range [0, num_bins-1]
        indices = torch.clamp(indices, 0, num_bins - 1)

        # Count occurrences in each bin
        hist = torch.zeros(num_bins, device=device, dtype=tensor.dtype)
        hist.scatter_add_(0, indices.long(), torch.ones_like(indices, dtype=tensor.dtype))

        return hist


@Observer.register("memoryless_kl")
class MemorylessKLObserver(Observer):
    """
    Compute quantization parameters by finding the optimal min/max values which minimize
    the KL divergence between the original and quantized distributions.

    This approach searches for the optimal quantization range that preserves the
    distribution characteristics of the original tensor.

    ```psuedocode
    kl_divergence := KL(P || Q)
    where P is the original distribution and Q is the quantized distribution
    scale, zp <- min[min_vals, max_vals](kl_divergence(x))
    ```

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization\n
        num_bins: number of bins for histogram calculation (default: 2048)\n
        num_quantized_bins: number of bins for quantized histogram (default: 128)\n
        maxshrink: maximum shrink amount (in "grid steps"). The number of
            search steps is int(maxshrink * grid)\n
        grid: resolution of the shrink search. Larger values give finer granularity
            in shrink factors (default: 100)\n
        global_scale: precomputed global scale to use for quantization. Ignored if
            `optimize_global_scale` is True\n
        optimize_global_scale: If True, recompute ``global_scale`` from the
            candidate min/max during each step of the search
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.num_bins = observer_kwargs.get("num_bins", 2048)
        self.num_quantized_bins = observer_kwargs.get("num_quantized_bins", 128)
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.grid = observer_kwargs.get("grid", 100)

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        global_scale = self._get_module_param("global_scale")
        return _grid_search_kl(
            observed,
            self.args,
            self.num_bins,
            self.num_quantized_bins,
            self.maxshrink,
            self.grid,
            global_scale=global_scale,
            optimize_global_scale=False,
        )

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _grid_search_kl(
            observed,
            self.args,
            self.num_bins,
            self.num_quantized_bins,
            self.maxshrink,
            self.grid,
            global_scale=None,
            optimize_global_scale=True,
        )


@Observer.register("kl")
class KLObserver(MovingAverageObserverBase):
    """
    Compute quantization parameters by finding the optimal min/max values which minimize
    the KL divergence between the original and quantized distributions.

    This approach searches for the optimal quantization range that preserves the
    distribution characteristics of the original tensor.

    ```psuedocode
    kl_divergence := KL(P || Q)
    where P is the original distribution and Q is the quantized distribution
    scale, zp <- min[min_vals, max_vals](kl_divergence(x))
    ```

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization\n
        num_bins: number of bins for histogram calculation (default: 2048)\n
        num_quantized_bins: number of bins for quantized histogram (default: 128)\n
        maxshrink: maximum shrink amount (in "grid steps"). The number of
            search steps is int(maxshrink * grid)\n
        grid: resolution of the shrink search. Larger values give finer granularity
            in shrink factors (default: 100)\n
        global_scale: precomputed global scale to use for quantization. Ignored if
            `optimize_global_scale` is True\n
        optimize_global_scale: If True, recompute ``global_scale`` from the
            candidate min/max during each step of the search
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.num_bins = observer_kwargs.get("num_bins", 2048)
        self.num_quantized_bins = observer_kwargs.get("num_quantized_bins", 128)
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.grid = observer_kwargs.get("grid", 100)

    def get_current_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        global_scale = self._get_module_param("global_scale")
        return _grid_search_kl(
            observed,
            self.args,
            self.num_bins,
            self.num_quantized_bins,
            self.maxshrink,
            self.grid,
            global_scale=global_scale,
            optimize_global_scale=False,
        )

    def get_current_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _grid_search_kl(
            observed,
            self.args,
            self.num_bins,
            self.num_quantized_bins,
            self.maxshrink,
            self.grid,
            global_scale=None,
            optimize_global_scale=True,
        )


def _grid_search_kl(
    observed: torch.Tensor,
    args: QuantizationArgs,
    num_bins: int,
    num_quantized_bins: int,
    maxshrink: float,
    grid: float,
    global_scale: Optional[torch.Tensor] = None,
    optimize_global_scale: bool = False,
) -> MinMaxTuple:
    """
    Perform a 1-D grid search to find per-channel min/max ranges that minimize
    KL divergence between original and quantized distributions.

    This routine progressively "shrinks" the absolute min/max ranges of the
    observed tensor and evaluates the KL divergence at each candidate range.

    :param observed: value of shape (num_observations, *qparams_shape, group_size)
    :param args: quantization args used for computing qparams and fake quant
    :param num_bins: number of bins for histogram calculation
    :param num_quantized_bins: number of bins for quantized histogram
    :param maxshrink: maximum shrink amount (in "grid steps"). The number of
        search steps is int(maxshrink * grid)
    :param grid: resolution of the shrink search. Larger values give finer granularity
        in shrink factors
    :param global_scale: precomputed global scale to use for quantization. Ignored if
        `optimize_global_scale` is True
    :param optimize_global_scale: If True, recompute ``global_scale`` from the
        candidate min/max during each step of the search
    """
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    best_kl = torch.full_like(min_val, torch.finfo(min_val.dtype).max)
    best_min_val = min_val.clone()
    best_max_val = max_val.clone()

    # Pre-compute the original histogram and distribution
    observed_flat = observed.flatten(0, -2).flatten()
    obs_min = observed_flat.min().item()
    obs_max = observed_flat.max().item()

    # Use GPU-accelerated histogram computation
    hist_original = _compute_histogram_gpu(observed_flat, num_bins, obs_min, obs_max)

    # Normalize to get probability distribution
    P = hist_original.float() + 1e-10
    P = P / P.sum()

    # Grid search over shrink factors
    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        shrinked_min_val = p * min_val
        shrinked_max_val = p * max_val

        if optimize_global_scale:
            global_scale = generate_gparam(shrinked_min_val, shrinked_max_val)

        candidate_scales, candidate_zero_points = calculate_qparams(
            min_vals=shrinked_min_val,
            max_vals=shrinked_max_val,
            quantization_args=args,
            global_scale=global_scale,
        )

        # Fake quantize the observed tensor
        with patch_attr(args, "strategy", QuantizationStrategy.TOKEN):
            q = fake_quantize(
                observed,
                candidate_scales.unsqueeze(-1),
                candidate_zero_points.unsqueeze(-1),
                args,
                global_scale=global_scale,
            ).to(observed.dtype)

        # Compute histogram of quantized tensor
        q_flat = q.flatten(0, -2).flatten()
        q_min = q_flat.min().item()
        q_max = q_flat.max().item()

        # Use GPU-accelerated histogram computation
        hist_quantized = _compute_histogram_gpu(q_flat, num_quantized_bins, q_min, q_max)

        # Normalize to get probability distribution
        Q = hist_quantized.float() + 1e-10
        Q = Q / Q.sum()

        # Compute KL divergence
        # Interpolate Q to match the number of bins in P
        if Q.shape[0] != P.shape[0]:
            Q = torch.nn.functional.interpolate(
                Q.unsqueeze(0).unsqueeze(0),
                size=P.shape[0],
                mode="linear",
                align_corners=False,
            ).squeeze()

        kl_div = (P * (torch.log(P) - torch.log(Q))).sum()

        # Update best min/max if this shrink factor gives lower KL divergence
        if kl_div < best_kl.item():
            best_kl = torch.full_like(best_kl, kl_div)
            best_min_val = shrinked_min_val.clone()
            best_max_val = shrinked_max_val.clone()

    return best_min_val, best_max_val