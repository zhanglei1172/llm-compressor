import torch

from llmcompressor.observers.base import MinMaxTuple, Observer, ScaleZpTuple

__all__ = ["LCTObserver"]


@Observer.register("lct")
class LCTObserver(Observer):
    """
    Learnable Clipping Thresholds (LCT) Observer.

    This observer extends the basic min-max approach by introducing learnable
    clip factors that scale the min/max values through a sigmoid function:
        xmax = xmax * sigmoid(clip_factor_max)
        xmin = xmin * sigmoid(clip_factor_min)

    The clip factors are nn.Parameters that can be optimized during training
    to find optimal clipping thresholds for quantization.

    Args:
        init_clip_factor: Initial value for clip factors (default: 4.0).
            With init_value=4.0, sigmoid(4.0) ≈ 0.982, providing a slight
            initial clipping effect.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get init value from observer_kwargs, default to 4.0
        init_value = self.args.observer_kwargs.get("init_clip_factor", 4.0)

        # Learnable clip factors
        self.clip_factor_max = torch.nn.Parameter(
            torch.ones((1,)) * init_value, requires_grad=True
        )
        self.clip_factor_min = torch.nn.Parameter(
            torch.ones((1,)) * init_value, requires_grad=True
        )

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Compute min/max values with learnable clipping thresholds.

        Args:
            observed: Input tensor with shape (num_observations, *qparam_shape, group_size)

        Returns:
            Tuple of (min_vals, max_vals) with learnable clipping applied
        """
        # Compute base min/max
        min_vals = torch.amin(observed, dim=(0, -1))
        max_vals = torch.amax(observed, dim=(0, -1))

        # Apply learnable clipping via sigmoid
        # sigmoid ensures the factor is in (0, 1), providing controlled clipping
        max_vals = max_vals * torch.sigmoid(self.clip_factor_max)
        min_vals = min_vals * torch.sigmoid(self.clip_factor_min)

        return min_vals, max_vals

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Compute global min/max values with learnable clipping thresholds.

        Args:
            observed: Input tensor

        Returns:
            Tuple of (min_val, max_val) with learnable clipping applied
        """
        # Compute base global min/max
        min_val = torch.amin(observed, dim=(0, -1))
        max_val = torch.amax(observed, dim=(0, -1))

        # Apply learnable clipping via sigmoid
        max_val = max_val * torch.sigmoid(self.clip_factor_max)
        min_val = min_val * torch.sigmoid(self.clip_factor_min)

        return min_val, max_val

    def forward(self, observed: torch.Tensor) -> ScaleZpTuple:
        """
        Calculate updated scales and zero points from observed value
        (weight, activation, or attention state).

        :param observed: value being observed
        :return: calibrated scale and zero point
        """
        scales, zero_points, _min, _max = self._forward_with_minmax(observed)
        return (scales, zero_points)
