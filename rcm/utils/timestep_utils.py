import torch
from typing import Sequence, Tuple, Union


def shift_rf_time(u: torch.Tensor, shift: float) -> torch.Tensor:
    if shift <= 0:
        return u
    return shift * u / (1 + (shift - 1) * u)


def sigma_to_rf_time(sigma: torch.Tensor) -> torch.Tensor:
    return sigma / (sigma + 1)


def rf_to_sigma(rf_t: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(rf_t.dtype).eps
    rf_t = rf_t.clamp(min=0.0, max=1.0 - eps)
    return rf_t / (1 - rf_t)


def sigma_to_trig_time(sigma: torch.Tensor) -> torch.Tensor:
    return torch.arctan(sigma)


def trig_to_sigma(trig_t: torch.Tensor) -> torch.Tensor:
    return torch.tan(trig_t)


def rf_to_trig_time(rf_t: torch.Tensor) -> torch.Tensor:
    return sigma_to_trig_time(rf_to_sigma(rf_t))


def trig_to_rf_time(trig_t: torch.Tensor) -> torch.Tensor:
    return sigma_to_rf_time(trig_to_sigma(trig_t))


def _normalize_sample_shape(shape: Union[int, Sequence[int], torch.Size]) -> Tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    if isinstance(shape, torch.Size):
        return tuple(shape)
    if isinstance(shape, Sequence):
        return tuple(int(dim) for dim in shape)
    raise TypeError(f"Unsupported sample shape type: {type(shape)}")


class LogNormal:
    """Log-normal sampler returning RF-domain time."""

    output_domain = "rf"

    def __init__(self, p_mean: float = 0.0, p_std: float = 1.0, **kwargs):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(
        self, shape: Union[int, Sequence[int], torch.Size], device: Union[torch.device, str] = "cuda", dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        sample_shape = _normalize_sample_shape(shape)
        log_sigma = torch.randn(sample_shape, device=device).to(dtype) * self.p_std + self.p_mean
        sigma = torch.exp(log_sigma)
        return sigma_to_rf_time(sigma)


class UniformShift:
    """Uniform sampler on [0,1) shifted in RF domain."""

    output_domain = "rf"

    def __init__(self, shift: float = 0.0, **kwargs):
        self.shift = shift

    def __call__(
        self, shape: Union[int, Sequence[int], torch.Size], device: Union[torch.device, str] = "cuda", dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        sample_shape = _normalize_sample_shape(shape)
        u = torch.rand(sample_shape, device=device).to(dtype)
        return shift_rf_time(u, self.shift)
