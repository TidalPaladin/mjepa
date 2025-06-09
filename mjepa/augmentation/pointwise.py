from dataclasses import dataclass
from typing import Any, Tuple, cast

import torch
import yaml
from torch import Tensor

from .invert import invert_
from .mixup import mixup
from .noise import (
    DEFAULT_NOISE_PROB,
    MULTIPLICATIVE_NOISE_MAX,
    MULTIPLICATIVE_NOISE_MIN,
    SALT_PEPPER_NOISE_MAX,
    SALT_PEPPER_NOISE_MIN,
    SALT_PEPPER_NOISE_PROB,
    UNIFORM_NOISE_MAX,
    UNIFORM_NOISE_MIN,
    apply_noise_batched,
)
from .posterize import posterize_


@dataclass
class AugmentationConfig:
    """
    Configuration for augmentation related hyperparameters.

    Args:
        mixup_alpha: Alpha parameter for the Beta distribution used to sample the mixup weight.
        mixup_prob: Probability of applying mixup to the input and target.
        use_noise: If True, apply noise to the input.
        uniform_noise_scale: Scale of the uniform noise to apply to the input.
        multiplicative_noise_scale: Scale of the multiplicative noise to apply to the input.
        salt_pepper_prob: Proportion of salt and pepper noise to apply to the input.
        salt_pepper_pixel_prob: Probability of applying salt and pepper noise to a given pixel.
        noise_prob: Probability of applying a given noise transform.
        noise_clip: If True, clip the noise to the range [0, 1].
        invert_prob: Probability of inverting the input.
        solarize_prob: Probability of solarizing the input.
        solarize_threshold: Threshold for solarizing the input.
        posterize_prob: Probability of posterizing the input.
        posterize_bits: Number of bits to posterize the input to.
    """

    mixup_alpha: float = 1.0
    mixup_prob: float = 0.2
    use_noise: bool = True
    uniform_noise_scale: float | Tuple[float, float] = (UNIFORM_NOISE_MIN, UNIFORM_NOISE_MAX)
    multiplicative_noise_scale: float | Tuple[float, float] = (MULTIPLICATIVE_NOISE_MIN, MULTIPLICATIVE_NOISE_MAX)
    salt_pepper_prob: float = SALT_PEPPER_NOISE_PROB
    salt_pepper_pixel_prob: float | Tuple[float, float] = (SALT_PEPPER_NOISE_MIN, SALT_PEPPER_NOISE_MAX)
    noise_prob: float = DEFAULT_NOISE_PROB
    noise_clip: bool = True
    invert_prob: float = 0.0
    solarize_prob: float = 0.0
    solarize_threshold: float = 0.5
    posterize_prob: float = 0.0
    posterize_bits: int = 6

    def __post_init__(self) -> None:
        if not 0 < self.mixup_alpha:
            raise ValueError("mixup_alpha must be positive")
        if not 0 <= self.mixup_prob <= 1:
            raise ValueError("mixup_prob must be in the range [0, 1]")
        if not 0 <= self.invert_prob <= 1:
            raise ValueError("invert_prob must be in the range [0, 1]")
        if not 0 <= self.solarize_prob <= 1:
            raise ValueError("solarize_prob must be in the range [0, 1]")
        if not 0 <= self.posterize_prob <= 1:
            raise ValueError("posterize_prob must be in the range [0, 1]")
        if self.posterize_bits < 1 or self.posterize_bits > 8:
            raise ValueError("posterize_bits must be in the range [1, 8]")


@torch.no_grad()
def apply_noise(augmentation_config: AugmentationConfig, x: Tensor) -> Tensor:
    torch.cuda.nvtx.range_push("noise")
    if augmentation_config.use_noise and x.device.type == "cuda":
        x = apply_noise_batched(
            x,
            prob=augmentation_config.noise_prob,
            uniform_scale=augmentation_config.uniform_noise_scale,
            multiplicative_scale=augmentation_config.multiplicative_noise_scale,
            salt_pepper_prob=augmentation_config.salt_pepper_prob,
            salt_pepper_pixel_prob=augmentation_config.salt_pepper_pixel_prob,
            clip=augmentation_config.noise_clip,
        )
    torch.cuda.nvtx.range_pop()
    return x


@torch.no_grad()
def apply_mixup(
    augmentation_config: AugmentationConfig, *tensors: Tensor, seed: int | None = None
) -> Tuple[int | None, Tuple[Tensor, ...]]:
    if augmentation_config.mixup_prob > 0 and tensors[0].device.type == "cuda":
        torch.cuda.nvtx.range_push("mixup")
        mixup_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if seed is None else seed
        tensors = cast(
            Any,
            [mixup(t, augmentation_config.mixup_prob, augmentation_config.mixup_alpha, mixup_seed) for t in tensors],
        )
        torch.cuda.nvtx.range_pop()
    else:
        mixup_seed = 0
    return mixup_seed, tuple(tensors)


@torch.no_grad()
def apply_invert(
    augmentation_config: AugmentationConfig, *tensors: Tensor, seed: int | None = None
) -> Tuple[Tensor, ...]:
    if (
        augmentation_config.invert_prob > 0
        or augmentation_config.solarize_prob > 0
        and tensors[0].device.type == "cuda"
    ):
        torch.cuda.nvtx.range_push("invert")
        invert_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if seed is None else seed
        for t in tensors:
            invert_(
                t,
                augmentation_config.invert_prob,
                augmentation_config.solarize_prob,
                augmentation_config.solarize_threshold,
                invert_seed,
            )
        torch.cuda.nvtx.range_pop()
    return tensors


@torch.no_grad()
def apply_posterize(
    augmentation_config: AugmentationConfig, *tensors: Tensor, seed: int | None = None
) -> Tuple[Tensor, ...]:
    if augmentation_config.posterize_prob > 0 and tensors[0].device.type == "cuda":
        torch.cuda.nvtx.range_push("posterize")
        posterize_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if seed is None else seed
        for t in tensors:
            posterize_(
                t,
                augmentation_config.posterize_prob,
                augmentation_config.posterize_bits,
                posterize_seed,
            )
        torch.cuda.nvtx.range_pop()
    return tensors


def config_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    return AugmentationConfig(**values)


def register_constructors():
    tags = [
        "tag:yaml.org,2002:python/object:mjepa.AugmentationConfig",
    ]
    loaders = [yaml.SafeLoader, yaml.FullLoader, yaml.UnsafeLoader]
    for tag in tags:
        for loader in loaders:
            loader.add_constructor(tag, config_constructor)


register_constructors()
