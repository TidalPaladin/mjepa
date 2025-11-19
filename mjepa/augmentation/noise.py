import timeit
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Final, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.utils import make_grid


UNIFORM_NOISE_MIN: Final = -0.1
UNIFORM_NOISE_MAX: Final = 0.1
MULTIPLICATIVE_NOISE_MIN: Final = 0.8
MULTIPLICATIVE_NOISE_MAX: Final = 1.2
SALT_PEPPER_NOISE_PROB: Final = 0.1
SALT_PEPPER_NOISE_MIN: Final = 0.01
SALT_PEPPER_NOISE_MAX: Final = 0.05
DEFAULT_NOISE_PROB: Final = 0.1


@torch.compile
def _fused_noise_kernel(
    x: Tensor,
    uniform_mask: Tensor,
    mult_mask: Tensor,
    sp_mask: Tensor,
    uniform_noise: Tensor,
    mult_noise: Tensor,
    sp_prob: Tensor,
    sp_value: Tensor,
    sp_trigger: Tensor,
    clip: bool,
) -> Tensor:
    """Apply fused noise operations.

    Args:
        x: Input tensor (B, ...)
        uniform_mask: Per-batch mask for uniform noise (B,)
        mult_mask: Per-batch mask for multiplicative noise (B,)
        sp_mask: Per-batch mask for salt & pepper noise (B,)
        uniform_noise: Per-pixel uniform noise values (B, ...)
        mult_noise: Per-pixel multiplicative noise values (B, ...)
        sp_prob: Per-pixel salt & pepper probabilities (B, ...)
        sp_value: Per-pixel salt & pepper values (B, ...)
        sp_trigger: Per-pixel salt & pepper triggers (B, ...)
        clip: Whether to clip output to [0, 1]

    Returns:
        Output tensor with noise applied
    """
    # Reshape batch masks for broadcasting
    batch_size = x.shape[0]
    uniform_mask = uniform_mask.view(batch_size, *([1] * (x.ndim - 1)))
    mult_mask = mult_mask.view(batch_size, *([1] * (x.ndim - 1)))
    sp_mask = sp_mask.view(batch_size, *([1] * (x.ndim - 1)))

    # Compute multiplicative factor (will be 1.0 if not applied)
    mult_factor = torch.where(mult_mask, mult_noise, torch.ones_like(mult_noise))

    # Compute additive factor (will be 0.0 if not applied)
    add_factor = torch.where(uniform_mask, uniform_noise, torch.zeros_like(uniform_noise))

    # Apply multiplicative and additive noise
    output = x * mult_factor + add_factor

    # Apply salt & pepper noise
    sp_blend = sp_mask & (sp_trigger < sp_prob)
    output = torch.where(sp_blend, sp_value, output)

    # Clip if requested
    if clip:
        output = output.clamp(0.0, 1.0)

    return output


@torch.no_grad()
def apply_noise_batched(
    x: Tensor,
    prob: float = DEFAULT_NOISE_PROB,
    uniform_scale: float | Tuple[float, float] = (UNIFORM_NOISE_MIN, UNIFORM_NOISE_MAX),
    multiplicative_scale: float | Tuple[float, float] = (MULTIPLICATIVE_NOISE_MIN, MULTIPLICATIVE_NOISE_MAX),
    salt_pepper_prob: float = SALT_PEPPER_NOISE_PROB,
    salt_pepper_pixel_prob: float | Tuple[float, float] = (SALT_PEPPER_NOISE_MIN, SALT_PEPPER_NOISE_MAX),
    clip: bool = True,
    seed: int | None = None,
    inplace: bool = False,
) -> Tensor:
    """Apply noise to a batch of images using a fused kernel.

    Args:
        x: Input tensor
        prob: Probability of applying noise to the input
        uniform_scale: Scale of the uniform noise to apply to the input
        multiplicative_scale: Scale of the multiplicative noise to apply to the input
        salt_pepper_prob: Proportion of salt and pepper noise to apply to the input
        salt_pepper_pixel_prob: Proportion of salt and pepper noise to apply to each pixel
        clip: Whether to clip the result to the range :math:`[0, 1]`
        seed: Random seed for noise generation
        inplace: Whether to modify the input tensor in place

    Shape:
        - Input: :math:`(N, ...)`
        - Output: Same shape as input

    Returns:
        Input with noise applied
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    if isinstance(uniform_scale, (int, float)):
        uniform_scale = (uniform_scale, uniform_scale)
    if isinstance(multiplicative_scale, (int, float)):
        multiplicative_scale = (multiplicative_scale, multiplicative_scale)
    if isinstance(salt_pepper_pixel_prob, (int, float)):
        salt_pepper_pixel_prob = (salt_pepper_pixel_prob, salt_pepper_pixel_prob)

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())

    batch_size = x.shape[0]
    device = x.device
    dtype = x.dtype

    # Generate per-batch masks
    batch_gen = torch.Generator(device=device).manual_seed(seed)
    uniform_mask = torch.rand(batch_size, generator=batch_gen, device=device) < prob
    mult_mask = torch.rand(batch_size, generator=batch_gen, device=device) < prob
    sp_mask = torch.rand(batch_size, generator=batch_gen, device=device) < salt_pepper_prob

    # Generate per-pixel random values
    # Use a different generator to match the CUDA kernel's per-pixel RNG behavior
    pixel_gen = torch.Generator(device=device).manual_seed(seed + 1000000)

    # Uniform noise: sample range parameters per pixel, then sample noise value
    unif_center = (uniform_scale[0] + uniform_scale[1]) / 2.0
    unif_min_range = uniform_scale[0] + (unif_center - uniform_scale[0]) * torch.rand(
        x.shape, generator=pixel_gen, device=device, dtype=x.dtype
    )
    unif_max_range = unif_center + (uniform_scale[1] - unif_center) * torch.rand(
        x.shape, generator=pixel_gen, device=device, dtype=x.dtype
    )
    uniform_noise = unif_min_range + (unif_max_range - unif_min_range) * torch.rand(
        x.shape, generator=pixel_gen, device=device, dtype=x.dtype
    )

    # Multiplicative noise: sample range parameters per pixel, then sample noise value
    mult_center = (multiplicative_scale[0] + multiplicative_scale[1]) / 2.0
    mult_min_range = multiplicative_scale[0] + (mult_center - multiplicative_scale[0]) * torch.rand(
        x.shape, generator=pixel_gen, device=device, dtype=x.dtype
    )
    mult_max_range = mult_center + (multiplicative_scale[1] - mult_center) * torch.rand(
        x.shape, generator=pixel_gen, device=device, dtype=x.dtype
    )
    mult_noise = mult_min_range + (mult_max_range - mult_min_range) * torch.rand(
        x.shape, generator=pixel_gen, device=device, dtype=x.dtype
    )

    # Salt & pepper noise: sample probability per pixel, trigger, and value
    sp_prob = salt_pepper_pixel_prob[0] + (salt_pepper_pixel_prob[1] - salt_pepper_pixel_prob[0]) * torch.rand(
        x.shape, generator=pixel_gen, device=device, dtype=x.dtype
    )
    sp_trigger = torch.rand(x.shape, generator=pixel_gen, device=device, dtype=x.dtype)
    sp_value = (torch.rand(x.shape, generator=pixel_gen, device=device, dtype=x.dtype) < 0.5).float()

    # Apply fused noise kernel
    output = _fused_noise_kernel(
        x.float(),
        uniform_mask,
        mult_mask,
        sp_mask,
        uniform_noise,
        mult_noise,
        sp_prob,
        sp_value,
        sp_trigger,
        clip,
    )

    output = output.to(dtype)

    if inplace:
        x.copy_(output)
        return x
    else:
        return output


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="noise", description="Preview noise applied to an image")
    parser.add_argument("image", type=Path, help="Path to the image to apply noise to")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("-p", "--prob", type=float, default=0.5, help="Probability of applying noise")
    parser.add_argument("-i", "--inplace", default=False, action="store_true", help="Run the fused kernel inplace")
    parser.add_argument(
        "-u",
        "--uniform-scale",
        type=float,
        nargs=2,
        default=(UNIFORM_NOISE_MIN, UNIFORM_NOISE_MAX),
        help="Scale of the uniform noise",
    )
    parser.add_argument(
        "-m",
        "--multiplicative-scale",
        type=float,
        nargs=2,
        default=(MULTIPLICATIVE_NOISE_MIN, MULTIPLICATIVE_NOISE_MAX),
        help="Scale of the multiplicative noise",
    )
    parser.add_argument(
        "-s",
        "--salt-pepper-prob",
        type=float,
        default=SALT_PEPPER_NOISE_PROB,
        help="Probability of applying salt and pepper noise",
    )
    parser.add_argument(
        "-sp",
        "--salt-pepper-pixel-prob",
        type=float,
        nargs=2,
        default=(SALT_PEPPER_NOISE_MIN, SALT_PEPPER_NOISE_MAX),
        help="Probability of applying salt and pepper noise to a given pixel",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
    return parser.parse_args()


def main(args: Namespace):
    image = Image.open(args.image)
    image = np.array(image)
    image = torch.from_numpy(image)
    image = image.to(torch.float32) / torch.iinfo(image.dtype).max
    if image.ndim == 2:
        image.unsqueeze_(-1)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze_(0).expand(args.batch_size, -1, -1, -1)

    torch.manual_seed(args.seed)
    image = image.cuda()

    torch.cuda.synchronize()
    start = timeit.default_timer()
    noise = apply_noise_batched(
        image,
        prob=args.prob,
        uniform_scale=args.uniform_scale,
        multiplicative_scale=args.multiplicative_scale,
        salt_pepper_prob=args.salt_pepper_prob,
        salt_pepper_pixel_prob=args.salt_pepper_pixel_prob,
        clip=True,
        inplace=args.inplace,
    )
    torch.cuda.synchronize()
    end = timeit.default_timer()
    print(f"Time taken: {end - start} seconds")
    noise = noise.cpu()
    noise = noise.mul_(255).clamp_(0, 255).to(torch.uint8)
    grid = make_grid(noise)
    grid = grid.permute(1, 2, 0)
    grid = Image.fromarray(grid.numpy())
    grid.save("noise_preview.png")


if __name__ == "__main__":
    main(parse_args())
