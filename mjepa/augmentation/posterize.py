import timeit
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.utils import make_grid


@torch.compile
def _posterize_kernel(
    input: Tensor,
    posterize_mask: Tensor,
    levels: float,
) -> Tensor:
    """Apply posterize operation per batch element.

    Args:
        input: Input tensor of shape (B, ...)
        posterize_mask: Boolean mask of shape (B,) indicating which samples to posterize
        levels: Number of levels (2^bits - 1)

    Returns:
        Output tensor with same shape as input
    """
    batch_size = input.shape[0]
    # Reshape mask to broadcast over spatial dimensions
    posterize_mask = posterize_mask.view(batch_size, *([1] * (input.ndim - 1)))

    # Quantize to specified number of levels
    posterized = (input * levels).round() / levels

    # Apply mask to select which samples to posterize
    output = torch.where(posterize_mask, posterized, input)

    return output


def posterize(
    input: Tensor,
    posterize_prob: float,
    bits: int,
    seed: int | None = None,
) -> Tensor:
    """Apply posterize operation to input tensor.

    Args:
        input: Input tensor of shape (B, ...)
        posterize_prob: Probability of posterizing each sample
        bits: Number of bits to posterize to (1-8)
        seed: Random seed for reproducibility

    Returns:
        Transformed tensor with same shape and dtype as input
    """
    if not (1 <= bits <= 8):
        raise ValueError(f"Number of bits must be between 1 and 8, got {bits}")

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())

    batch_size = input.shape[0]
    dtype = input.dtype
    device = input.device

    # Compute levels outside compiled kernel to avoid bit shift in dynamo
    levels = float((1 << bits) - 1)

    # Generate per-batch random mask
    gen = torch.Generator(device=device).manual_seed(seed)
    posterize_mask = torch.rand(batch_size, generator=gen, device=device) < posterize_prob

    # Apply transformation
    output = _posterize_kernel(input.float(), posterize_mask, levels)

    return output.to(dtype)


def posterize_(
    input: Tensor,
    posterize_prob: float,
    bits: int,
    seed: int | None = None,
) -> Tensor:
    """In-place version of posterize operation.

    Args:
        input: Input tensor of shape (B, ...)
        posterize_prob: Probability of posterizing each sample
        bits: Number of bits to posterize to (1-8)
        seed: Random seed for reproducibility

    Returns:
        Modified input tensor
    """
    result = posterize(input, posterize_prob, bits, seed)
    input.copy_(result)
    return input


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="posterize", description="Preview posterize applied to an image")
    parser.add_argument("image", type=Path, help="Path to the image to apply posterize to")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("-p", "--prob", type=float, default=0.5, help="Probability of applying posterize")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
    parser.add_argument("--bits", type=int, default=6, help="Number of bits to posterize to")
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
    output = posterize(
        image,
        posterize_prob=args.prob,
        bits=args.bits,
        seed=args.seed,
    )
    torch.cuda.synchronize()
    end = timeit.default_timer()
    print(f"Time taken: {end - start} seconds")
    output = output.cpu()
    output = output.mul_(255).clamp_(0, 255).to(torch.uint8)
    grid = make_grid(output)
    grid = grid.permute(1, 2, 0)
    grid = Image.fromarray(grid.numpy())
    grid.save("posterize_preview.png")


if __name__ == "__main__":
    main(parse_args())
