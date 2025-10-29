import timeit
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Final, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.cpp_extension import load
from torchvision.utils import make_grid


UNIFORM_NOISE_MIN: Final = -0.1
UNIFORM_NOISE_MAX: Final = 0.1
MULTIPLICATIVE_NOISE_MIN: Final = 0.8
MULTIPLICATIVE_NOISE_MAX: Final = 1.2
SALT_PEPPER_NOISE_PROB: Final = 0.1
SALT_PEPPER_NOISE_MIN: Final = 0.01
SALT_PEPPER_NOISE_MAX: Final = 0.05
DEFAULT_NOISE_PROB: Final = 0.1

def _get_cuda_source_path() -> str:
    """Get path to CUDA source file, handling both development and installed package scenarios."""
    # Try development path first (when running from source)
    dev_path = Path(__file__).parents[2] / "csrc" / "noise.cu"
    if dev_path.exists():
        return str(dev_path)
    
    # Try installed package path
    pkg_path = Path(__file__).parent.parent / "csrc" / "noise.cu"
    if pkg_path.exists():
        return str(pkg_path)
    
    raise FileNotFoundError(f"Could not find noise.cu in expected locations: {dev_path}, {pkg_path}")


try:
    import noise_cuda  # type: ignore

    _noise_cuda = noise_cuda
except ImportError:
    if torch.cuda.is_available():
        _noise_cuda = load(
            name="noise_cuda",
            sources=[_get_cuda_source_path()],
            extra_cuda_cflags=["-O3"],
        )
    else:
        _noise_cuda = None


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
    """Apply noise to a batch of images using a fused CUDA kernel.

    This fused kernel is substantially faster than the unfused equivalent (~10x or more in testing).

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
    if _noise_cuda is None:
        raise ValueError("CUDA is not available")
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

    return _noise_cuda.fused_noise(
        x,
        float(uniform_scale[0]),
        float(uniform_scale[1]),
        float(multiplicative_scale[0]),
        float(multiplicative_scale[1]),
        float(salt_pepper_pixel_prob[0]),
        float(salt_pepper_pixel_prob[1]),
        float(prob),
        float(prob),
        float(salt_pepper_prob),
        bool(clip),
        seed,
        inplace,
    )


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
