import timeit
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.utils import make_grid


def _get_cuda_source_path() -> str:
    """Get path to CUDA source file, handling both development and installed package scenarios."""
    # Try development path first (when running from source)
    dev_path = Path(__file__).parents[2] / "csrc" / "posterize.cu"
    if dev_path.exists():
        return str(dev_path)
    
    # Try installed package path
    pkg_path = Path(__file__).parent.parent / "csrc" / "posterize.cu"
    if pkg_path.exists():
        return str(pkg_path)
    
    raise FileNotFoundError(f"Could not find posterize.cu in expected locations: {dev_path}, {pkg_path}")


try:
    import posterize_cuda  # type: ignore
    _posterize_cuda = posterize_cuda
except ImportError:
    if torch.cuda.is_available():
        try:
            from torch.utils.cpp_extension import load
            _posterize_cuda = load(
                name="posterize_cuda",
                sources=[_get_cuda_source_path()],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
            )
        except Exception as e:
            print(f"Warning: Failed to compile posterize CUDA extension: {e}")
            _posterize_cuda = None
    else:
        _posterize_cuda = None


def posterize(
    input: Tensor,
    posterize_prob: float,
    bits: int,
    seed: int | None = None,
) -> Tensor:
    if _posterize_cuda is None:
        raise RuntimeError("Posterize is not available on this system")

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    return _posterize_cuda.posterize(input, posterize_prob, bits, seed)


def posterize_(
    input: Tensor,
    posterize_prob: float,
    bits: int,
    seed: int | None = None,
) -> Tensor:
    if _posterize_cuda is None:
        raise RuntimeError("Posterize is not available on this system")

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    return _posterize_cuda.posterize_(input, posterize_prob, bits, seed)


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
