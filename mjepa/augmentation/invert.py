from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import load


def _get_cuda_source_path() -> str:
    """Get path to CUDA source file, handling both development and installed package scenarios."""
    # Try development path first (when running from source)
    dev_path = Path(__file__).parents[2] / "csrc" / "invert.cu"
    if dev_path.exists():
        return str(dev_path)
    
    # Try installed package path
    pkg_path = Path(__file__).parent.parent / "csrc" / "invert.cu"
    if pkg_path.exists():
        return str(pkg_path)
    
    raise FileNotFoundError(f"Could not find invert.cu in expected locations: {dev_path}, {pkg_path}")


try:
    import invert_cuda  # type: ignore

    _invert_cuda = invert_cuda
except ImportError:
    if torch.cuda.is_available():
        _invert_cuda = load(
            name="invert_cuda",
            sources=[_get_cuda_source_path()],
            extra_cuda_cflags=["-O3"],
        )
    else:
        _invert_cuda = None


def invert(
    input: Tensor,
    invert_prob: float,
    solarize_prob: float = 0.0,
    solarize_threshold: float = 0.5,
    seed: int | None = None,
) -> Tensor:
    if _invert_cuda is None:
        raise RuntimeError("Invert is not available on this system")

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    return _invert_cuda.invert(input, invert_prob, solarize_prob, solarize_threshold, seed)


def invert_(
    input: Tensor,
    invert_prob: float,
    solarize_prob: float = 0.0,
    solarize_threshold: float = 0.5,
    seed: int | None = None,
) -> Tensor:
    if _invert_cuda is None:
        raise RuntimeError("Invert is not available on this system")

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    return _invert_cuda.invert_(input, invert_prob, solarize_prob, solarize_threshold, seed)
