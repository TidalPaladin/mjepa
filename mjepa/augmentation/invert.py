from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import load


try:
    import invert_cuda  # type: ignore

    _invert_cuda = invert_cuda
except ImportError:
    if torch.cuda.is_available():
        _invert_cuda = load(
            name="invert_cuda",
            sources=[str(Path(__file__).parents[2] / "csrc" / "invert.cu")],
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
