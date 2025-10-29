import timeit
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.autograd import Function
from torchvision.utils import make_grid


def _get_cuda_source_path() -> str:
    """Get path to CUDA source file, handling both development and installed package scenarios."""
    # Try development path first (when running from source)
    dev_path = Path(__file__).parents[2] / "csrc" / "mixup.cu"
    if dev_path.exists():
        return str(dev_path)
    
    # Try installed package path
    pkg_path = Path(__file__).parent.parent / "csrc" / "mixup.cu"
    if pkg_path.exists():
        return str(pkg_path)
    
    raise FileNotFoundError(f"Could not find mixup.cu in expected locations: {dev_path}, {pkg_path}")


try:
    import mixup_cuda  # type: ignore
    _mixup_cuda = mixup_cuda
except ImportError:
    if torch.cuda.is_available():
        try:
            from torch.utils.cpp_extension import load
            _mixup_cuda = load(
                name="mixup_cuda",
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
            print(f"Warning: Failed to compile mixup CUDA extension: {e}")
            _mixup_cuda = None
    else:
        _mixup_cuda = None


@torch.no_grad()
def get_weights(
    batch_size: int,
    mixup_prob: float = 0.2,
    mixup_alpha: float = 1.0,
    seed: int | None = None,
    device: torch.device = torch.device("cuda"),
) -> Tensor:
    if _mixup_cuda is None:
        raise RuntimeError("MixUp is not available on this system")

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    return _mixup_cuda.get_weights(batch_size, mixup_prob, mixup_alpha, seed, device)


@torch.no_grad()
def is_mixed(
    batch_size: int,
    mixup_prob: float = 0.2,
    mixup_alpha: float = 1.0,
    seed: int | None = None,
    device: torch.device = torch.device("cuda"),
) -> Tensor:
    if _mixup_cuda is None:
        raise RuntimeError("MixUp is not available on this system")

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    return _mixup_cuda.is_mixed(batch_size, mixup_prob, mixup_alpha, seed, device)


@torch.no_grad()
def mixup(x: Tensor, mixup_prob: float = 0.2, mixup_alpha: float = 1.0, seed: int | None = None) -> Tensor:
    r"""Apply MixUp to an input tensor using a given weight.

    The input tensor is rolled along the first dimension and linearly interpolated
    with the original input using the provided weight.

    Args:
        x: The input tensor.
        mixup_prob: The probability of applying mixup to the input and target.
        mixup_alpha: The alpha parameter for the Beta distribution used to sample the mixup weight.
        seed: The seed for the random number generator.

    Returns:
        ``x.lerp(x.roll(1, 0), weight)``
    """
    if _mixup_cuda is None:
        raise RuntimeError("MixUp is not available on this system")

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    return _mixup_cuda.mixup(x, mixup_prob, mixup_alpha, seed)


class CrossEntropyMixup(Function):
    @staticmethod
    def forward(ctx, logits: Tensor, labels: Tensor, mixup_prob: float, mixup_alpha: float, seed: int) -> Tensor:
        assert _mixup_cuda is not None
        ctx.mixup_prob = mixup_prob
        ctx.mixup_alpha = mixup_alpha
        ctx.seed = seed
        loss, denom, max_val = _mixup_cuda.cross_entropy_mixup_fwd(logits, labels, mixup_prob, mixup_alpha, seed)
        ctx.save_for_backward(logits, labels, denom, max_val)
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None]:
        assert _mixup_cuda is not None
        logits, labels, denom, max_val = ctx.saved_tensors
        mixup_prob = ctx.mixup_prob
        mixup_alpha = ctx.mixup_alpha
        seed = ctx.seed
        grad = _mixup_cuda.cross_entropy_mixup_bwd(
            logits, labels, denom, max_val, grad_output, mixup_prob, mixup_alpha, seed
        )
        return grad, None, None, None, None


def cross_entropy_mixup(
    logits: Tensor, labels: Tensor, seed: int, mixup_prob: float = 0.2, mixup_alpha: float = 1.0
) -> Tensor:
    """Cross entropy loss with MixUp.

    Applies MixUp to the target labels by mixing them with a shifted version of the batch.
    The mixing weight is sampled from a Beta distribution. If a label is -1 (unknown),
    that sample is excluded from the loss calculation.

    This implementation avoids materializing the one-hot encoded labels, instead using
    a single kernel with online softmax to compute the loss.

    Args:
        logits: The predicted class logits
        labels: The target class labels
        seed: Random seed for reproducibility. Should match the seed used when applying MixUp
            to the input.
        mixup_prob: Probability of applying mixup to each sample
        mixup_alpha: Alpha parameter for Beta distribution used to sample mixup weight

    Returns:
        The cross entropy loss for each sample in the batch. Samples with unknown
        labels (-1) will have a loss value of -1.

    Shapes:
        - logits: :math:`(N, C)`
        - labels: :math:`(N,)`
        - Output: :math:`(N,)`
    """
    if _mixup_cuda is None:
        raise RuntimeError("MixUp is not available on this system")
    return CrossEntropyMixup.apply(logits, labels, mixup_prob, mixup_alpha, seed)


class BCEMixup(Function):
    @staticmethod
    def forward(
        ctx,
        logits: Tensor,
        labels: Tensor,
        mixup_prob: float,
        mixup_alpha: float,
        seed: int,
        pos_weight: float | None = None,
    ) -> Tensor:
        assert _mixup_cuda is not None
        ctx.mixup_prob = mixup_prob
        ctx.mixup_alpha = mixup_alpha
        ctx.seed = seed
        ctx.pos_weight = -1.0 if pos_weight is None else pos_weight
        loss = _mixup_cuda.bce_mixup_fwd(logits, labels, mixup_prob, mixup_alpha, ctx.pos_weight, seed)
        ctx.save_for_backward(logits, labels)
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None, None]:
        assert _mixup_cuda is not None
        logits, labels = ctx.saved_tensors
        mixup_prob = ctx.mixup_prob
        mixup_alpha = ctx.mixup_alpha
        seed = ctx.seed
        pos_weight = ctx.pos_weight
        grad = _mixup_cuda.bce_mixup_bwd(logits, labels, grad_output, mixup_prob, mixup_alpha, pos_weight, seed)
        return grad, None, None, None, None, None


def bce_mixup(
    logits: Tensor,
    labels: Tensor,
    seed: int,
    mixup_prob: float = 0.2,
    mixup_alpha: float = 1.0,
    pos_weight: float | None = None,
) -> Tensor:
    """BCE loss with MixUp.

    Applies MixUp to the target labels by mixing them with a shifted version of the batch.
    The mixing weight is sampled from a Beta distribution. If a label is -1 (unknown),
    that sample is excluded from the loss calculation.

    This implementation avoids materializing the one-hot encoded labels, instead using
    a single kernel with online softmax to compute the loss.

    Args:
        logits: The predicted class logits
        labels: The target class labels
        seed: Random seed for reproducibility. Should match the seed used when applying MixUp
            to the input.
        mixup_prob: Probability of applying mixup to each sample
        mixup_alpha: Alpha parameter for Beta distribution used to sample mixup weight
        pos_weight: Optional weight for positive examples in range [0, 1]. If provided, positive
            examples are weighted by pos_weight and negative examples by (1 - pos_weight).

    Returns:
        The cross entropy loss for each sample in the batch. Samples with unknown
        labels (-1) will have a loss value of -1.

    Shapes:
        - logits: :math:`(N, ...)`
        - labels: :math:`(N, ...)`
        - Output: :math:`(N, ...)`
    """
    if _mixup_cuda is None:
        raise RuntimeError("MixUp is not available on this system")
    if pos_weight is not None and not (0 <= pos_weight <= 1):
        raise ValueError("pos_weight must be in range [0, 1]")
    return BCEMixup.apply(logits, labels, mixup_prob, mixup_alpha, seed, pos_weight)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to the image to apply mixup to")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-p", "--prob", type=float, default=0.2, help="Probability of applying mixup")
    parser.add_argument("-a", "--alpha", type=float, default=1.0, help="Alpha parameter for the Beta distribution")
    return parser.parse_args()


def main(args: Namespace):
    image = Image.open(args.image)
    image = np.array(image)
    image = torch.from_numpy(image)
    image = image.to(torch.float32) / torch.iinfo(image.dtype).max
    if image.ndim == 2:
        image.unsqueeze_(-1)
    image = image.permute(2, 0, 1)

    # Create a length-2 batch by flipping the image horizontally
    image = torch.stack([image, image.flip(2)], dim=0)
    image = image.repeat(args.batch_size, 1, 1, 1)

    torch.manual_seed(args.seed)
    image = image.cuda()

    torch.random.manual_seed(args.seed)
    torch.cuda.synchronize()
    start = timeit.default_timer()
    out = mixup(image, args.prob, args.alpha, args.seed)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    print(f"Time taken: {end - start} seconds")
    out = out.cpu()
    out = out.mul_(255).clamp_(0, 255).to(torch.uint8)
    grid = make_grid(out)
    grid = grid.permute(1, 2, 0)
    grid = Image.fromarray(grid.numpy())
    grid.save("mixup_preview.png")


if __name__ == "__main__":
    main(parse_args())
