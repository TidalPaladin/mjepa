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


def _sample_beta(u: Tensor, v: Tensor, alpha: float) -> Tensor:
    """Sample from Beta distribution using the same method as CUDA kernel.

    Args:
        u: Uniform random values [0, 1]
        v: Uniform random values [0, 1]
        alpha: Alpha parameter for Beta distribution

    Returns:
        Samples from Beta(alpha, alpha)
    """
    divisor = 1.0 / alpha
    u_pow = u.pow(divisor)
    v_pow = v.pow(divisor)
    return u_pow / (u_pow + v_pow)


@torch.no_grad()
def get_weights(
    batch_size: int,
    mixup_prob: float = 0.2,
    mixup_alpha: float = 1.0,
    seed: int | None = None,
    device: torch.device = torch.device("cuda"),
) -> Tensor:
    """Get per-batch mixup weights.

    Args:
        batch_size: Number of samples in batch
        mixup_prob: Probability of applying mixup to each sample
        mixup_alpha: Alpha parameter for Beta distribution
        seed: Random seed
        device: Device to create tensor on

    Returns:
        Tensor of shape (batch_size,) with mixup weights
    """
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())

    # Generate per-batch random values
    gen = torch.Generator(device=device).manual_seed(seed)
    apply_mixup = torch.rand(batch_size, generator=gen, device=device) < mixup_prob

    # Sample Beta distribution
    u = torch.rand(batch_size, generator=gen, device=device)
    v = torch.rand(batch_size, generator=gen, device=device)
    weights = _sample_beta(u, v, mixup_alpha)

    # Set weight to 1.0 for samples that don't get mixup
    weights = torch.where(apply_mixup, weights, torch.ones_like(weights))

    return weights


@torch.no_grad()
def is_mixed(
    batch_size: int,
    mixup_prob: float = 0.2,
    mixup_alpha: float = 1.0,
    seed: int | None = None,
    device: torch.device = torch.device("cuda"),
) -> Tensor:
    """Get per-batch boolean mask of which samples are mixed.

    Args:
        batch_size: Number of samples in batch
        mixup_prob: Probability of applying mixup to each sample
        mixup_alpha: Alpha parameter for Beta distribution
        seed: Random seed
        device: Device to create tensor on

    Returns:
        Boolean tensor of shape (batch_size,)
    """
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())

    # Generate per-batch random values
    gen = torch.Generator(device=device).manual_seed(seed)
    apply_mixup = torch.rand(batch_size, generator=gen, device=device) < mixup_prob

    return apply_mixup


def _mixup_kernel(x: Tensor, weights: Tensor) -> Tensor:
    """Apply mixup operation.

    Args:
        x: Input tensor (B, ...)
        weights: Per-batch mixup weights (B,)

    Returns:
        Mixed tensor
    """
    batch_size = x.shape[0]
    weights = weights.view(batch_size, *([1] * (x.ndim - 1)))
    other = x.roll(-1, 0)
    return x * weights + other * (1 - weights)


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
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())

    batch_size = x.shape[0]
    device = x.device

    # Get mixup weights and ensure they match input dtype
    weights = get_weights(batch_size, mixup_prob, mixup_alpha, seed, device).type_as(x)

    # Apply mixup
    output = _mixup_kernel(x, weights)

    return output


@torch.compile
def _cross_entropy_mixup_fwd_kernel(
    logits: Tensor,
    labels: Tensor,
    weights: Tensor,
    apply_mixup: Tensor,
) -> Tensor:
    """Forward pass for cross-entropy with mixup.

    Args:
        logits: Logits tensor (B, C)
        labels: Label tensor (B,)
        weights: Mixup weights (B,)
        apply_mixup: Boolean mask (B,)

    Returns:
        Loss tensor (B,)
    """
    batch_size, num_classes = logits.shape
    logits.device

    # Handle unknown labels (-1)
    unknown_mask = labels == -1

    # Create one-hot labels
    valid_labels = torch.where(unknown_mask, torch.zeros_like(labels), labels)
    labels_one_hot = torch.nn.functional.one_hot(valid_labels, num_classes).float()

    # Mix labels with rolled version
    labels_one_hot_other = labels_one_hot.roll(-1, 0)
    weights_expanded = weights.view(batch_size, 1)
    mixed_labels = weights_expanded * labels_one_hot + (1 - weights_expanded) * labels_one_hot_other

    # Check if mixed label is unknown
    labels_other = labels.roll(-1, 0)
    unknown_other = labels_other == -1
    both_unknown = unknown_mask | (apply_mixup & unknown_other)

    # Compute cross-entropy loss
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    loss = -(mixed_labels * log_probs).sum(dim=-1)

    # Mark unknown samples with -1
    loss = torch.where(both_unknown, torch.full_like(loss, -1.0), loss)

    return loss


@torch.compile
def _cross_entropy_mixup_bwd_kernel(
    logits: Tensor,
    labels: Tensor,
    grad_output: Tensor,
    weights: Tensor,
    apply_mixup: Tensor,
) -> Tensor:
    """Backward pass for cross-entropy with mixup.

    Args:
        logits: Logits tensor (B, C)
        labels: Label tensor (B,)
        grad_output: Gradient from upstream (B,)
        weights: Mixup weights (B,)
        apply_mixup: Boolean mask (B,)

    Returns:
        Gradient w.r.t. logits (B, C)
    """
    batch_size, num_classes = logits.shape

    # Handle unknown labels
    unknown_mask = labels == -1
    valid_labels = torch.where(unknown_mask, torch.zeros_like(labels), labels)
    labels_one_hot = torch.nn.functional.one_hot(valid_labels, num_classes).float()

    # Mix labels
    labels_one_hot_other = labels_one_hot.roll(-1, 0)
    weights_expanded = weights.view(batch_size, 1)
    mixed_labels = weights_expanded * labels_one_hot + (1 - weights_expanded) * labels_one_hot_other

    # Check if mixed label is unknown
    labels_other = labels.roll(-1, 0)
    unknown_other = labels_other == -1
    both_unknown = unknown_mask | (apply_mixup & unknown_other)

    # Compute gradient: softmax(logits) - mixed_labels
    probs = torch.nn.functional.softmax(logits, dim=-1)
    grad = probs - mixed_labels

    # Apply chain rule with upstream gradient
    grad = grad * grad_output.view(batch_size, 1)

    # Zero out gradients for unknown samples
    grad = torch.where(both_unknown.view(batch_size, 1), torch.zeros_like(grad), grad)

    return grad


class CrossEntropyMixup(Function):
    @staticmethod
    def forward(ctx, logits: Tensor, labels: Tensor, mixup_prob: float, mixup_alpha: float, seed: int) -> Tensor:
        ctx.mixup_prob = mixup_prob
        ctx.mixup_alpha = mixup_alpha
        ctx.seed = seed

        batch_size = logits.shape[0]
        device = logits.device

        # Get mixup parameters
        weights = get_weights(batch_size, mixup_prob, mixup_alpha, seed, device)
        apply_mixup = is_mixed(batch_size, mixup_prob, mixup_alpha, seed, device)

        ctx.save_for_backward(logits, labels, weights, apply_mixup)

        # Compute loss
        loss = _cross_entropy_mixup_fwd_kernel(logits, labels, weights, apply_mixup)

        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor, *grad_outputs) -> Tuple[Tensor, None, None, None, None]:  # type: ignore[override]
        logits, labels, weights, apply_mixup = ctx.saved_tensors

        # Compute gradient
        grad = _cross_entropy_mixup_bwd_kernel(logits, labels, grad_output, weights, apply_mixup)

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
    result = CrossEntropyMixup.apply(logits, labels, mixup_prob, mixup_alpha, seed)
    assert result is not None
    return result


@torch.compile
def _bce_mixup_fwd_kernel(
    logits: Tensor,
    labels: Tensor,
    weights: Tensor,
    apply_mixup: Tensor,
    pos_weight: float,
) -> Tensor:
    """Forward pass for BCE with mixup.

    Args:
        logits: Logits tensor (B, ...)
        labels: Label tensor (B, ...)
        weights: Mixup weights (B,)
        apply_mixup: Boolean mask (B,)
        pos_weight: Optional positive class weight

    Returns:
        Loss tensor (B, ...)
    """
    batch_size = logits.shape[0]

    # Handle unknown labels (-1)
    unknown_mask = labels == -1

    # Mix labels with rolled version
    labels_other = labels.roll(-1, 0)
    weights_expanded = weights.view(batch_size, *([1] * (logits.ndim - 1)))
    mixed_labels = weights_expanded * labels + (1 - weights_expanded) * labels_other

    # Check if mixed label is unknown
    unknown_other = labels_other == -1
    unknown_mask_expanded = unknown_mask | (apply_mixup.view(batch_size, *([1] * (logits.ndim - 1))) & unknown_other)

    # Compute BCE loss: softplus(logit) - logit * target
    # Using stable computation: max(0, x) + log(1 + exp(-|x|))
    zeros = torch.zeros_like(logits)
    softplus = torch.maximum(logits, zeros) + torch.log1p(torch.exp(-torch.abs(logits)))
    loss = softplus - logits * mixed_labels

    # Apply pos_weight if provided
    if pos_weight >= 0.0:
        weight_factor = mixed_labels * pos_weight + (1 - mixed_labels) * (1 - pos_weight)
        loss = loss * weight_factor

    # Mark unknown samples with -1
    loss = torch.where(unknown_mask_expanded, torch.full_like(loss, -1.0), loss)

    return loss


@torch.compile
def _bce_mixup_bwd_kernel(
    logits: Tensor,
    labels: Tensor,
    grad_output: Tensor,
    weights: Tensor,
    apply_mixup: Tensor,
    pos_weight: float,
) -> Tensor:
    """Backward pass for BCE with mixup.

    Args:
        logits: Logits tensor (B, ...)
        labels: Label tensor (B, ...)
        grad_output: Gradient from upstream (B, ...)
        weights: Mixup weights (B,)
        apply_mixup: Boolean mask (B,)
        pos_weight: Optional positive class weight

    Returns:
        Gradient w.r.t. logits (B, ...)
    """
    batch_size = logits.shape[0]

    # Handle unknown labels
    unknown_mask = labels == -1

    # Mix labels
    labels_other = labels.roll(-1, 0)
    weights_expanded = weights.view(batch_size, *([1] * (logits.ndim - 1)))
    mixed_labels = weights_expanded * labels + (1 - weights_expanded) * labels_other

    # Check if mixed label is unknown
    unknown_other = labels_other == -1
    unknown_mask_expanded = unknown_mask | (apply_mixup.view(batch_size, *([1] * (logits.ndim - 1))) & unknown_other)

    # Compute gradient: sigmoid(logit) - target
    # Use stable sigmoid computation
    sigmoid = torch.where(logits > 0, 1.0 / (1.0 + torch.exp(-logits)), torch.exp(logits) / (1.0 + torch.exp(logits)))
    grad = sigmoid - mixed_labels

    # Apply pos_weight if provided
    if pos_weight >= 0.0:
        weight_factor = mixed_labels * pos_weight + (1 - mixed_labels) * (1 - pos_weight)
        grad = grad * weight_factor

    # Apply chain rule with upstream gradient
    grad = grad * grad_output

    # Zero out gradients for unknown samples
    grad = torch.where(unknown_mask_expanded, torch.zeros_like(grad), grad)

    return grad


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
        ctx.mixup_prob = mixup_prob
        ctx.mixup_alpha = mixup_alpha
        ctx.seed = seed
        ctx.pos_weight = -1.0 if pos_weight is None else pos_weight

        batch_size = logits.shape[0]
        device = logits.device

        # Get mixup parameters
        weights = get_weights(batch_size, mixup_prob, mixup_alpha, seed, device)
        apply_mixup = is_mixed(batch_size, mixup_prob, mixup_alpha, seed, device)

        ctx.save_for_backward(logits, labels, weights, apply_mixup)

        # Compute loss
        loss = _bce_mixup_fwd_kernel(logits, labels, weights, apply_mixup, ctx.pos_weight)

        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor, *grad_outputs) -> Tuple[Tensor, None, None, None, None, None]:  # type: ignore[override]
        logits, labels, weights, apply_mixup = ctx.saved_tensors
        pos_weight = ctx.pos_weight

        # Compute gradient
        grad = _bce_mixup_bwd_kernel(logits, labels, grad_output, weights, apply_mixup, pos_weight)

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
    if pos_weight is not None and not (0 <= pos_weight <= 1):
        raise ValueError("pos_weight must be in range [0, 1]")
    result = BCEMixup.apply(logits, labels, mixup_prob, mixup_alpha, seed, pos_weight)
    assert result is not None
    return result


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
