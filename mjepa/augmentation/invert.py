import torch
from torch import Tensor


@torch.compile
def _invert_kernel(
    input: Tensor,
    invert_mask: Tensor,
    solarize_mask: Tensor,
    solarize_threshold: float,
) -> Tensor:
    """Apply invert and solarize operations per batch element.

    Args:
        input: Input tensor of shape (B, ...)
        invert_mask: Boolean mask of shape (B,) indicating which samples to invert
        solarize_mask: Boolean mask of shape (B,) indicating which samples to solarize
        solarize_threshold: Threshold above which to invert values during solarize

    Returns:
        Output tensor with same shape as input
    """
    batch_size = input.shape[0]
    # Reshape masks to broadcast over spatial dimensions
    invert_mask = invert_mask.view(batch_size, *([1] * (input.ndim - 1)))
    solarize_mask = solarize_mask.view(batch_size, *([1] * (input.ndim - 1)))

    # Apply invert
    output = torch.where(invert_mask, 1.0 - input, input)

    # Apply solarize (invert values above threshold)
    output = torch.where(solarize_mask & (output > solarize_threshold), 1.0 - output, output)

    return output


def invert(
    input: Tensor,
    invert_prob: float,
    solarize_prob: float = 0.0,
    solarize_threshold: float = 0.5,
    seed: int | None = None,
) -> Tensor:
    """Apply invert and/or solarize operations to input tensor.

    Args:
        input: Input tensor of shape (B, ...)
        invert_prob: Probability of inverting each sample
        solarize_prob: Probability of solarizing each sample
        solarize_threshold: Threshold for solarization
        seed: Random seed for reproducibility

    Returns:
        Transformed tensor with same shape and dtype as input
    """
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())

    batch_size = input.shape[0]
    dtype = input.dtype
    device = input.device

    # Generate per-batch random masks using separate generators
    # Match CUDA behavior: seed + batch_idx for invert, seed + batch_idx + batch_size for solarize
    invert_gen = torch.Generator(device=device).manual_seed(seed)
    invert_mask = torch.rand(batch_size, generator=invert_gen, device=device) < invert_prob

    solarize_gen = torch.Generator(device=device).manual_seed(seed + batch_size)
    solarize_mask = torch.rand(batch_size, generator=solarize_gen, device=device) < solarize_prob

    # Apply transformations
    output = _invert_kernel(input.float(), invert_mask, solarize_mask, solarize_threshold)

    return output.to(dtype)


def invert_(
    input: Tensor,
    invert_prob: float,
    solarize_prob: float = 0.0,
    solarize_threshold: float = 0.5,
    seed: int | None = None,
) -> Tensor:
    """In-place version of invert operation.

    Args:
        input: Input tensor of shape (B, ...)
        invert_prob: Probability of inverting each sample
        solarize_prob: Probability of solarizing each sample
        solarize_threshold: Threshold for solarization
        seed: Random seed for reproducibility

    Returns:
        Modified input tensor
    """
    result = invert(input, invert_prob, solarize_prob, solarize_threshold, seed)
    input.copy_(result)
    return input
