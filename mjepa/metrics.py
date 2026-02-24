from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric


class CLSPatchAlignmentMetric(Metric):
    """Measure how much patch tokens align with CLS tokens.

    Computes global mean and standard deviation exactly across all updates, and estimates
    p90/p99 using a fixed histogram over cosine similarities in [-1, 1].
    """

    full_state_update = False

    def __init__(self, num_bins: int = 2048, eps: float = 1e-12, sync_on_compute: bool = True) -> None:
        super().__init__(sync_on_compute=sync_on_compute)
        if num_bins <= 1:
            raise ValueError("num_bins must be greater than 1")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.num_bins = num_bins
        self.eps = eps
        self._bin_width = 2.0 / float(num_bins)

        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("sum_sq", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("hist", default=torch.zeros(num_bins, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, cls_tokens: Tensor, patch_tokens: Tensor) -> None:
        if cls_tokens.ndim == 2:
            cls_tokens = cls_tokens.unsqueeze(1)
        elif cls_tokens.ndim != 3:
            raise ValueError("cls_tokens must have shape [B, D] or [B, C, D]")

        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must have shape [B, N, D]")

        if cls_tokens.shape[0] != patch_tokens.shape[0]:
            raise ValueError("Batch size mismatch between cls_tokens and patch_tokens")

        if cls_tokens.shape[-1] != patch_tokens.shape[-1]:
            raise ValueError("Hidden dimension mismatch between cls_tokens and patch_tokens")

        cls_norm = F.normalize(cls_tokens, dim=-1, eps=self.eps)
        patch_norm = F.normalize(patch_tokens, dim=-1, eps=self.eps)

        sims = torch.einsum("bcd,bnd->bcn", cls_norm, patch_norm).reshape(-1)
        sims = sims.clamp(-1.0, 1.0)
        sims64 = sims.to(dtype=torch.float64)

        count_state = cast(Tensor, self.count)
        sum_state = cast(Tensor, self.sum)
        sum_sq_state = cast(Tensor, self.sum_sq)
        hist_state = cast(Tensor, self.hist)

        count_state.add_(sims64.numel())
        sum_state.add_(sims64.sum())
        sum_sq_state.add_(torch.square(sims64).sum())

        bin_indices = torch.floor((sims64 + 1.0) / 2.0 * self.num_bins).to(dtype=torch.long)
        bin_indices = bin_indices.clamp(min=0, max=self.num_bins - 1)
        hist_update = torch.bincount(bin_indices, minlength=self.num_bins).to(hist_state.dtype)
        hist_state.add_(hist_update)

    def _compute_quantile_from_hist(self, q: float) -> Tensor:
        count_state = cast(Tensor, self.count)
        sum_state = cast(Tensor, self.sum)
        hist_state = cast(Tensor, self.hist)

        if count_state <= 0:
            return torch.tensor(float("nan"), dtype=sum_state.dtype, device=sum_state.device)

        count = count_state.to(dtype=sum_state.dtype)
        cdf = torch.cumsum(hist_state, dim=0)

        rank = torch.tensor(q, dtype=sum_state.dtype, device=sum_state.device) * count
        idx = torch.searchsorted(cdf, rank, right=False)
        idx = idx.clamp(max=self.num_bins - 1)

        prev_idx = (idx - 1).clamp(min=0)
        prev_cdf = torch.where(idx > 0, cdf[prev_idx], torch.zeros_like(rank))
        bin_count = cdf[idx] - prev_cdf
        safe_bin_count = torch.where(bin_count > 0, bin_count, torch.ones_like(bin_count))

        frac = (rank - prev_cdf) / safe_bin_count
        frac = frac.clamp(0.0, 1.0)

        bin_left = -1.0 + idx.to(dtype=sum_state.dtype) * self._bin_width
        return bin_left + frac * self._bin_width

    def compute(self) -> dict[str, Tensor]:
        count_state = cast(Tensor, self.count)
        sum_state = cast(Tensor, self.sum)
        sum_sq_state = cast(Tensor, self.sum_sq)

        if count_state <= 0:
            nan = torch.tensor(float("nan"), dtype=sum_state.dtype, device=sum_state.device)
            return {
                "cpa_mean": nan,
                "cpa_std": nan,
                "cpa_p90": nan,
                "cpa_p99": nan,
            }

        count = count_state.to(dtype=sum_state.dtype)
        mean = sum_state / count
        variance = sum_sq_state / count - torch.square(mean)
        variance = variance.clamp(min=0.0)
        std = torch.sqrt(variance)

        return {
            "cpa_mean": mean,
            "cpa_std": std,
            "cpa_p90": self._compute_quantile_from_hist(0.90),
            "cpa_p99": self._compute_quantile_from_hist(0.99),
        }

    def plot(self, val: Any = None, ax: Any = None) -> Any:
        raise NotImplementedError("Plotting is not implemented for CLSPatchAlignmentMetric")
