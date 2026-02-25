from __future__ import annotations

from numbers import Integral
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric


SIMILARITY_MIN = -1.0
SIMILARITY_MAX = 1.0
SIMILARITY_SPAN = SIMILARITY_MAX - SIMILARITY_MIN
CPA_P90 = 0.90
CPA_P99 = 0.99
GRID_HW_NUM_DIMS = 2
CPA_RESULT_KEYS = ("cpa_mean", "cpa_std", "cpa_p90", "cpa_p99")
SDC_RESULT_KEY = "sdc_spearman_proxy"


def _nan_like(reference: Tensor) -> Tensor:
    return torch.tensor(float("nan"), dtype=reference.dtype, device=reference.device)


def _nan_result(reference: Tensor, keys: tuple[str, ...]) -> dict[str, Tensor]:
    nan = _nan_like(reference)
    return {key: nan for key in keys}


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
        self._bin_width = SIMILARITY_SPAN / float(num_bins)

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
        sims = sims.clamp(SIMILARITY_MIN, SIMILARITY_MAX)
        sims64 = sims.to(dtype=torch.float64)

        count_state = cast(Tensor, self.count)
        sum_state = cast(Tensor, self.sum)
        sum_sq_state = cast(Tensor, self.sum_sq)
        hist_state = cast(Tensor, self.hist)

        count_state.add_(sims64.numel())
        sum_state.add_(sims64.sum())
        sum_sq_state.add_(torch.square(sims64).sum())

        bin_indices = torch.floor((sims64 - SIMILARITY_MIN) / SIMILARITY_SPAN * self.num_bins).to(dtype=torch.long)
        bin_indices = bin_indices.clamp(min=0, max=self.num_bins - 1)
        hist_update = torch.bincount(bin_indices, minlength=self.num_bins).to(hist_state.dtype)
        hist_state.add_(hist_update)

    def _compute_quantile_from_hist(self, q: float) -> Tensor:
        count_state = cast(Tensor, self.count)
        sum_state = cast(Tensor, self.sum)
        hist_state = cast(Tensor, self.hist)

        if count_state <= 0:
            return _nan_like(sum_state)

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

        bin_left = SIMILARITY_MIN + idx.to(dtype=sum_state.dtype) * self._bin_width
        return bin_left + frac * self._bin_width

    def compute(self) -> dict[str, Tensor]:
        count_state = cast(Tensor, self.count)
        sum_state = cast(Tensor, self.sum)
        sum_sq_state = cast(Tensor, self.sum_sq)

        if count_state <= 0:
            return _nan_result(sum_state, CPA_RESULT_KEYS)

        count = count_state.to(dtype=sum_state.dtype)
        mean = sum_state / count
        variance = sum_sq_state / count - torch.square(mean)
        variance = variance.clamp(min=0.0)
        std = torch.sqrt(variance)

        return {
            "cpa_mean": mean,
            "cpa_std": std,
            "cpa_p90": self._compute_quantile_from_hist(CPA_P90),
            "cpa_p99": self._compute_quantile_from_hist(CPA_P99),
        }

    def plot(self, val: Any = None, ax: Any = None) -> Any:
        raise NotImplementedError("Plotting is not implemented for CLSPatchAlignmentMetric")


class SimilarityDistanceCouplingMetric(Metric):
    """Measure how strongly similarity is coupled to spatial proximity.

    Samples patch pairs, computes cosine similarity and Euclidean distance, then
    estimates a Spearman-style correlation by running Pearson on rank-normalized
    values. Aggregation across updates is a pair-count weighted mean.
    """

    full_state_update = False

    def __init__(self, pairs_per_img: int = 2048, eps: float = 1e-12, sync_on_compute: bool = True) -> None:
        super().__init__(sync_on_compute=sync_on_compute)
        if pairs_per_img <= 0:
            raise ValueError("pairs_per_img must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.pairs_per_img = pairs_per_img
        self.eps = eps

        self.add_state("pair_count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("rho_weighted_sum", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    def _rank_normalize(self, values: Tensor) -> Tensor:
        ranks = torch.argsort(torch.argsort(values)).to(dtype=torch.float64)
        mean = ranks.mean()
        std = ranks.std(unbiased=False)
        return (ranks - mean) / (std + self.eps)

    @staticmethod
    def _parse_grid_hw(grid_hw: tuple[int, int] | list[int]) -> tuple[int, int]:
        if len(grid_hw) != GRID_HW_NUM_DIMS:
            raise ValueError("grid_hw must have length 2 as (H, W)")

        h_raw, w_raw = grid_hw
        if not isinstance(h_raw, Integral) or not isinstance(w_raw, Integral):
            raise ValueError("grid_hw entries must be integers")

        h = int(h_raw)
        w = int(w_raw)
        if h <= 0 or w <= 0:
            raise ValueError("grid_hw entries must be positive")
        return h, w

    def update(self, patch_tokens: Tensor, grid_hw: tuple[int, int] | list[int]) -> None:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must have shape [B, N, D]")

        h, w = self._parse_grid_hw(grid_hw)

        batch_size, num_patches, dim = patch_tokens.shape
        expected_patches = h * w
        if num_patches != expected_patches:
            raise ValueError("Number of patches must match H * W")

        patch_norm = F.normalize(patch_tokens, dim=-1, eps=self.eps)

        device = patch_norm.device
        ys = torch.arange(h, device=device, dtype=patch_norm.dtype)
        xs = torch.arange(w, device=device, dtype=patch_norm.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)

        i = torch.randint(0, num_patches, (batch_size, self.pairs_per_img), device=device)
        j = torch.randint(0, num_patches, (batch_size, self.pairs_per_img), device=device)

        zi = patch_norm.gather(1, i[..., None].expand(batch_size, self.pairs_per_img, dim))
        zj = patch_norm.gather(1, j[..., None].expand(batch_size, self.pairs_per_img, dim))
        sims = (zi * zj).sum(dim=-1).reshape(-1)

        pi = coords[i.reshape(-1)]
        pj = coords[j.reshape(-1)]
        dists = (pi - pj).norm(dim=-1)

        sim_rank = self._rank_normalize(sims)
        dist_rank = self._rank_normalize(dists)
        rho_update = (sim_rank * (-dist_rank)).mean()

        pair_count_state = cast(Tensor, self.pair_count)
        rho_weighted_sum_state = cast(Tensor, self.rho_weighted_sum)

        num_pairs = sims.numel()
        pair_count_state.add_(num_pairs)
        rho_weighted_sum_state.add_(
            rho_update.to(dtype=rho_weighted_sum_state.dtype, device=rho_weighted_sum_state.device) * num_pairs
        )

    def compute(self) -> dict[str, Tensor]:
        pair_count_state = cast(Tensor, self.pair_count)
        rho_weighted_sum_state = cast(Tensor, self.rho_weighted_sum)

        if pair_count_state <= 0:
            return _nan_result(rho_weighted_sum_state, (SDC_RESULT_KEY,))

        pair_count = pair_count_state.to(dtype=rho_weighted_sum_state.dtype)
        return {SDC_RESULT_KEY: rho_weighted_sum_state / pair_count}

    def plot(self, val: Any = None, ax: Any = None) -> Any:
        raise NotImplementedError("Plotting is not implemented for SimilarityDistanceCouplingMetric")
