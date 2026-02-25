from typing import cast

import pytest
import torch
import torch.nn.functional as F

from mjepa.metrics import CLSPatchAlignmentMetric, SimilarityDistanceCouplingMetric


STRICT_ATOL = 1e-7
DETERMINISTIC_ATOL = 1e-12
QUANTILE_TOLERANCE_SCALE = 4.0
RANDOM_TEST_SEED = 123
COUPLING_TEST_SEED = 0
DETERMINISM_TEST_SEED = 1234


def assert_metric_allclose(actual: torch.Tensor, expected: torch.Tensor, *, atol: float) -> None:
    assert torch.allclose(actual, expected.to(actual.dtype), atol=atol)


class TestCLSPatchAlignmentMetric:
    def test_outputs_named_keys(self):
        metric = CLSPatchAlignmentMetric()
        cls_tokens = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        patch_tokens = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )

        metric.update(cls_tokens, patch_tokens)
        out = metric.compute()

        assert set(out.keys()) == {"cpa_mean", "cpa_std", "cpa_p90", "cpa_p99"}

    def test_shapes_accept_b_d_and_b_c_d(self):
        metric = CLSPatchAlignmentMetric()

        cls_b_d = torch.randn(2, 8)
        patch = torch.randn(2, 5, 8)
        metric.update(cls_b_d, patch)

        cls_b_c_d = torch.randn(2, 3, 8)
        metric.update(cls_b_c_d, patch)

        out = metric.compute()
        assert torch.isfinite(out["cpa_mean"])

    def test_rejects_mismatched_shapes(self):
        metric = CLSPatchAlignmentMetric()

        with pytest.raises(ValueError, match="cls_tokens must have shape"):
            metric.update(torch.randn(2, 3, 4, 5), torch.randn(2, 6, 5))

        with pytest.raises(ValueError, match="patch_tokens must have shape"):
            metric.update(torch.randn(2, 5), torch.randn(2, 3, 4, 5))

        with pytest.raises(ValueError, match="Batch size mismatch"):
            metric.update(torch.randn(2, 5), torch.randn(3, 4, 5))

        with pytest.raises(ValueError, match="Hidden dimension mismatch"):
            metric.update(torch.randn(2, 6), torch.randn(2, 4, 5))

    def test_internal_normalization_matches_manual_cosine(self):
        metric = CLSPatchAlignmentMetric(num_bins=4096)

        cls_tokens = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        patch_tokens = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 5.0]],
                [[0.0, 2.0], [4.0, 0.0]],
            ]
        )

        metric.update(cls_tokens, patch_tokens)
        out = metric.compute()

        cls_norm = F.normalize(cls_tokens, dim=-1)
        patch_norm = F.normalize(patch_tokens, dim=-1)
        expected = (patch_norm * cls_norm[:, None, :]).sum(-1).reshape(-1)

        assert_metric_allclose(out["cpa_mean"], expected.mean(), atol=STRICT_ATOL)
        assert_metric_allclose(out["cpa_std"], expected.std(unbiased=False), atol=STRICT_ATOL)

    def test_mean_std_on_known_small_example(self):
        metric = CLSPatchAlignmentMetric(num_bins=2048)

        cls_tokens = torch.tensor([[1.0, 0.0]])
        patch_tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])

        metric.update(cls_tokens, patch_tokens)
        out = metric.compute()

        values = torch.tensor([1.0, 0.0, -1.0])
        assert_metric_allclose(out["cpa_mean"], values.mean(), atol=STRICT_ATOL)
        assert_metric_allclose(out["cpa_std"], values.std(unbiased=False), atol=STRICT_ATOL)

    def test_quantiles_histogram_reasonable_accuracy(self):
        num_bins = 2048
        metric = CLSPatchAlignmentMetric(num_bins=num_bins)

        cls_tokens = torch.tensor([[1.0, 0.0]]).repeat(1, 1)
        sims = torch.linspace(-1.0, 1.0, 20000)
        patch_tokens = torch.stack([sims, torch.sqrt(torch.clamp(1 - sims**2, min=0.0))], dim=-1).unsqueeze(0)

        metric.update(cls_tokens, patch_tokens)
        out = metric.compute()

        exact_p90 = torch.quantile(sims, 0.90)
        exact_p99 = torch.quantile(sims, 0.99)

        tolerance = QUANTILE_TOLERANCE_SCALE / num_bins
        assert_metric_allclose(out["cpa_p90"], exact_p90, atol=tolerance)
        assert_metric_allclose(out["cpa_p99"], exact_p99, atol=tolerance)

    def test_empty_returns_nans(self):
        metric = CLSPatchAlignmentMetric()
        out = metric.compute()

        for value in out.values():
            assert torch.isnan(value)

    def test_reset_clears_state(self):
        metric = CLSPatchAlignmentMetric()
        metric.update(torch.randn(2, 4), torch.randn(2, 5, 4))

        metric.reset()
        out = metric.compute()
        for value in out.values():
            assert torch.isnan(value)

    def test_multi_cls_uses_all_pairs(self):
        metric = CLSPatchAlignmentMetric(num_bins=4096)

        cls_tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        patch_tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        metric.update(cls_tokens, patch_tokens)
        out = metric.compute()

        cls_norm = F.normalize(cls_tokens, dim=-1)
        patch_norm = F.normalize(patch_tokens, dim=-1)
        expected = torch.einsum("bcd,bnd->bcn", cls_norm, patch_norm).reshape(-1)

        assert_metric_allclose(out["cpa_mean"], expected.mean(), atol=STRICT_ATOL)
        assert_metric_allclose(out["cpa_std"], expected.std(unbiased=False), atol=STRICT_ATOL)


class TestSimilarityDistanceCouplingMetric:
    def test_outputs_named_keys(self):
        metric = SimilarityDistanceCouplingMetric(pairs_per_img=256)
        patch_tokens = torch.randn(2, 6, 8)

        metric.update(patch_tokens, [2, 3])
        out = metric.compute()

        assert set(out.keys()) == {"sdc_spearman_proxy"}

    def test_rejects_invalid_init(self):
        with pytest.raises(ValueError, match="pairs_per_img must be positive"):
            SimilarityDistanceCouplingMetric(pairs_per_img=0)

        with pytest.raises(ValueError, match="eps must be positive"):
            SimilarityDistanceCouplingMetric(eps=0.0)

    def test_rejects_invalid_update_inputs(self):
        metric = SimilarityDistanceCouplingMetric()

        with pytest.raises(ValueError, match="patch_tokens must have shape"):
            metric.update(torch.randn(2, 4), (2, 2))

        with pytest.raises(ValueError, match="grid_hw must have length 2"):
            metric.update(torch.randn(2, 4, 3), [2])

        with pytest.raises(ValueError, match="grid_hw entries must be integers"):
            metric.update(torch.randn(2, 4, 3), cast(list[int], [2.0, 2]))

        with pytest.raises(ValueError, match="grid_hw entries must be positive"):
            metric.update(torch.randn(2, 4, 3), (0, 2))

        with pytest.raises(ValueError, match="Number of patches must match H \\* W"):
            metric.update(torch.randn(2, 5, 3), (2, 2))

    def test_output_is_finite_on_random_input(self):
        torch.manual_seed(RANDOM_TEST_SEED)
        metric = SimilarityDistanceCouplingMetric(pairs_per_img=1024)
        patch_tokens = torch.randn(3, 16, 12)

        metric.update(patch_tokens, (4, 4))
        out = metric.compute()

        assert torch.isfinite(out["sdc_spearman_proxy"])

    def test_empty_returns_nan(self):
        metric = SimilarityDistanceCouplingMetric()
        out = metric.compute()

        assert torch.isnan(out["sdc_spearman_proxy"])

    def test_reset_clears_state(self):
        metric = SimilarityDistanceCouplingMetric()
        metric.update(torch.randn(2, 9, 7), (3, 3))

        metric.reset()
        out = metric.compute()
        assert torch.isnan(out["sdc_spearman_proxy"])

    def test_localized_features_have_positive_coupling(self):
        h, w = 8, 8
        n = h * w
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), dim=-1).reshape(n, 2)
        coords = coords.to(dtype=torch.float32)
        dist = torch.cdist(coords, coords)
        sigma = 1.5
        patch_tokens = torch.exp(-(dist**2) / (2.0 * sigma**2)).unsqueeze(0)

        torch.manual_seed(COUPLING_TEST_SEED)
        metric = SimilarityDistanceCouplingMetric(pairs_per_img=8192)
        metric.update(patch_tokens, (h, w))
        out = metric.compute()

        assert out["sdc_spearman_proxy"] > 0.7

    def test_random_features_have_low_coupling(self):
        torch.manual_seed(COUPLING_TEST_SEED)
        metric = SimilarityDistanceCouplingMetric(pairs_per_img=8192)
        patch_tokens = torch.randn(4, 64, 32)

        metric.update(patch_tokens, (8, 8))
        out = metric.compute()

        assert torch.abs(out["sdc_spearman_proxy"]) < 0.15

    def test_sampling_is_deterministic_with_seed(self):
        patch_tokens = torch.randn(2, 16, 10)

        torch.manual_seed(DETERMINISM_TEST_SEED)
        metric_a = SimilarityDistanceCouplingMetric(pairs_per_img=4096)
        metric_a.update(patch_tokens, (4, 4))
        out_a = metric_a.compute()

        torch.manual_seed(DETERMINISM_TEST_SEED)
        metric_b = SimilarityDistanceCouplingMetric(pairs_per_img=4096)
        metric_b.update(patch_tokens, (4, 4))
        out_b = metric_b.compute()

        assert torch.allclose(out_a["sdc_spearman_proxy"], out_b["sdc_spearman_proxy"], atol=DETERMINISTIC_ATOL)
