import pytest
import torch

from mjepa.metrics import CLSPatchAlignmentMetric


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

        cls_norm = torch.nn.functional.normalize(cls_tokens, dim=-1)
        patch_norm = torch.nn.functional.normalize(patch_tokens, dim=-1)
        expected = (patch_norm * cls_norm[:, None, :]).sum(-1).reshape(-1)

        assert torch.allclose(out["cpa_mean"], expected.mean().to(out["cpa_mean"].dtype), atol=1e-7)
        assert torch.allclose(out["cpa_std"], expected.std(unbiased=False).to(out["cpa_std"].dtype), atol=1e-7)

    def test_mean_std_on_known_small_example(self):
        metric = CLSPatchAlignmentMetric(num_bins=2048)

        cls_tokens = torch.tensor([[1.0, 0.0]])
        patch_tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])

        metric.update(cls_tokens, patch_tokens)
        out = metric.compute()

        values = torch.tensor([1.0, 0.0, -1.0])
        assert torch.allclose(out["cpa_mean"], values.mean().to(out["cpa_mean"].dtype), atol=1e-7)
        assert torch.allclose(out["cpa_std"], values.std(unbiased=False).to(out["cpa_std"].dtype), atol=1e-7)

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

        tolerance = 4.0 / num_bins
        assert torch.allclose(out["cpa_p90"], exact_p90.to(out["cpa_p90"].dtype), atol=tolerance)
        assert torch.allclose(out["cpa_p99"], exact_p99.to(out["cpa_p99"].dtype), atol=tolerance)

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

        cls_norm = torch.nn.functional.normalize(cls_tokens, dim=-1)
        patch_norm = torch.nn.functional.normalize(patch_tokens, dim=-1)
        expected = torch.einsum("bcd,bnd->bcn", cls_norm, patch_norm).reshape(-1)

        assert torch.allclose(out["cpa_mean"], expected.mean().to(out["cpa_mean"].dtype), atol=1e-7)
        assert torch.allclose(out["cpa_std"], expected.std(unbiased=False).to(out["cpa_std"].dtype), atol=1e-7)
