import pytest
import torch
from torch.testing import assert_close

from mjepa.augmentation.posterize import posterize, posterize_


@pytest.mark.cuda
class TestPosterize:

    def test_posterize_pointwise_prob_0(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        x_orig = x.clone()
        y = posterize(x, posterize_prob=0.0, bits=8)
        assert_close(y, x)
        assert_close(y, x_orig)

    @pytest.mark.parametrize("bits", [4, 6, 8])
    def test_posterize_pointwise_prob_1(self, bits):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        x_orig = x.clone()
        y = posterize(x, posterize_prob=1.0, bits=bits)
        assert_close(x, x_orig)
        expected = (x * (2**bits - 1)).round().div(2**bits - 1)
        assert_close(y, expected)

    def test_posterize_determinism(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        y1 = posterize(x, posterize_prob=0.5, bits=8, seed=0)
        y2 = posterize(x, posterize_prob=0.5, bits=8, seed=0)
        assert_close(y1, y2)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_posterize_dtypes(self, dtype):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda", dtype=dtype)
        y = posterize(x, posterize_prob=0.5, bits=8, seed=0)
        assert y.dtype == x.dtype
        assert y.shape == x.shape
        assert y.device == x.device


@pytest.mark.cuda
class TestPosterizeInplace:

    def test_posterize_pointwise_prob_0(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        x_orig = x.clone()
        posterize_(x, posterize_prob=0.0, bits=8)
        assert_close(x, x_orig)

    @pytest.mark.parametrize("bits", [4, 6, 8])
    def test_posterize_pointwise_prob_1(self, bits):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        expected = (x * (2**bits - 1)).round().div(2**bits - 1)
        posterize_(x, posterize_prob=1.0, bits=bits)
        assert_close(x, expected)

    def test_posterize_determinism(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        y1 = x.clone()
        y2 = x.clone()
        posterize_(y1, posterize_prob=0.5, bits=8, seed=0)
        posterize_(y2, posterize_prob=0.5, bits=8, seed=0)
        assert_close(y1, y2)
