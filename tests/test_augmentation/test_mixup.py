import timeit

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from mjepa.augmentation.mixup import bce_mixup, cross_entropy_mixup, get_weights, is_mixed, mixup


@pytest.mark.cuda
class TestMixUp:

    def test_get_weights(self):
        weights = get_weights(4, mixup_prob=0.0, mixup_alpha=1.0, seed=0)
        assert weights.device.type == "cuda"
        assert weights.shape == (4,)
        assert (weights == 1.0).all()

        weights = get_weights(4, mixup_prob=1.0, mixup_alpha=1.0, seed=0)
        assert weights.device.type == "cuda"
        assert weights.shape == (4,)
        assert (weights < 1.0).all()

    def test_is_mixed(self):
        mixed = is_mixed(4, mixup_prob=0.0, mixup_alpha=1.0, seed=0)
        assert mixed.device.type == "cuda"
        assert mixed.shape == (4,)
        assert not mixed.any()

        mixed = is_mixed(4, mixup_prob=1.0, mixup_alpha=1.0, seed=0)
        assert mixed.device.type == "cuda"
        assert mixed.shape == (4,)
        assert mixed.all()

        weights = get_weights(4, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        mixed = is_mixed(4, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        assert mixed.device.type == "cuda"
        assert mixed.shape == (4,)
        assert (mixed == (weights < 1.0)).all()

    def test_mixup_pointwise_prob_0(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        x_orig = x.clone()
        y = mixup(x, mixup_prob=0.0)
        assert_close(y, x)
        assert_close(y, x_orig)

    def test_mixup_pointwise_prob_1(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        x_orig = x.clone()
        y = mixup(x, mixup_prob=1.0)
        assert_close(x, x_orig)
        assert not (y == x_orig).view(4, -1).all(dim=-1).any()

    def test_mixup_determinism(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        y1 = mixup(x, mixup_prob=1.0, seed=0)
        y2 = mixup(x, mixup_prob=1.0, seed=0)
        y3 = mixup(x, mixup_prob=1.0, seed=1)
        assert_close(y1, y2)
        assert not torch.allclose(y1, y3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_mixup(self, dtype):
        torch.random.manual_seed(0)
        B = 32
        x = torch.randn(B, 1, device="cuda", dtype=dtype)
        seed = 0
        mixup_prob = 0.5
        mixup_alpha = 1.0
        weights = get_weights(B, mixup_prob, mixup_alpha, seed).type_as(x)
        weights = weights.view(B, 1).expand_as(x)
        other = x.roll(-1, 0)
        expected = x * weights + other * (1 - weights)

        y = mixup(x, mixup_prob, mixup_alpha, seed)
        assert_close(y, expected)


@pytest.mark.cuda
class TestCrossEntropyMixup:

    @pytest.mark.parametrize(
        "batch,nclass",
        [
            (
                32,
                2,
            ),
            (
                1024,
                1024,
            ),
        ],
    )
    def test_cross_entropy_mixup_prob_0(self, batch, nclass):
        torch.random.manual_seed(0)
        logits = torch.randn(batch, nclass, device="cuda")
        labels = torch.randint(0, nclass, (batch,), device="cuda")
        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.0, mixup_alpha=1.0, seed=0)
        expected = F.cross_entropy(logits, labels, reduction="none")
        assert_close(loss, expected)

    def test_cross_entropy_mixup_prob_0_stability(self):
        torch.random.manual_seed(0)
        logits = torch.randn(4, 3, device="cuda")
        logits[0, 0] = -1000
        logits[1, 1] = 1000
        labels = torch.randint(0, 3, (4,), device="cuda")
        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.0, mixup_alpha=1.0, seed=0)
        expected = F.cross_entropy(logits, labels, reduction="none")
        assert_close(loss, expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_cross_entropy_mixup(self, dtype):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda", dtype=dtype)
        labels = torch.randint(0, C, (B,), device="cuda")
        mixup_prob = 0.5
        mixup_alpha = 1.0
        seed = 0
        weights = get_weights(B, mixup_prob, mixup_alpha, seed).type_as(logits)
        weights = weights.view(B, 1).expand_as(logits)
        labels_one_hot = F.one_hot(labels, num_classes=C)
        labels_one_hot_other = labels_one_hot.roll(-1, 0)
        mixed_labels = labels_one_hot * weights + labels_one_hot_other * (1 - weights)
        expected = F.cross_entropy(logits, mixed_labels, reduction="none")

        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        assert_close(loss, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "batch,nclass",
        [
            (
                32,
                2,
            ),
            (
                1024,
                1024,
            ),
        ],
    )
    def test_cross_entropy_mixup_speed(self, batch, nclass):
        torch.random.manual_seed(0)
        logits = torch.randn(batch, nclass, device="cuda")
        labels = torch.randint(0, nclass, (batch,), device="cuda")

        # Baseline using cross entropy with no mixup
        torch.cuda.synchronize()
        start = timeit.default_timer()
        F.cross_entropy(logits, labels, reduction="none")
        torch.cuda.synchronize()
        end = timeit.default_timer()
        ce_baseline = end - start

        # Using our mixup implementation
        torch.cuda.synchronize()
        start = timeit.default_timer()
        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        actual = end - start
        print(f"Time taken: {end - start} seconds")
        assert actual <= 10 * ce_baseline, f"Mixup is slower than baseline: {actual} vs {ce_baseline}"

    def test_cross_entropy_mixup_ignore_unknown(self):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda")
        labels = torch.randint(0, C, (B,), device="cuda")
        seed = 0
        mixup_prob = 0.5
        mixup_alpha = 1.0
        unknown_stride = 4

        # To reject unknowns in the baseline case we set one hot to a large negative value
        # and mask any mixed results that are negative along the class dimension
        labels_one_hot = F.one_hot(labels, num_classes=C)
        labels_one_hot[::unknown_stride] = -1000
        labels_one_hot_other = labels_one_hot.roll(-1, 0)
        weights = get_weights(B, mixup_prob, mixup_alpha, seed).view(B, 1).expand_as(logits)
        mixed_labels = labels_one_hot * weights + labels_one_hot_other * (1 - weights)
        valid = mixed_labels.sum(-1) > 0
        _logits = logits[valid]
        _mixed_labels = mixed_labels[valid]
        expected = F.cross_entropy(_logits, _mixed_labels, reduction="none")

        labels[::unknown_stride] = -1
        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        loss = loss[loss != -1.0]
        assert_close(loss, expected)

    def test_cross_entropy_mixup_prob_0_backward(self):
        torch.random.manual_seed(0)
        batch = 32
        nclass = 4
        logits = torch.randn(batch, nclass, device="cuda", requires_grad=True)
        labels = torch.randint(0, nclass, (batch,), device="cuda")

        expected = F.cross_entropy(logits, labels, reduction="none")
        expected.sum().backward()
        expected_grad = logits.grad
        logits.grad = None

        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.0, mixup_alpha=1.0, seed=0)
        loss.sum().backward()
        assert_close(logits.grad, expected_grad, atol=1e-4, rtol=0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_cross_entropy_mixup_backward(self, dtype):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda", requires_grad=True, dtype=dtype)
        labels = torch.randint(0, C, (B,), device="cuda")
        mixup_prob = 0.5
        mixup_alpha = 1.0
        seed = 0
        weights = get_weights(B, mixup_prob, mixup_alpha, seed)
        weights = weights.view(B, 1).expand_as(logits)
        labels_one_hot = F.one_hot(labels, num_classes=C)
        labels_one_hot_other = labels_one_hot.roll(-1, 0)
        mixed_labels = labels_one_hot * weights + labels_one_hot_other * (1 - weights)
        expected = F.cross_entropy(logits, mixed_labels, reduction="none")
        expected.sum().backward()
        expected_grad = logits.grad
        logits.grad = None

        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        loss.sum().backward()

        atol = 1e-4 if dtype == torch.float32 else 1e-2
        assert_close(logits.grad, expected_grad, atol=atol, rtol=0)

    def test_cross_entropy_mixup_backward_ignore_unknown(self):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda", requires_grad=True)
        labels = torch.randint(0, C, (B,), device="cuda")
        seed = 0
        mixup_prob = 0.5
        mixup_alpha = 1.0
        unknown_stride = 4

        # To reject unknowns in the baseline case we set one hot to a large negative value
        # and mask any mixed results that are negative along the class dimension
        labels_one_hot = F.one_hot(labels, num_classes=C)
        labels_one_hot[::unknown_stride] = -1000
        labels_one_hot_other = labels_one_hot.roll(-1, 0)
        weights = get_weights(B, mixup_prob, mixup_alpha, seed).view(B, 1).expand_as(logits)
        mixed_labels = labels_one_hot * weights + labels_one_hot_other * (1 - weights)
        valid = mixed_labels.sum(-1) > 0
        _logits = logits[valid]
        _mixed_labels = mixed_labels[valid]
        expected = F.cross_entropy(_logits, _mixed_labels, reduction="none")
        expected.sum().backward()
        expected_grad = logits.grad
        logits.grad = None

        labels[::unknown_stride] = -1
        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        loss = loss[loss != -1.0]
        loss.sum().backward()
        assert_close(logits.grad, expected_grad, atol=1e-4, rtol=0)


@pytest.mark.cuda
class TestBCEMixup:

    @pytest.mark.parametrize(
        "shape",
        [
            (
                32,
                2,
            ),
            (1024, 3, 32, 32),
        ],
    )
    def test_bce_mixup_prob_0(self, shape):
        torch.random.manual_seed(0)
        logits = torch.randn(shape, device="cuda")
        labels = torch.rand(shape, device="cuda")
        loss = bce_mixup(logits, labels, mixup_prob=0.0, mixup_alpha=1.0, seed=0)
        expected = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        assert_close(loss, expected)

    def test_bce_mixup_prob_0_stability(self):
        torch.random.manual_seed(0)
        logits = torch.randn(4, 3, device="cuda")
        logits[0, 0] = -1000
        logits[1, 1] = 1000
        labels = torch.rand(4, 3, device="cuda")
        loss = bce_mixup(logits, labels, mixup_prob=0.0, mixup_alpha=1.0, seed=0)
        expected = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        assert_close(loss, expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_bce_mixup(self, dtype):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda", dtype=dtype)
        labels = torch.rand(B, C, device="cuda", dtype=dtype)
        mixup_prob = 0.5
        mixup_alpha = 1.0
        seed = 0
        weights = get_weights(B, mixup_prob, mixup_alpha, seed).type_as(logits)
        weights = weights.view(B, 1).expand_as(logits)
        mixed_labels = labels * weights + (1 - weights) * labels.roll(-1, 0)
        expected = F.binary_cross_entropy_with_logits(logits, mixed_labels, reduction="none")

        loss = bce_mixup(logits, labels, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        atol = 1e-4 if dtype == torch.float32 else 1e-1
        assert_close(loss, expected, atol=atol, rtol=0, check_dtype=False)

    def test_bce_mixup_ignore_unknown(self):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda")
        labels = torch.rand(B, C, device="cuda")
        seed = 0
        mixup_prob = 0.5
        mixup_alpha = 1.0
        unknown_stride = 4

        # To reject unknowns in the baseline case we set one hot to a large negative value
        labels[::unknown_stride, ::unknown_stride] = -1000000
        weights = get_weights(B, mixup_prob, mixup_alpha, seed).view(B, 1).expand_as(logits)
        mixed_labels = labels * weights + (1 - weights) * labels.roll(-1, 0)
        expected = F.binary_cross_entropy_with_logits(logits, mixed_labels, reduction="none")
        expected = expected[mixed_labels >= 0]

        labels[::unknown_stride, ::unknown_stride] = -1
        loss = bce_mixup(logits, labels, mixup_prob=mixup_prob, mixup_alpha=mixup_alpha, seed=seed)
        loss = loss[loss != -1.0]
        assert_close(loss, expected)

    def test_bce_mixup_prob_0_backward(self):
        torch.random.manual_seed(0)
        batch = 32
        nclass = 4
        logits = torch.randn(batch, nclass, device="cuda", requires_grad=True)
        labels = torch.rand(batch, nclass, device="cuda")

        expected = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        expected.sum().backward()
        expected_grad = logits.grad
        logits.grad = None

        loss = bce_mixup(logits, labels, mixup_prob=0.0, mixup_alpha=1.0, seed=0)
        loss.sum().backward()
        assert_close(logits.grad, expected_grad, atol=1e-4, rtol=0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_bce_mixup_backward(self, dtype):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda", requires_grad=True, dtype=dtype)
        labels = torch.rand(B, C, device="cuda", dtype=dtype)
        mixup_prob = 0.5
        mixup_alpha = 1.0
        seed = 0
        weights = get_weights(B, mixup_prob, mixup_alpha, seed)
        weights = weights.view(B, 1).expand_as(logits)
        mixed_labels = labels * weights + (1 - weights) * labels.roll(-1, 0)
        expected = F.binary_cross_entropy_with_logits(logits, mixed_labels, reduction="none")
        expected.sum().backward()
        expected_grad = logits.grad
        logits.grad = None

        loss = bce_mixup(logits, labels, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        loss.sum().backward()
        atol = 1e-4 if dtype == torch.float32 else 1e-2
        assert_close(logits.grad, expected_grad, atol=atol, rtol=0)

    def test_bce_mixup_backward_ignore_unknown(self):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda", requires_grad=True)
        labels = torch.rand(B, C, device="cuda")
        seed = 0
        mixup_prob = 0.5
        mixup_alpha = 1.0
        unknown_stride = 4

        labels[::unknown_stride, ::unknown_stride] = -1000000
        weights = get_weights(B, mixup_prob, mixup_alpha, seed).view(B, 1).expand_as(logits)
        mixed_labels = labels * weights + (1 - weights) * labels.roll(-1, 0)
        expected = F.binary_cross_entropy_with_logits(logits, mixed_labels, reduction="none")
        expected = expected[mixed_labels >= 0]
        expected.sum().backward()
        expected_grad = logits.grad
        logits.grad = None

        labels[::unknown_stride, ::unknown_stride] = -1
        loss = bce_mixup(logits, labels, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        loss = loss[loss != -1.0]
        loss.sum().backward()
        assert_close(logits.grad, expected_grad, atol=1e-4, rtol=0)

    @pytest.mark.parametrize("pos_weight", [0.2, 0.5, 0.8])
    def test_bce_mixup_pos_weight(self, pos_weight):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda")
        labels = torch.rand(B, C, device="cuda")
        seed = 0
        mixup_prob = 0.0  # Disable mixup to test just pos_weight
        mixup_alpha = 1.0

        # Compute expected loss with manual weighting
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        expected = loss * (labels * pos_weight + (1 - labels) * (1 - pos_weight))

        # Compute actual loss with pos_weight
        actual = bce_mixup(
            logits, labels, seed=seed, mixup_prob=mixup_prob, mixup_alpha=mixup_alpha, pos_weight=pos_weight
        )
        assert_close(actual, expected, atol=1e-4, rtol=0)

    @pytest.mark.parametrize("pos_weight", [0.2, 0.5, 0.8])
    def test_bce_mixup_pos_weight_backward(self, pos_weight):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda", requires_grad=True)
        labels = torch.rand(B, C, device="cuda")
        seed = 0
        mixup_prob = 0.0  # Disable mixup to test just pos_weight
        mixup_alpha = 1.0

        # Compute expected gradient with manual weighting
        def manual_weighted_bce(logits, labels, pos_weight):
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            return loss * (labels * pos_weight + (1 - labels) * (1 - pos_weight))

        expected = manual_weighted_bce(logits, labels, pos_weight)
        expected.sum().backward()
        expected_grad = logits.grad
        logits.grad = None

        # Compute actual gradient with pos_weight
        actual = bce_mixup(
            logits, labels, seed=seed, mixup_prob=mixup_prob, mixup_alpha=mixup_alpha, pos_weight=pos_weight
        )
        actual.sum().backward()
        assert_close(logits.grad, expected_grad, atol=1e-4, rtol=0)

    @pytest.mark.parametrize("pos_weight", [0.2, 0.5, 0.8])
    def test_bce_mixup_pos_weight_with_mixup(self, pos_weight):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda", requires_grad=True)
        labels = torch.rand(B, C, device="cuda")
        seed = 0
        mixup_prob = 1.0  # Always apply mixup
        mixup_alpha = 1.0

        # Get mixup weights for manual computation
        weights = get_weights(B, mixup_prob, mixup_alpha, seed).view(B, 1).expand_as(logits)
        mixed_labels = labels * weights + (1 - weights) * labels.roll(-1, 0)

        # Compute expected loss with manual weighting
        loss = F.binary_cross_entropy_with_logits(logits, mixed_labels, reduction="none")
        expected = loss * (mixed_labels * pos_weight + (1 - mixed_labels) * (1 - pos_weight))
        expected.sum().backward()
        expected_grad = logits.grad
        logits.grad = None

        # Compute actual loss with pos_weight
        actual = bce_mixup(
            logits, labels, seed=seed, mixup_prob=mixup_prob, mixup_alpha=mixup_alpha, pos_weight=pos_weight
        )
        actual.sum().backward()
        assert_close(actual, expected, atol=1e-4, rtol=0)
        assert_close(logits.grad, expected_grad, atol=1e-4, rtol=0)

    def test_bce_mixup_invalid_pos_weight(self):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda")
        labels = torch.rand(B, C, device="cuda")
        seed = 0

        with pytest.raises(ValueError, match="pos_weight must be in range"):
            bce_mixup(logits, labels, seed=seed, pos_weight=-0.1)

        with pytest.raises(ValueError, match="pos_weight must be in range"):
            bce_mixup(logits, labels, seed=seed, pos_weight=1.1)
