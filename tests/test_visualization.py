import pytest
import torch

# Skip entire module if matplotlib is not available
pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import matplotlib.pyplot as plt

from mjepa.visualization.cosine import cosine_similarity_heatmap
from mjepa.visualization.norm import create_norm_histogram
from mjepa.visualization.pca import pca_topk
from mjepa.visualization.pos_enc import create_pos_enc_maps
from mjepa.visualization.runtime import plot_times


class TestPCATopK:
    """Test pca_topk function."""

    def test_basic_output_shape(self):
        """Test pca_topk returns correct output shape."""
        features = torch.randn(2, 8, 8, 64)  # (n, h, w, d)
        result = pca_topk(features, offset=0, k=3)

        assert result.shape == (2, 8, 8, 3)  # (n, h, w, k)

    def test_single_component(self):
        """Test pca_topk with single component."""
        features = torch.randn(1, 4, 4, 32)
        result = pca_topk(features, offset=0, k=1)

        assert result.shape == (1, 4, 4, 1)

    def test_offset_components(self):
        """Test pca_topk with offset."""
        features = torch.randn(2, 8, 8, 64)
        result_no_offset = pca_topk(features, offset=0, k=3)
        result_offset = pca_topk(features, offset=3, k=3)

        # Results should be different with different offsets
        assert not torch.allclose(result_no_offset, result_offset)

    def test_output_range(self):
        """Test that output is normalized to [0, 1] range."""
        features = torch.randn(2, 8, 8, 64)
        result = pca_topk(features, offset=0, k=3, normalize=["spatial"])

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_batch(self):
        """Test normalization across batch dimension."""
        features = torch.randn(4, 8, 8, 64)
        result = pca_topk(features, offset=0, k=1, normalize=["batch", "spatial"])

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_channel(self):
        """Test normalization across channel dimension."""
        features = torch.randn(2, 8, 8, 64)
        result = pca_topk(features, offset=0, k=3, normalize=["channel"])

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype(self):
        """Test that output is float32 regardless of input dtype."""
        features = torch.randn(2, 8, 8, 64, dtype=torch.float16)
        result = pca_topk(features, offset=0, k=3)

        assert result.dtype == torch.float32

    def test_deterministic(self):
        """Test that pca_topk is deterministic."""
        features = torch.randn(2, 8, 8, 64)
        result1 = pca_topk(features, offset=0, k=3)
        result2 = pca_topk(features, offset=0, k=3)

        assert torch.allclose(result1, result2)

    def test_k_greater_than_dims(self):
        """Test pca_topk when k approaches feature dimension."""
        features = torch.randn(2, 4, 4, 8)  # Small d
        result = pca_topk(features, offset=0, k=5)

        assert result.shape == (2, 4, 4, 5)


class TestCosineSimilarityHeatmap:
    """Test cosine_similarity_heatmap function."""

    def test_basic_output_shape(self):
        """Test cosine_similarity_heatmap returns correct output shape."""
        features = torch.randn(2, 8, 8, 64)  # (n, h, w, d)
        target_coords = [(4, 4)]
        result = cosine_similarity_heatmap(features, target_coords)

        assert result.shape == (2, 8, 8, 1)  # (n, h, w, num_targets)

    def test_multiple_coords(self):
        """Test with multiple target coordinates."""
        features = torch.randn(2, 8, 8, 64)
        target_coords = [(0, 0), (4, 4), (7, 7)]
        result = cosine_similarity_heatmap(features, target_coords)

        assert result.shape == (2, 8, 8, 3)

    def test_combine_coords(self):
        """Test combining multiple coordinates into single heatmap."""
        features = torch.randn(2, 8, 8, 64)
        target_coords = [(0, 0), (4, 4), (7, 7)]
        result = cosine_similarity_heatmap(features, target_coords, combine_coords=True)

        assert result.shape == (2, 8, 8, 1)

    def test_output_range(self):
        """Test that output is normalized to [0, 1] range."""
        features = torch.randn(2, 8, 8, 64)
        target_coords = [(4, 4)]
        result = cosine_similarity_heatmap(features, target_coords, normalize=["spatial"])

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_self_similarity(self):
        """Test that target coordinate has high similarity with itself."""
        features = torch.randn(1, 8, 8, 64)
        target_coords = [(4, 4)]
        result = cosine_similarity_heatmap(features, target_coords, normalize=["spatial"])

        # The target position should have relatively high similarity
        target_similarity = result[0, 4, 4, 0]
        assert target_similarity >= 0.5  # Should be normalized high

    def test_coords_clipped_to_bounds(self):
        """Test that out-of-bounds coordinates are clipped."""
        features = torch.randn(1, 8, 8, 64)
        # Coordinates outside bounds should be clipped
        target_coords = [(100, 100)]  # Way out of bounds
        result = cosine_similarity_heatmap(features, target_coords)

        assert result.shape == (1, 8, 8, 1)
        # Should not raise an error

    def test_output_dtype(self):
        """Test that output is float32."""
        features = torch.randn(2, 8, 8, 64, dtype=torch.float16)
        target_coords = [(4, 4)]
        result = cosine_similarity_heatmap(features, target_coords)

        assert result.dtype == torch.float32

    def test_normalize_batch(self):
        """Test batch normalization."""
        features = torch.randn(4, 8, 8, 64)
        target_coords = [(4, 4)]
        result = cosine_similarity_heatmap(features, target_coords, normalize=["batch", "spatial"])

        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestCreateNormHistogram:
    """Test create_norm_histogram function."""

    def test_returns_figure(self):
        """Test that create_norm_histogram returns a matplotlib Figure."""
        features = torch.randn(2, 64, 128)  # (n, num_tokens, d)
        register_tokens = torch.randn(2, 4, 128)  # (n, num_register_tokens, d)

        fig = create_norm_histogram(features, register_tokens)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_input_shapes(self):
        """Test with different input shapes."""
        features = torch.randn(4, 256, 64)
        register_tokens = torch.randn(4, 8, 64)

        fig = create_norm_histogram(features, register_tokens)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_batch(self):
        """Test with single batch."""
        features = torch.randn(1, 32, 64)
        register_tokens = torch.randn(1, 2, 64)

        fig = create_norm_histogram(features, register_tokens)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_axes(self):
        """Test that the figure has axes."""
        features = torch.randn(2, 64, 128)
        register_tokens = torch.randn(2, 4, 128)

        fig = create_norm_histogram(features, register_tokens)

        assert len(fig.axes) > 0
        plt.close(fig)


class TestCreatePosEncMaps:
    """Test create_pos_enc_maps function."""

    def test_returns_figure(self):
        """Test that create_pos_enc_maps returns a matplotlib Figure."""
        positions = torch.randn(8, 8, 64)  # (h, w, dim)

        fig = create_pos_enc_maps(positions)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_spatial_sizes(self):
        """Test with different spatial sizes."""
        positions = torch.randn(16, 16, 128)

        fig = create_pos_enc_maps(positions)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_small_spatial_size(self):
        """Test with small spatial size."""
        positions = torch.randn(4, 4, 32)

        fig = create_pos_enc_maps(positions)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_five_subplots(self):
        """Test that figure has 5 subplots (corners + center)."""
        positions = torch.randn(8, 8, 64)

        fig = create_pos_enc_maps(positions)

        # Should have 5 subplots: 4 corners + center
        assert len(fig.axes) == 5
        plt.close(fig)

    def test_non_square_spatial_size(self):
        """Test with non-square spatial dimensions."""
        positions = torch.randn(8, 16, 64)

        fig = create_pos_enc_maps(positions)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTimes:
    """Test plot_times function."""

    def test_returns_figure(self):
        """Test that plot_times returns a matplotlib Figure."""
        times = {
            (128, 128): [0.1, 0.11, 0.09, 0.1, 0.12],
            (256, 256): [0.3, 0.31, 0.29, 0.3, 0.32],
            (512, 512): [0.8, 0.81, 0.79, 0.8, 0.82],
        }
        device = torch.device("cpu")

        fig = plot_times(times, device, batch_size=1)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_size(self):
        """Test with single size."""
        times = {
            (256, 256): [0.1, 0.11, 0.09],
        }
        device = torch.device("cpu")

        fig = plot_times(times, device, batch_size=1)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_times_in_milliseconds(self):
        """Test with times that should be displayed in milliseconds."""
        times = {
            (128, 128): [0.01, 0.011, 0.009],
            (256, 256): [0.03, 0.031, 0.029],
        }
        device = torch.device("cpu")

        fig = plot_times(times, device, batch_size=1)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_times_in_seconds(self):
        """Test with times that should be displayed in seconds."""
        times = {
            (512, 512): [1.5, 1.6, 1.4],
            (1024, 1024): [5.0, 5.1, 4.9],
        }
        device = torch.device("cpu")

        fig = plot_times(times, device, batch_size=1)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        times = {
            (256, 256): [0.2, 0.21, 0.19],
        }
        device = torch.device("cpu")

        fig = plot_times(times, device, batch_size=8)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_axes(self):
        """Test that figure has axes."""
        times = {
            (128, 128): [0.1, 0.11, 0.09],
            (256, 256): [0.3, 0.31, 0.29],
        }
        device = torch.device("cpu")

        fig = plot_times(times, device, batch_size=1)

        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_non_square_sizes(self):
        """Test with non-square input sizes."""
        times = {
            (128, 256): [0.15, 0.16, 0.14],
            (256, 512): [0.45, 0.46, 0.44],
        }
        device = torch.device("cpu")

        fig = plot_times(times, device, batch_size=1)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
