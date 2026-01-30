from argparse import ArgumentParser

import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt
import torch

pytest.importorskip("dicom_preprocessing")

from mjepa.visualization.cosine import coordinate_type, cosine_similarity_heatmap
from mjepa.visualization.norm import create_norm_histogram
from mjepa.visualization.pca import ExpandImagePathsAction, existing_file_type, output_path_type, pca_topk, torch_dtype_type
from mjepa.visualization.pos_enc import create_pos_enc_maps
from mjepa.visualization.runtime import plot_times


def test_pca_topk_shape_and_range():
    features = torch.randn(2, 4, 4, 6)
    output = pca_topk(features, offset=0, k=2)

    assert output.shape == (2, 4, 4, 2)
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_cosine_similarity_heatmap_shapes_and_range():
    features = torch.randn(1, 3, 3, 4)
    heatmaps = cosine_similarity_heatmap(features, target_coords=[(0, 0), (10, 10)])

    assert heatmaps.shape == (1, 3, 3, 2)
    assert heatmaps.min() >= 0.0
    assert heatmaps.max() <= 1.0

    combined = cosine_similarity_heatmap(features, target_coords=[(0, 0), (10, 10)], combine_coords=True)
    assert combined.shape == (1, 3, 3, 1)


def test_coordinate_type():
    assert coordinate_type("3,5") == (3, 5)
    with pytest.raises(ValueError):
        coordinate_type("bad")


def test_torch_dtype_type():
    assert torch_dtype_type("fp32") == torch.float32
    assert torch_dtype_type("bf16") == torch.bfloat16
    with pytest.raises(ValueError):
        torch_dtype_type("invalid")


def test_existing_file_type_and_output_path_type(tmp_path):
    existing = tmp_path / "file.txt"
    existing.write_text("data")
    assert existing_file_type(str(existing)) == existing

    output = tmp_path / "out.png"
    assert output_path_type(str(output)) == output

    with pytest.raises(FileNotFoundError):
        existing_file_type(str(tmp_path / "missing.txt"))

    with pytest.raises(NotADirectoryError):
        output_path_type(str(tmp_path / "missing" / "out.png"))


def test_expand_image_paths_action_with_directory(tmp_path):
    (tmp_path / "a.tiff").write_text("data")
    (tmp_path / "b.tiff").write_text("data")

    parser = ArgumentParser()
    parser.add_argument("input", nargs="+", action=ExpandImagePathsAction)
    ns = parser.parse_args([str(tmp_path)])

    assert len(ns.input) == 2


def test_create_pos_enc_maps_and_norm_histogram():
    positions = torch.randn(2, 2, 4)
    fig = create_pos_enc_maps(positions)
    assert fig is not None
    plt.close(fig)

    features = torch.randn(2, 10, 4)
    register_tokens = torch.randn(2, 2, 4)
    fig = create_norm_histogram(features, register_tokens)
    assert fig is not None
    plt.close(fig)


def test_plot_times_returns_figure():
    times = {(32, 32): [0.1, 0.2], (64, 64): [0.3, 0.25]}
    fig = plot_times(times, torch.device("cpu"), batch_size=2)
    assert fig is not None
    plt.close(fig)
