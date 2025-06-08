from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from torchvision.transforms.v2 import RandomHorizontalFlip
from torchvision.tv_tensors import Image as TVImage
from torchvision.tv_tensors import Video as TVVideo

from mjepa.data.tiff import PreprocessedTIFFDataset, load_and_wrap_pixels


@pytest.fixture
def mock_tiff_data():
    """Create mock TIFF data for testing."""
    # Create a 2D image (1, H, W, C)
    single_frame = np.random.randn(1, 10, 12, 3).astype(np.float32)

    # Create a 3D volume (N, H, W, C)
    multi_frame = np.random.randn(5, 10, 12, 3).astype(np.float32)

    return {"single_frame": single_frame, "multi_frame": multi_frame}


@pytest.fixture
def tiff_path(tmp_path: Path, mock_tiff_data: dict) -> Path:
    """Create a test TIFF file path."""
    filepath = tmp_path / "test.tiff"
    # Create an empty file to avoid FileNotFoundError
    filepath.touch()
    return filepath


@pytest.fixture
def preprocessed_root(tmp_path: Path) -> Path:
    """Create a test directory structure with a manifest for PreprocessedTIFFDataset."""
    num_studies = 3
    images_per_study = 4

    # Create directories and dummy files
    manifest_data = []
    for i in range(num_studies):
        study_id = f"study_{i}"
        study_uid = f"study_uid_{i}"
        study_path = tmp_path / study_id
        study_path.mkdir(parents=True, exist_ok=True)

        for j in range(images_per_study):
            sop_id = f"sop_{i * images_per_study + j}"
            sop_uid = f"sop_uid_{i * images_per_study + j}"
            rel_path = f"{study_id}/{sop_id}.tiff"
            image_path = tmp_path / rel_path

            # Create an empty file (content doesn't matter since we mock the loader)
            image_path.touch()

            # Add entry to manifest
            manifest_data.append(
                {
                    "path": rel_path,
                    "study_instance_uid": study_uid,
                    "sop_instance_uid": sop_uid,
                    "num_frames": 1 if j % 2 == 0 else 5,  # Alternate between 2D and 3D
                    "inode": i * images_per_study + j,  # Use simple inode for sorting
                }
            )

    # Create manifest file
    manifest_df = pd.DataFrame(manifest_data)
    manifest_path = tmp_path / "manifest.parquet"

    # Make sure pandas is available and manifest is properly saved
    try:
        manifest_df.to_parquet(manifest_path)
        # Verify the file was created
        if not manifest_path.exists():
            raise FileNotFoundError(f"Failed to create manifest file at {manifest_path}")
    except Exception as e:
        # For debugging: print the error
        print(f"Error creating manifest: {e}")
        raise

    return tmp_path


def test_load_and_wrap_pixels_2d(tiff_path, mock_tiff_data):
    """Test loading a 2D image and wrapping it as a TVImage."""
    single_frame = mock_tiff_data["single_frame"]

    with patch("mjepa.data.tiff.load_tiff_f32", return_value=single_frame):
        result = load_and_wrap_pixels(tiff_path)

        # Check it's the right type
        assert isinstance(result, TVImage)

        # Check dimensions: should be C,H,W from N,H,W,C
        assert result.shape == (3, 10, 12)

        # Check it has the correct data (allowing for small floating point differences)
        expected = torch.from_numpy(single_frame[0].transpose(2, 0, 1))  # C,H,W
        assert torch.allclose(result, expected)


def test_load_and_wrap_pixels_3d(tiff_path, mock_tiff_data):
    """Test loading a 3D volume and wrapping it as a TVVideo."""
    multi_frame = mock_tiff_data["multi_frame"]

    with patch("mjepa.data.tiff.load_tiff_f32", return_value=multi_frame):
        result = load_and_wrap_pixels(tiff_path)

        # Check it's the right type
        assert isinstance(result, TVVideo)

        # Check dimensions: should be N,C,H,W from N,H,W,C
        assert result.shape == (5, 3, 10, 12)

        # Verify the data matches
        expected = torch.from_numpy(multi_frame.transpose(0, 3, 1, 2))  # N,C,H,W
        assert torch.allclose(result, expected)


def test_load_and_wrap_pixels_frames_selection(tiff_path, mock_tiff_data):
    """Test loading specific frames from a volume."""
    multi_frame = mock_tiff_data["multi_frame"]
    frames = [1, 3]  # Select specific frames

    # Mock load_tiff_f32 to verify frames parameter is passed correctly
    mock_loader = MagicMock(return_value=multi_frame[frames])

    with patch("mjepa.data.tiff.load_tiff_f32", mock_loader):
        result = load_and_wrap_pixels(tiff_path, frames)

        # Verify the correct frames parameter was passed
        mock_loader.assert_called_once_with(str(tiff_path), frames)

        # Check it's the right type
        assert isinstance(result, TVVideo)

        # Check dimensions
        assert result.shape == (2, 3, 10, 12)  # 2 frames, 3 channels, 10x12 size


def test_load_and_wrap_pixels_file_not_found(tmp_path):
    """Test FileNotFoundError is raised when the file doesn't exist."""
    nonexistent_path = tmp_path / "nonexistent.tiff"

    with pytest.raises(FileNotFoundError):
        load_and_wrap_pixels(nonexistent_path)


def test_preprocessed_dataset_init(preprocessed_root):
    """Test initialization of PreprocessedTIFFDataset."""
    dataset = PreprocessedTIFFDataset(preprocessed_root)

    assert len(dataset) == 12  # 3 studies * 4 images
    assert dataset.training is False
    assert dataset.keep_volume is False


def test_preprocessed_dataset_properties(preprocessed_root):
    """Test the properties of PreprocessedTIFFDataset."""
    dataset = PreprocessedTIFFDataset(preprocessed_root)

    # Check study_uids
    study_uids = list(dataset.study_uids)
    assert len(study_uids) == 3
    assert all(f"study_uid_{i}" in study_uids for i in range(3))

    # Check sop_uids
    sop_uids = list(dataset.sop_uids)
    assert len(sop_uids) == 12
    assert all(f"sop_uid_{i}" in sop_uids for i in range(12))


def test_preprocessed_dataset_select_frames(preprocessed_root):
    """Test the _select_frames method for different configurations."""
    # 2D image (1 frame)
    dataset_default = PreprocessedTIFFDataset(preprocessed_root)
    assert dataset_default._select_frames(1) is None

    # 3D image, keep volume
    dataset_keep_volume = PreprocessedTIFFDataset(preprocessed_root, keep_volume=True)
    assert dataset_keep_volume._select_frames(5) is None

    # 3D image, training, don't keep volume
    dataset_training = PreprocessedTIFFDataset(preprocessed_root, training=True)
    with patch("torch.randint", return_value=torch.tensor([2])):
        frames = dataset_training._select_frames(5)
        assert frames == [2]

    # 3D image, not training, don't keep volume
    dataset_not_training = PreprocessedTIFFDataset(preprocessed_root)
    frames = dataset_not_training._select_frames(5)
    assert frames == [2]  # middle frame (5//2)


@pytest.mark.parametrize(
    "training,keep_volume,expected_type",
    [
        (False, False, TVImage),  # Default: Test time, don't keep volume -> return middle slice
        (True, False, TVImage),  # Training, don't keep volume -> return random slice
        (False, True, TVVideo),  # Test time, keep volume -> return all frames
        (True, True, TVVideo),  # Training, keep volume -> return all frames
    ],
)
def test_preprocessed_dataset_getitem(preprocessed_root, mock_tiff_data, training, keep_volume, expected_type):
    """Test the __getitem__ method for different configurations."""
    # Create the dataset with the test configuration
    dataset = PreprocessedTIFFDataset(preprocessed_root, training=training, keep_volume=keep_volume)

    # Get the first item (which is a 2D image according to our manifest)
    with patch(
        "mjepa.data.tiff.load_tiff_f32",
        return_value=mock_tiff_data["single_frame" if expected_type == TVImage else "multi_frame"],
    ):
        result = dataset[0]

        # Check the type
        assert isinstance(result, expected_type)


def test_preprocessed_dataset_transform(preprocessed_root, mock_tiff_data):
    """Test that transforms are applied correctly."""
    # Create a transform
    transform = RandomHorizontalFlip(p=1.0)  # Always flip

    # Create dataset with the transform
    dataset = PreprocessedTIFFDataset(preprocessed_root, transform=transform)

    # Create a mock image with a distinct pattern to verify flipping
    test_data = np.zeros((1, 10, 12, 3), dtype=np.float32)
    test_data[:, :, 0, :] = 1.0  # Set first column to 1

    with patch("mjepa.data.tiff.load_tiff_f32", return_value=test_data):
        # Get an image
        result = dataset[0]

        # The image should be flipped (last column should now be 1.0)
        assert torch.all(result[..., -1] == 1.0)
        assert torch.all(result[..., :-1] < 1.0)


def test_preprocessed_dataset_nonexistent_directory():
    """Test that an error is raised for a nonexistent directory."""
    with pytest.raises(NotADirectoryError):
        PreprocessedTIFFDataset(Path("/nonexistent/directory"))


def test_preprocessed_dataset_no_manifest(tmp_path):
    """Test that an error is raised when there's no manifest file."""
    with pytest.raises(FileNotFoundError):
        PreprocessedTIFFDataset(tmp_path)
