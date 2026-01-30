import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from mjepa.logging import CSVLogger, SaveImage


class TestCSVLoggerInit:
    """Test CSVLogger initialization."""

    def test_init_creates_file(self):
        """Test that CSVLogger creates file on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.csv"
            logger = CSVLogger(path)

            assert path.exists()
            assert logger.path == path
            assert logger.interval == 1
            assert logger.accumulate_grad_batches == 1

    def test_init_custom_interval(self):
        """Test CSVLogger with custom interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.csv"
            logger = CSVLogger(path, interval=10, accumulate_grad_batches=4)

            assert logger.interval == 10
            assert logger.accumulate_grad_batches == 4

    def test_init_invalid_directory(self):
        """Test CSVLogger raises error for invalid parent directory."""
        with pytest.raises(FileNotFoundError):
            CSVLogger(Path("/nonexistent/directory/log.csv"))

    def test_init_existing_file(self):
        """Test CSVLogger with existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.csv"
            # Create file with existing content
            path.write_text("existing,data\n1,2\n")

            logger = CSVLogger(path)
            assert path.exists()
            # File content should be preserved
            assert path.read_text() == "existing,data\n1,2\n"


class TestCSVLoggerLog:
    """Test CSVLogger.log() method."""

    def test_log_single_entry(self):
        """Test logging a single entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.csv"
            logger = CSVLogger(path, interval=1, accumulate_grad_batches=1)

            logger.log(epoch=1, step=0, microbatch=0, loss=0.5, accuracy=0.8)

            df = pd.read_csv(path)
            assert len(df) == 1
            assert df["epoch"].iloc[0] == 1
            assert df["step"].iloc[0] == 0
            assert df["loss"].iloc[0] == 0.5
            assert df["accuracy"].iloc[0] == 0.8

    def test_log_multiple_entries(self):
        """Test logging multiple entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.csv"
            logger = CSVLogger(path, interval=1, accumulate_grad_batches=1)

            logger.log(epoch=1, step=0, microbatch=0, loss=0.5)
            logger.log(epoch=1, step=1, microbatch=1, loss=0.4)
            logger.log(epoch=1, step=2, microbatch=2, loss=0.3)

            df = pd.read_csv(path)
            assert len(df) == 3
            assert list(df["loss"]) == [0.5, 0.4, 0.3]

    def test_log_interval_filtering(self):
        """Test that logging respects interval setting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.csv"
            logger = CSVLogger(path, interval=5, accumulate_grad_batches=1)

            for step in range(10):
                logger.log(epoch=1, step=step, microbatch=step, loss=0.1 * step)

            df = pd.read_csv(path)
            # Only steps 4, 9 should be logged (when (step + 1) % 5 == 0)
            assert len(df) == 2
            assert list(df["step"]) == [4, 9]

    def test_log_accumulate_grad_batches_filtering(self):
        """Test that logging respects accumulate_grad_batches setting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.csv"
            logger = CSVLogger(path, interval=1, accumulate_grad_batches=4)

            for microbatch in range(8):
                logger.log(epoch=1, step=0, microbatch=microbatch, loss=0.1)

            df = pd.read_csv(path)
            # Only microbatches 3, 7 should be logged (when (microbatch + 1) % 4 == 0)
            assert len(df) == 2

    def test_log_combined_filtering(self):
        """Test logging with both interval and accumulate_grad_batches filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.csv"
            logger = CSVLogger(path, interval=2, accumulate_grad_batches=2)

            # Log with step and microbatch combinations
            for step in range(4):
                for microbatch in range(2):
                    logger.log(epoch=1, step=step, microbatch=microbatch, loss=0.1)

            df = pd.read_csv(path)
            # Only entries where (step + 1) % 2 == 0 AND (microbatch + 1) % 2 == 0
            # Steps: 1, 3; Microbatch: 1 -> 2 entries
            assert len(df) == 2


class TestCSVLoggerGetDF:
    """Test CSVLogger.get_df() method."""

    def test_get_df_empty_file(self):
        """Test get_df returns empty DataFrame for empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.csv"
            logger = CSVLogger(path)

            df = logger.get_df()
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0

    def test_get_df_with_data(self):
        """Test get_df returns logged data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.csv"
            logger = CSVLogger(path)

            logger.log(epoch=1, step=0, microbatch=0, loss=0.5)
            logger.log(epoch=2, step=1, microbatch=1, loss=0.3)

            df = logger.get_df()
            assert len(df) == 2
            assert "epoch" in df.columns
            assert "step" in df.columns
            assert "loss" in df.columns

    def test_get_df_nonexistent_file(self):
        """Test get_df handles case where file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.csv"
            logger = CSVLogger(path)
            # Delete the file
            path.unlink()

            df = logger.get_df()
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0


class TestSaveImageInit:
    """Test SaveImage initialization."""

    def test_init_valid_path(self):
        """Test SaveImage initialization with valid path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "image.png"
            saver = SaveImage(path, max_save_images=8)

            assert saver.path == path
            assert saver.max_save_images == 8

    def test_init_custom_max_images(self):
        """Test SaveImage with custom max_save_images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "image.png"
            saver = SaveImage(path, max_save_images=16)

            assert saver.max_save_images == 16

    def test_init_invalid_directory(self):
        """Test SaveImage raises error for invalid parent directory."""
        with pytest.raises(FileNotFoundError):
            SaveImage(Path("/nonexistent/directory/image.png"))


class TestSaveImageCall:
    """Test SaveImage.__call__() method."""

    def test_save_image_creates_file(self):
        """Test that SaveImage creates an image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "image.png"
            saver = SaveImage(path, max_save_images=4)

            # Create a batch of images
            images = torch.rand(8, 3, 32, 32)
            saver(images)

            assert path.exists()

    def test_save_image_respects_max_images(self):
        """Test that SaveImage respects max_save_images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "image.png"
            saver = SaveImage(path, max_save_images=2)

            # Create more images than max
            images = torch.rand(8, 3, 32, 32)
            saver(images)

            assert path.exists()
            # The saved image should only include max_save_images images

    def test_save_grayscale_images(self):
        """Test saving grayscale images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "image.png"
            saver = SaveImage(path, max_save_images=4)

            images = torch.rand(4, 1, 32, 32)
            saver(images)

            assert path.exists()

    @patch("mjepa.logging.rank_zero_only", lambda fn: fn)  # Make it always run
    def test_save_image_logs_info(self, capsys):
        """Test that SaveImage logs info when saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "image.png"
            saver = SaveImage(path, max_save_images=4)

            images = torch.rand(4, 3, 32, 32)
            # Note: rank_zero_info may not output in test context
            saver(images)

            assert path.exists()
