import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from vit import ViTConfig

from mjepa.jepa import CrossAttentionPredictor
from mjepa.trainer import (
    ResolutionConfig,
    TrainerConfig,
    calculate_total_steps,
    count_parameters,
    format_large_number,
    format_pbar_description,
    is_rank_zero,
    rank_zero_only,
    save_checkpoint,
    seed_everything,
    should_step_optimizer,
)


class TestSeedEverything:
    """Test seed_everything function for reproducibility."""

    def test_seed_reproducibility_torch(self):
        """Test that seeding produces reproducible torch random numbers."""
        seed_everything(42)
        rand1 = torch.rand(10)

        seed_everything(42)
        rand2 = torch.rand(10)

        assert torch.allclose(rand1, rand2)

    def test_seed_reproducibility_python_random(self):
        """Test that seeding produces reproducible Python random numbers."""
        seed_everything(42)
        rand1 = [random.random() for _ in range(10)]

        seed_everything(42)
        rand2 = [random.random() for _ in range(10)]

        assert rand1 == rand2

    def test_seed_reproducibility_numpy(self):
        """Test that seeding produces reproducible NumPy random numbers."""
        seed_everything(42)
        rand1 = np.random.rand(10)

        seed_everything(42)
        rand2 = np.random.rand(10)

        assert np.allclose(rand1, rand2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different random numbers."""
        seed_everything(42)
        rand1 = torch.rand(10)

        seed_everything(123)
        rand2 = torch.rand(10)

        assert not torch.allclose(rand1, rand2)


class TestShouldStepOptimizer:
    """Test should_step_optimizer function."""

    @pytest.mark.parametrize(
        "microbatch,accumulate,expected",
        [
            (0, 1, True),  # First step with no accumulation
            (1, 1, True),  # Second step with no accumulation
            (0, 2, False),  # First step with 2x accumulation
            (1, 2, True),  # Second step with 2x accumulation
            (2, 2, False),  # Third step with 2x accumulation
            (3, 2, True),  # Fourth step with 2x accumulation
            (0, 4, False),
            (1, 4, False),
            (2, 4, False),
            (3, 4, True),  # Step every 4th microbatch
            (7, 4, True),
        ],
    )
    def test_should_step_optimizer(self, microbatch, accumulate, expected):
        """Test should_step_optimizer with various configurations."""
        result = should_step_optimizer(microbatch, accumulate)
        assert result == expected


class TestCalculateTotalSteps:
    """Test calculate_total_steps function."""

    def test_basic_calculation(self):
        """Test basic total steps calculation."""
        # Mock dataloader with __len__
        dataloader = MagicMock()
        dataloader.__len__ = MagicMock(return_value=100)

        total_steps = calculate_total_steps(dataloader, num_epochs=10, accumulate_grad_batches=1)
        assert total_steps == 1000  # 10 * 100 / 1

    def test_with_accumulation(self):
        """Test total steps with gradient accumulation."""
        dataloader = MagicMock()
        dataloader.__len__ = MagicMock(return_value=100)

        total_steps = calculate_total_steps(dataloader, num_epochs=10, accumulate_grad_batches=4)
        assert total_steps == 250  # 10 * 100 / 4


class TestCountParameters:
    """Test count_parameters function."""

    def test_count_all_parameters(self):
        """Test counting all parameters."""
        model = nn.Linear(10, 20)  # 10*20 + 20 = 220 params
        count = count_parameters(model, trainable_only=False)
        assert count == 220

    def test_count_trainable_only(self):
        """Test counting only trainable parameters."""
        model = nn.Sequential(
            nn.Linear(10, 20),  # 220 params
            nn.Linear(20, 10),  # 210 params
        )
        # Freeze second layer
        for param in model[1].parameters():
            param.requires_grad = False

        trainable = count_parameters(model, trainable_only=True)
        all_params = count_parameters(model, trainable_only=False)

        assert trainable == 220
        assert all_params == 430

    def test_nested_module(self):
        """Test counting parameters in nested modules."""
        model = nn.Sequential(
            nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10)),
            nn.Linear(10, 5),
        )
        count = count_parameters(model, trainable_only=False)
        expected = (10 * 10 + 10) * 2 + (10 * 5 + 5)  # 220 + 55 = 275
        assert count == expected


class TestFormatLargeNumber:
    """Test format_large_number function."""

    @pytest.mark.parametrize(
        "number,expected_suffix",
        [
            (100, ""),
            (999, ""),
            (1000, "K"),
            (999999, "K"),
            (1000000, "M"),
            (999999999, "M"),
            (1000000000, "B"),
            (5000000000, "B"),
        ],
    )
    def test_format_suffix(self, number, expected_suffix):
        """Test that numbers get correct suffix."""
        result = format_large_number(number)
        if expected_suffix:
            assert result.endswith(expected_suffix)
        else:
            assert result == str(number)

    def test_format_specific_values(self):
        """Test specific formatted values."""
        assert format_large_number(500) == "500"
        assert "1.0000K" == format_large_number(1000)
        assert "1.2000K" == format_large_number(1200)
        assert "1.0000M" == format_large_number(1000000)
        assert "1.5000M" == format_large_number(1500000)
        assert "1.0000B" == format_large_number(1000000000)


class TestIsRankZero:
    """Test is_rank_zero function."""

    def test_not_initialized(self):
        """Test is_rank_zero when distributed is not initialized."""
        # When dist is not initialized, should return True
        result = is_rank_zero()
        assert result is True

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    def test_rank_zero(self, mock_get_rank, mock_is_init):
        """Test is_rank_zero when rank is 0."""
        mock_is_init.return_value = True
        mock_get_rank.return_value = 0

        result = is_rank_zero()
        assert result is True

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    def test_non_zero_rank(self, mock_get_rank, mock_is_init):
        """Test is_rank_zero when rank is not 0."""
        mock_is_init.return_value = True
        mock_get_rank.return_value = 1

        result = is_rank_zero()
        assert result is False


class TestRankZeroOnly:
    """Test rank_zero_only decorator."""

    def test_decorator_runs_when_not_distributed(self):
        """Test decorator runs function when distributed is not initialized."""
        call_count = [0]

        @rank_zero_only
        def test_func():
            call_count[0] += 1
            return "executed"

        result = test_func()
        assert call_count[0] == 1
        assert result == "executed"

    @patch("mjepa.trainer.is_rank_zero")
    def test_decorator_skips_non_zero_rank(self, mock_is_rank_zero):
        """Test decorator skips function when not rank zero."""
        mock_is_rank_zero.return_value = False
        call_count = [0]

        @rank_zero_only
        def test_func():
            call_count[0] += 1
            return "executed"

        result = test_func()
        assert call_count[0] == 0
        assert result is None


class TestResolutionConfig:
    """Test ResolutionConfig dataclass."""

    def test_init(self):
        """Test ResolutionConfig initialization."""
        config = ResolutionConfig(size=(256, 256), batch_size=32)
        assert config.size == (256, 256)
        assert config.batch_size == 32

    def test_different_sizes(self):
        """Test ResolutionConfig with different sizes."""
        config = ResolutionConfig(size=[128, 256], batch_size=64)
        assert list(config.size) == [128, 256]
        assert config.batch_size == 64


class TestTrainerConfig:
    """Test TrainerConfig dataclass."""

    def test_default_values(self):
        """Test TrainerConfig default values."""
        config = TrainerConfig(
            batch_size=32,
            num_workers=4,
            num_epochs=100,
        )
        assert config.batch_size == 32
        assert config.num_workers == 4
        assert config.num_epochs == 100
        assert config.accumulate_grad_batches == 1
        assert config.log_interval == 50
        assert config.check_val_every_n_epoch == 1
        assert config.sizes == {}

    def test_custom_values(self):
        """Test TrainerConfig with custom values."""
        sizes = {
            10: ResolutionConfig(size=(128, 128), batch_size=64),
            20: ResolutionConfig(size=(256, 256), batch_size=32),
        }
        config = TrainerConfig(
            batch_size=32,
            num_workers=8,
            num_epochs=200,
            accumulate_grad_batches=4,
            log_interval=100,
            check_val_every_n_epoch=5,
            sizes=sizes,
        )
        assert config.batch_size == 32
        assert config.num_workers == 8
        assert config.num_epochs == 200
        assert config.accumulate_grad_batches == 4
        assert config.log_interval == 100
        assert config.check_val_every_n_epoch == 5
        assert len(config.sizes) == 2


class TestTrainerConfigSizeChange:
    """Test TrainerConfig size change detection methods."""

    @pytest.fixture
    def config_with_sizes(self):
        """Create TrainerConfig with size changes at epochs 10 and 20."""
        sizes = {
            10: ResolutionConfig(size=(128, 128), batch_size=64),
            20: ResolutionConfig(size=(256, 256), batch_size=32),
        }
        return TrainerConfig(
            batch_size=32,
            num_workers=4,
            num_epochs=100,
            sizes=sizes,
        )

    def test_is_size_change_epoch_true(self, config_with_sizes):
        """Test is_size_change_epoch returns True at size change epochs."""
        assert config_with_sizes.is_size_change_epoch(10) is True
        assert config_with_sizes.is_size_change_epoch(20) is True

    def test_is_size_change_epoch_false(self, config_with_sizes):
        """Test is_size_change_epoch returns False at non-change epochs."""
        assert config_with_sizes.is_size_change_epoch(0) is False
        assert config_with_sizes.is_size_change_epoch(5) is False
        assert config_with_sizes.is_size_change_epoch(15) is False
        assert config_with_sizes.is_size_change_epoch(25) is False

    def test_get_size_for_epoch_before_any_change(self, config_with_sizes):
        """Test get_size_for_epoch before any size change."""
        result = config_with_sizes.get_size_for_epoch(5)
        assert result is None

    def test_get_size_for_epoch_at_first_change(self, config_with_sizes):
        """Test get_size_for_epoch at first size change."""
        result = config_with_sizes.get_size_for_epoch(10)
        assert result is not None
        assert result.size == (128, 128)
        assert result.batch_size == 64

    def test_get_size_for_epoch_between_changes(self, config_with_sizes):
        """Test get_size_for_epoch between size changes."""
        result = config_with_sizes.get_size_for_epoch(15)
        assert result is not None
        assert result.size == (128, 128)
        assert result.batch_size == 64

    def test_get_size_for_epoch_at_second_change(self, config_with_sizes):
        """Test get_size_for_epoch at second size change."""
        result = config_with_sizes.get_size_for_epoch(20)
        assert result is not None
        assert result.size == (256, 256)
        assert result.batch_size == 32

    def test_get_size_for_epoch_after_all_changes(self, config_with_sizes):
        """Test get_size_for_epoch after all size changes."""
        result = config_with_sizes.get_size_for_epoch(50)
        assert result is not None
        assert result.size == (256, 256)
        assert result.batch_size == 32

    def test_empty_sizes(self):
        """Test with no size changes configured."""
        config = TrainerConfig(batch_size=32, num_workers=4, num_epochs=100)
        assert config.is_size_change_epoch(10) is False
        assert config.get_size_for_epoch(10) is None


class TestSaveCheckpoint:
    """Test save_checkpoint function."""

    @pytest.fixture
    def vit_config(self):
        """Create a small ViT configuration."""
        return ViTConfig(
            in_channels=3,
            hidden_size=64,
            patch_size=[4, 4],
            img_size=[32, 32],
            depth=2,
            num_attention_heads=4,
            ffn_hidden_size=128,
            pos_enc="rope",
        )

    @pytest.fixture
    def checkpoint_components(self, vit_config):
        """Create components for checkpoint testing."""
        backbone = vit_config.instantiate()
        predictor = CrossAttentionPredictor(backbone, depth=2, out_dim=None)
        teacher = vit_config.instantiate()

        optimizer = AdamW(backbone.parameters(), lr=1e-3)
        scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=100)

        return backbone, predictor, teacher, optimizer, scheduler

    def test_save_checkpoint_creates_file(self, checkpoint_components):
        """Test that save_checkpoint creates a file."""
        backbone, predictor, teacher, optimizer, scheduler = checkpoint_components

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(
                path,
                backbone=backbone,
                predictor=predictor,
                teacher=teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                step=100,
                epoch=5,
            )
            assert path.exists()

    def test_save_checkpoint_contents(self, checkpoint_components):
        """Test that saved checkpoint contains expected keys."""
        backbone, predictor, teacher, optimizer, scheduler = checkpoint_components

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(
                path,
                backbone=backbone,
                predictor=predictor,
                teacher=teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                step=100,
                epoch=5,
            )

            checkpoint = torch.load(path, weights_only=False)
            assert "backbone" in checkpoint
            assert "predictor" in checkpoint
            assert "teacher" in checkpoint
            assert "optimizer" in checkpoint
            assert "scheduler" in checkpoint
            assert checkpoint["step"] == 100
            assert checkpoint["epoch"] == 5

    def test_save_checkpoint_invalid_directory(self, checkpoint_components):
        """Test save_checkpoint raises error for invalid directory."""
        backbone, predictor, teacher, optimizer, scheduler = checkpoint_components

        with pytest.raises(NotADirectoryError):
            save_checkpoint(
                Path("/nonexistent/directory/checkpoint.pt"),
                backbone=backbone,
                predictor=predictor,
                teacher=teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                step=100,
                epoch=5,
            )

    def test_save_checkpoint_none_components(self, checkpoint_components):
        """Test save_checkpoint with None predictor/teacher."""
        backbone, _, _, optimizer, scheduler = checkpoint_components

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(
                path,
                backbone=backbone,
                predictor=None,
                teacher=None,
                optimizer=optimizer,
                scheduler=scheduler,
                step=50,
                epoch=2,
            )

            checkpoint = torch.load(path, weights_only=False)
            assert checkpoint["predictor"] is None
            assert checkpoint["teacher"] is None


class TestFormatPbarDescription:
    """Test format_pbar_description function."""

    def test_basic_format(self):
        """Test basic progress bar description formatting."""
        import torchmetrics as tm

        loss_metric = tm.MeanMetric()
        loss_metric.update(0.5)

        desc = format_pbar_description(
            step=100,
            microbatch=50,
            epoch=5,
            loss=loss_metric,
        )

        assert "Epoch: 5" in desc
        assert "Step:" in desc
        assert "Microbatch:" in desc
        assert "loss=" in desc


class TestYAMLConstructors:
    """Test YAML constructors for trainer configs."""

    def test_trainer_config_yaml(self):
        """Test loading TrainerConfig from YAML."""
        yaml_str = """
!!python/object:mjepa.TrainerConfig
batch_size: 32
num_workers: 4
num_epochs: 100
accumulate_grad_batches: 2
log_interval: 25
"""
        config = yaml.safe_load(yaml_str)
        assert isinstance(config, TrainerConfig)
        assert config.batch_size == 32
        assert config.num_workers == 4
        assert config.num_epochs == 100
        assert config.accumulate_grad_batches == 2
        assert config.log_interval == 25

    def test_resolution_config_yaml(self):
        """Test loading ResolutionConfig from YAML."""
        yaml_str = """
!!python/object:mjepa.ResolutionConfig
size:
  - 256
  - 256
batch_size: 16
"""
        config = yaml.safe_load(yaml_str)
        assert isinstance(config, ResolutionConfig)
        assert list(config.size) == [256, 256]
        assert config.batch_size == 16
