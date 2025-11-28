import math

import pytest
import torch
import torch.nn as nn
import yaml

from mjepa.jepa import (
    JEPAConfig,
    compute_sigreg_loss,
    config_constructor,
    get_momentum,
    register_constructors,
    update_teacher,
)


class TestUpdateTeacher:
    def test_update_teacher_with_momentum(self):
        """Test teacher update with non-zero momentum."""
        student = nn.Linear(10, 10)
        teacher = nn.Linear(10, 10)

        # Initialize with different weights
        student.weight.data.fill_(1.0)
        teacher.weight.data.fill_(0.0)

        momentum = 0.9
        update_teacher(student, teacher, momentum)

        # Weight should be updated with (1-momentum) * student_weight
        expected = 0.1  # (1-0.9) * 1.0
        assert torch.allclose(teacher.weight.data, torch.full_like(teacher.weight.data, expected))

    def test_update_teacher_momentum_one(self):
        """Test teacher update with momentum=1.0, should be no-op."""
        student = nn.Linear(10, 10)
        teacher = nn.Linear(10, 10)

        # Initialize with different weights
        student.weight.data.fill_(1.0)
        teacher.weight.data.fill_(0.0)

        momentum = 1.0
        update_teacher(student, teacher, momentum)

        # Teacher weights should remain unchanged
        assert torch.allclose(teacher.weight.data, torch.zeros_like(teacher.weight.data))

    def test_update_teacher_invalid_momentum(self):
        """Test that invalid momentum values raise an assertion error."""
        student = nn.Linear(10, 10)
        teacher = nn.Linear(10, 10)

        with pytest.raises(AssertionError):
            update_teacher(student, teacher, -0.1)

        with pytest.raises(AssertionError):
            update_teacher(student, teacher, 1.1)


class TestGetMomentum:
    def test_get_momentum_start(self):
        """Test momentum at start of training."""
        assert get_momentum(0, 100, 0.9) == 0.9

    @pytest.mark.parametrize(
        "scheduled,expected",
        [
            (True, 0.95),
            (False, 0.9),
        ],
    )
    def test_get_momentum_middle(self, scheduled, expected):
        """Test momentum in middle of training."""
        assert get_momentum(50, 100, 0.9, scheduled) == expected

    @pytest.mark.parametrize(
        "scheduled,expected",
        [
            (True, 1.0),
            (False, 0.9),
        ],
    )
    def test_get_momentum_end(self, scheduled, expected):
        """Test momentum at end of training."""
        assert get_momentum(100, 100, 0.9, scheduled) == expected

    def test_get_momentum_interpolation(self):
        """Test momentum interpolation at arbitrary step."""
        acutal = get_momentum(25, 100, 0.8, scheduled=True)
        assert math.isclose(acutal, 0.85, rel_tol=1e-6)


class TestJEPAConfig:
    def test_default_config(self):
        """Test default configuration."""
        config = JEPAConfig()
        assert config.context_ratio == 0.5
        assert config.target_ratio == 0.25
        assert config.scale == 4
        assert config.momentum == 0.99
        assert config.predictor_depth == 4

    def test_custom_config(self):
        """Test custom configuration."""
        config = JEPAConfig(
            context_ratio=0.6,
            target_ratio=0.3,
            scale=8,
            momentum=0.95,
            predictor_depth=6,
        )
        assert config.context_ratio == 0.6
        assert config.target_ratio == 0.3
        assert config.scale == 8
        assert config.momentum == 0.95
        assert config.predictor_depth == 6

    def test_invalid_context_ratio(self):
        """Test invalid context ratio."""
        with pytest.raises(ValueError):
            JEPAConfig(context_ratio=0)

        with pytest.raises(ValueError):
            JEPAConfig(context_ratio=1.1)

    def test_invalid_target_ratio(self):
        """Test invalid target ratio."""
        with pytest.raises(ValueError):
            JEPAConfig(target_ratio=0)

        with pytest.raises(ValueError):
            JEPAConfig(target_ratio=1.1)


class TestYAMLConfig:
    def test_config_constructor(self, mocker):
        """Test the config_constructor function."""
        # Create mock loader and node
        loader = mocker.Mock()
        node = mocker.Mock()

        # Configure the loader to return a dictionary
        config_dict = {
            "context_ratio": 0.7,
            "target_ratio": 0.4,
            "scale": 16,
            "momentum": 0.98,
            "predictor_depth": 8,
        }
        loader.construct_mapping.return_value = config_dict

        # Call the constructor
        config = config_constructor(loader, node)

        # Verify the returned config
        assert isinstance(config, JEPAConfig)
        assert config.context_ratio == 0.7
        assert config.target_ratio == 0.4
        assert config.scale == 16
        assert config.momentum == 0.98
        assert config.predictor_depth == 8

        # Verify the loader was called correctly
        loader.construct_mapping.assert_called_once_with(node, deep=True)

    def test_yaml_load(self):
        """Test loading a JEPAConfig from YAML."""
        # Register constructors
        register_constructors()

        # Create a YAML string
        yaml_str = """
        !!python/object:mjepa.JEPAConfig
        context_ratio: 0.8
        target_ratio: 0.3
        scale: 12
        momentum: 0.97
        predictor_depth: 5
        """

        # Load the YAML
        config = yaml.safe_load(yaml_str)

        # Verify the loaded config
        assert isinstance(config, JEPAConfig)
        assert config.context_ratio == 0.8
        assert config.target_ratio == 0.3
        assert config.scale == 12
        assert config.momentum == 0.97
        assert config.predictor_depth == 5


class TestComputeSigREGLoss:

    @pytest.mark.parametrize(
        "x_shape",
        [
            (1, 32, 128),
            (2, 16, 128),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_compute_sigreg_loss(self, x_shape, dtype):
        """Test the compute_sigreg_loss function."""
        x = torch.randn(*x_shape, dtype=dtype)
        global_step = 0
        num_slices = 256
        loss = compute_sigreg_loss(x, global_step, num_slices)
        assert loss.shape == ()
        assert loss.item() > 0
        assert not torch.isnan(loss)

    @pytest.mark.parametrize(
        "x_shape",
        [
            (1, 32, 128),
            (2, 16, 128),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_compute_sigreg_loss_deterministic(self, x_shape, dtype):
        """Test the compute_sigreg_loss function."""
        x = torch.randn(*x_shape, dtype=dtype)
        global_step = 0
        num_slices = 256
        loss1 = compute_sigreg_loss(x, global_step, num_slices)
        loss2 = compute_sigreg_loss(x, global_step, num_slices)
        loss3 = compute_sigreg_loss(x, global_step + 1, num_slices)
        assert loss1 == loss2
        assert loss1 != loss3

    def test_compute_sigreg_loss_isotropic_gaussian(self):
        """Test that SigREG loss is lower for isotropic Gaussian embeddings."""
        B, L, D = 2, 32, 128
        global_step = 0
        num_slices = 256

        # Create isotropic Gaussian embeddings (should have lower loss)
        isotropic_x = torch.randn(B, L, D)

        # Create non-isotropic embeddings (concentrated in one direction)
        non_isotropic_x = torch.zeros(B, L, D)
        non_isotropic_x[:, :, 0] = torch.randn(B, L) * 10  # High variance in first dimension
        non_isotropic_x[:, :, 1:] = torch.randn(B, L, D - 1) * 0.1  # Low variance in other dimensions

        isotropic_loss = compute_sigreg_loss(isotropic_x, global_step, num_slices)
        non_isotropic_loss = compute_sigreg_loss(non_isotropic_x, global_step, num_slices)

        # Isotropic Gaussian should have lower SigREG loss
        assert isotropic_loss < non_isotropic_loss
        assert not torch.isnan(isotropic_loss)
        assert not torch.isnan(non_isotropic_loss)
