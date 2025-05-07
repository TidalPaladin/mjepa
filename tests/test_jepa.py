import pytest
import torch
import torch.nn as nn
import yaml

from mjepa.jepa import (
    JEPAConfig,
    config_constructor,
    generate_masks,
    get_momentum,
    register_constructors,
    update_teacher,
)


# class TestCrossAttentionPredictor:
#    @pytest.fixture
#    def predictor(self, vit):
#        return CrossAttentionPredictor(vit, depth=2, out_dim=None)
#
#    def test_init(self, predictor, vit):
#        """Test that the predictor is initialized correctly."""
#        assert isinstance(predictor.pos_enc, RelativeFactorizedPosition)
#        assert len(predictor.blocks) == 2
#        assert predictor.checkpoint == False
#
#    def test_forward(self, mocker, vit, predictor, dummy_image_batch, dummy_tensor_3d):
#        """Test the forward pass of the predictor."""
#        # Setup inputs
#        tokenized_size = (4, 4)
#        context = dummy_tensor_3d
#        context_ratio = 0.5
#        target_ratio = 0.25
#        scale = 1
#        x = dummy_image_batch
#
#        context_mask, target_mask = generate_masks(
#            vit, x, context_ratio, target_ratio, scale
#        )
#
#        # Call forward
#        output = predictor(tokenized_size, context, target_mask)
#
#        # Verify shape and types
#        assert isinstance(output, torch.Tensor)
#        assert output.shape == (2, 16, 128)
#


class TestGenerateMasks:
    def test_generate_masks(self, mocker, vit, dummy_image_batch):
        """Test mask generation."""
        x = dummy_image_batch
        context_ratio = 0.5
        target_ratio = 0.25
        scale = 4
        context_mask, target_mask = generate_masks(vit, x, context_ratio, target_ratio, scale)
        assert context_mask.shape == (2, 64)
        assert target_mask.shape == (2, 32)
        assert not (context_mask & target_mask).any()


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

    def test_get_momentum_middle(self):
        """Test momentum in middle of training."""
        assert get_momentum(50, 100, 0.9) == 0.95

    def test_get_momentum_end(self):
        """Test momentum at end of training."""
        assert get_momentum(100, 100, 0.9) == 1.0

    def test_get_momentum_interpolation(self):
        """Test momentum interpolation at arbitrary step."""
        assert get_momentum(25, 100, 0.8) == 0.85


class TestJEPAConfig:
    def test_default_config(self):
        """Test default configuration."""
        config = JEPAConfig()
        assert config.context_ratio == 0.5
        assert config.target_ratio == 0.25
        assert config.scale == 4
        assert config.momentum == 0.99
        assert config.predictor_depth == 4
        assert config.loss_fn == "cosine"

    def test_custom_config(self):
        """Test custom configuration."""
        config = JEPAConfig(
            context_ratio=0.6,
            target_ratio=0.3,
            scale=8,
            momentum=0.95,
            predictor_depth=6,
            loss_fn="smooth_l1",
        )
        assert config.context_ratio == 0.6
        assert config.target_ratio == 0.3
        assert config.scale == 8
        assert config.momentum == 0.95
        assert config.predictor_depth == 6
        assert config.loss_fn == "smooth_l1"

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

    def test_invalid_loss_fn(self):
        """Test invalid loss function."""
        with pytest.raises(ValueError):
            JEPAConfig(loss_fn="invalid")


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
            "loss_fn": "smooth_l1",
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
        assert config.loss_fn == "smooth_l1"

        # Verify the loader was called correctly
        loader.construct_mapping.assert_called_once_with(node, deep=True)

    def test_register_constructors(self, mocker):
        """Test that register_constructors adds constructors to loaders."""
        # Create a mock TestLoader class
        TestLoader = mocker.Mock()
        TestLoader.yaml_constructors = {}

        # Patch yaml.SafeLoader to return our test loader
        mocker.patch("yaml.SafeLoader", TestLoader)
        mocker.patch("yaml.FullLoader", TestLoader)
        mocker.patch("yaml.UnsafeLoader", TestLoader)

        # Register constructors
        register_constructors()

        # Check that the constructor was registered
        yaml_tag = "tag:yaml.org,2002:python/object:mjepa.JEPAConfig"
        assert yaml_tag in TestLoader.yaml_constructors
        assert TestLoader.yaml_constructors[yaml_tag] == config_constructor

    def test_yaml_load(self):
        """Test loading a JEPAConfig from YAML."""
        # Register constructors
        register_constructors()

        # Create a YAML string
        yaml_str = """
        !python/object:mjepa.JEPAConfig
        context_ratio: 0.8
        target_ratio: 0.3
        scale: 12
        momentum: 0.97
        predictor_depth: 5
        loss_fn: smooth_l1
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
        assert config.loss_fn == "smooth_l1"
