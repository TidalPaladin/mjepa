import tempfile
from pathlib import Path

import pytest
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, OneCycleLR

from mjepa.optimizer import (
    OptimizerConfig,
    _assign_parameter_groups,
    _match_parameters,
    config_constructor,
    register_constructors,
)


class TestOptimizerConfigInit:
    """Test OptimizerConfig initialization and field defaults."""

    def test_default_values(self):
        """Test that OptimizerConfig has correct default values."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        assert config.lr == 1e-3
        assert config.weight_decay == 0.01
        assert config.betas == (0.9, 0.999)
        assert config.fused is True
        assert config.foreach is None
        assert config.eps == 1e-8
        assert config.scheduled is False
        assert config.pct_start == 0.05
        assert config.base_momentum == 0.85
        assert config.max_momentum == 0.95
        assert config.div_factor == 10.0
        assert config.final_div_factor == 1e4
        assert config.parameter_groups == []
        assert config.skip_weight_decay_on_1d is False

    def test_custom_values(self):
        """Test OptimizerConfig with custom values."""
        config = OptimizerConfig(
            lr=5e-4,
            weight_decay=0.05,
            betas=(0.85, 0.95),
            fused=False,
            foreach=True,
            eps=1e-6,
            scheduled=True,
            pct_start=0.1,
            base_momentum=0.8,
            max_momentum=0.9,
            div_factor=5.0,
            final_div_factor=1e3,
            parameter_groups=[{"params": ("bias",), "weight_decay": 0.0}],
            skip_weight_decay_on_1d=True,
        )
        assert config.lr == 5e-4
        assert config.weight_decay == 0.05
        assert config.betas == (0.85, 0.95)
        assert config.fused is False
        assert config.foreach is True
        assert config.eps == 1e-6
        assert config.scheduled is True
        assert config.pct_start == 0.1
        assert config.base_momentum == 0.8
        assert config.max_momentum == 0.9
        assert config.div_factor == 5.0
        assert config.final_div_factor == 1e3
        assert len(config.parameter_groups) == 1
        assert config.skip_weight_decay_on_1d is True

    def test_frozen_dataclass(self):
        """Test that OptimizerConfig is frozen (immutable)."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        with pytest.raises(AttributeError):
            config.lr = 1e-4  # type: ignore


class TestOptimizerConfigInstantiate:
    """Test OptimizerConfig.instantiate() method."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

    def test_instantiate_with_linear_scheduler(self, simple_model):
        """Test instantiate with scheduled=False returns LinearLR."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            scheduled=False,
            fused=False,  # Disable fused for CPU tests
        )
        optimizer, scheduler = config.instantiate(simple_model, total_steps=1000)

        assert isinstance(optimizer, AdamW)
        assert isinstance(scheduler, LinearLR)
        assert optimizer.defaults["lr"] == 1e-3
        assert optimizer.defaults["weight_decay"] == 0.01
        assert optimizer.defaults["betas"] == (0.9, 0.999)

    def test_instantiate_with_onecycle_scheduler(self, simple_model):
        """Test instantiate with scheduled=True returns OneCycleLR."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            scheduled=True,
            pct_start=0.1,
            fused=False,
        )
        optimizer, scheduler = config.instantiate(simple_model, total_steps=1000)

        assert isinstance(optimizer, AdamW)
        assert isinstance(scheduler, OneCycleLR)

    def test_instantiate_with_parameter_groups(self, simple_model):
        """Test instantiate with custom parameter groups."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            scheduled=False,
            fused=False,
            parameter_groups=[{"params": ("bias",), "weight_decay": 0.0, "lr": 1e-4}],
        )
        optimizer, scheduler = config.instantiate(simple_model, total_steps=1000)

        assert isinstance(optimizer, AdamW)
        # Should have at least 2 parameter groups: bias group and default group
        assert len(optimizer.param_groups) >= 2

    def test_instantiate_skip_weight_decay_on_1d(self, simple_model):
        """Test instantiate can disable weight decay on 1D parameters."""
        base_weight_decay = 0.01
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=base_weight_decay,
            betas=(0.9, 0.999),
            scheduled=False,
            fused=False,
            skip_weight_decay_on_1d=True,
        )
        optimizer, _ = config.instantiate(simple_model, total_steps=1000)

        weight_decay_by_param_id = {
            id(param): group["weight_decay"] for group in optimizer.param_groups for param in group["params"]
        }

        for param in simple_model.parameters():
            expected_weight_decay = 0.0 if param.ndim == 1 else base_weight_decay
            assert weight_decay_by_param_id[id(param)] == expected_weight_decay

    def test_instantiate_linear_scheduler_warmup_steps(self, simple_model):
        """Test LinearLR has correct warmup steps based on pct_start."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            scheduled=False,
            pct_start=0.1,
            fused=False,
        )
        total_steps = 1000
        optimizer, scheduler = config.instantiate(simple_model, total_steps=total_steps)

        assert isinstance(scheduler, LinearLR)
        # LinearLR total_iters should be pct_start * total_steps
        assert scheduler.total_iters == int(total_steps * 0.1)

    def test_instantiate_onecycle_scheduler_params(self, simple_model):
        """Test OneCycleLR has correct parameters."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            scheduled=True,
            pct_start=0.05,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=10.0,
            final_div_factor=1e4,
            fused=False,
        )
        optimizer, scheduler = config.instantiate(simple_model, total_steps=1000)

        assert isinstance(scheduler, OneCycleLR)
        assert scheduler.total_steps == 1000


class TestOptimizerConfigFromYAML:
    """Test OptimizerConfig.from_yaml() method."""

    def test_from_yaml_basic(self):
        """Test loading OptimizerConfig from YAML file."""
        yaml_content = """
lr: 0.001
weight_decay: 0.01
betas:
  - 0.9
  - 0.999
scheduled: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = OptimizerConfig.from_yaml(Path(f.name))

        assert config.lr == 0.001
        assert config.weight_decay == 0.01
        # YAML loads lists, not tuples
        assert list(config.betas) == [0.9, 0.999]
        assert config.scheduled is False
        assert config.skip_weight_decay_on_1d is False

    def test_from_yaml_with_parameter_groups(self):
        """Test loading OptimizerConfig with parameter groups from YAML."""
        yaml_content = """
lr: 0.001
weight_decay: 0.01
betas:
  - 0.9
  - 0.999
scheduled: true
parameter_groups:
  - params:
      - bias
    weight_decay: 0.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = OptimizerConfig.from_yaml(Path(f.name))

        assert config.lr == 0.001
        assert config.scheduled is True
        assert len(config.parameter_groups) == 1
        assert config.parameter_groups[0]["weight_decay"] == 0.0

    def test_from_yaml_with_skip_weight_decay_on_1d(self):
        """Test loading OptimizerConfig with 1D weight decay skip."""
        yaml_content = """
lr: 0.001
weight_decay: 0.01
betas:
  - 0.9
  - 0.999
skip_weight_decay_on_1d: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = OptimizerConfig.from_yaml(Path(f.name))

        assert config.skip_weight_decay_on_1d is True


class TestYAMLConfig:
    """Test YAML constructor registration and loading."""

    def test_config_constructor(self, mocker):
        """Test the config_constructor function."""
        loader = mocker.Mock()
        node = mocker.Mock()

        config_dict = {
            "lr": 0.001,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        }
        loader.construct_mapping.return_value = config_dict

        config = config_constructor(loader, node)

        assert isinstance(config, OptimizerConfig)
        assert config.lr == 0.001
        assert config.weight_decay == 0.01
        loader.construct_mapping.assert_called_once_with(node, deep=True)

    def test_yaml_safe_load(self):
        """Test loading OptimizerConfig via yaml.safe_load with custom tag."""
        register_constructors()

        yaml_str = """
!!python/object:mjepa.OptimizerConfig
lr: 0.002
weight_decay: 0.05
betas:
  - 0.85
  - 0.95
scheduled: true
"""
        config = yaml.safe_load(yaml_str)

        assert isinstance(config, OptimizerConfig)
        assert config.lr == 0.002
        assert config.weight_decay == 0.05
        # YAML loads lists, not tuples
        assert list(config.betas) == [0.85, 0.95]
        assert config.scheduled is True


class TestMatchParameters:
    """Test _match_parameters function."""

    @pytest.fixture
    def nested_model(self):
        """Create a nested model for testing parameter matching."""

        class SubModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)
                self.norm = nn.LayerNorm(10)

            def forward(self, x):
                return self.norm(self.fc(x))

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = SubModule()
                self.layer2 = SubModule()
                self.output = nn.Linear(10, 5)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                return self.output(x)

        return TestModel()

    def test_match_by_param_name(self, nested_model):
        """Test matching parameters by parameter name part."""
        params = list(_match_parameters(nested_model, ("bias",)))
        # Should match all bias parameters
        bias_count = sum(1 for n, _ in nested_model.named_parameters() if "bias" in n)
        assert len(params) == bias_count

    def test_match_by_module_class_name(self, nested_model):
        """Test matching parameters by module class name."""
        params = list(_match_parameters(nested_model, ("LayerNorm",)))
        # LayerNorm has weight and bias parameters
        assert len(params) == 4  # 2 LayerNorm modules × 2 params each

    def test_match_multiple_keys(self, nested_model):
        """Test matching with multiple keys."""
        params = list(_match_parameters(nested_model, ("bias", "LayerNorm")))
        # Should match bias params + LayerNorm params (with some overlap)
        assert len(params) > 0

    def test_no_matches(self, nested_model):
        """Test when no parameters match."""
        params = list(_match_parameters(nested_model, ("nonexistent",)))
        assert len(params) == 0


class TestAssignParameterGroups:
    """Test _assign_parameter_groups function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20, bias=True),
            nn.ReLU(),
            nn.Linear(20, 10, bias=True),
        )

    def test_empty_parameter_groups(self, simple_model):
        """Test with empty parameter_groups list."""
        groups = _assign_parameter_groups(simple_model, [])
        # Should have single default group with all params
        assert len(groups) == 1
        all_params = list(p for p in simple_model.parameters() if p.requires_grad)
        assert len(groups[0]["params"]) == len(all_params)

    def test_single_parameter_group(self, simple_model):
        """Test with single custom parameter group."""
        groups = _assign_parameter_groups(
            simple_model,
            [{"params": ("bias",), "weight_decay": 0.0}],
        )
        # Should have bias group + default group
        assert len(groups) == 2
        # First group should have weight_decay=0
        assert groups[0].get("weight_decay") == 0.0

    def test_multiple_parameter_groups(self, simple_model):
        """Test with multiple custom parameter groups."""
        groups = _assign_parameter_groups(
            simple_model,
            [
                {"params": ("bias",), "weight_decay": 0.0, "lr": 1e-4},
                {"params": ("weight",), "lr": 1e-3},
            ],
        )
        # Should have bias group + weight group + default group
        # Note: default group may be empty if all params are assigned
        assert len(groups) >= 2

    def test_non_overlapping_assignment(self, simple_model):
        """Test that parameters are not assigned to multiple groups."""
        groups = _assign_parameter_groups(
            simple_model,
            [
                {"params": ("bias",), "weight_decay": 0.0},
                {"params": ("weight",), "weight_decay": 0.1},
            ],
        )

        # Collect all assigned parameters
        all_assigned = []
        for group in groups:
            all_assigned.extend(group["params"])

        # Check no duplicates
        param_ids = [id(p) for p in all_assigned]
        assert len(param_ids) == len(set(param_ids)), "Parameters should not appear in multiple groups"

    def test_frozen_params_excluded(self, simple_model):
        """Test that frozen parameters are excluded from all groups."""
        # Freeze the first layer
        for param in simple_model[0].parameters():
            param.requires_grad = False

        groups = _assign_parameter_groups(simple_model, [])

        total_params = sum(len(g["params"]) for g in groups)
        trainable_params = sum(1 for p in simple_model.parameters() if p.requires_grad)
        assert total_params == trainable_params

    def test_skip_weight_decay_on_1d_groups(self, simple_model):
        """Test 1D trainable parameters get assigned to no-decay group."""
        groups = _assign_parameter_groups(simple_model, [], skip_weight_decay_on_1d=True)

        no_decay_group = next((group for group in groups if group.get("weight_decay") == 0.0), None)
        expected_1d_params = [param for param in simple_model.parameters() if param.requires_grad and param.ndim == 1]
        assert no_decay_group is not None
        assert all(param.ndim == 1 for param in no_decay_group["params"])
        assert len(no_decay_group["params"]) == len(expected_1d_params)
