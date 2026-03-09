import tempfile
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, OneCycleLR

import mjepa.optimizer as optimizer_module
from mjepa.optimizer import (
    ADAMW_COMPONENT_NAME,
    HYBRID_MUON_OPTIMIZER_KIND,
    MUON_COMPONENT_NAME,
    CompositeOptimizer,
    CompositeScheduler,
    OptimizerConfig,
    _assign_parameter_groups,
    _match_parameters,
    config_constructor,
    register_constructors,
)


TEST_INPUT_DIM = 10


def _build_simple_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(TEST_INPUT_DIM, 20),
        nn.ReLU(),
        nn.Linear(20, TEST_INPUT_DIM),
    )


def _backward_sum(module: nn.Module, *, batch_size: int, loss_scale: float = 1.0) -> None:
    loss = module(torch.randn(batch_size, TEST_INPUT_DIM))
    (loss * loss_scale).sum().backward()


def _grads(module: nn.Module) -> list[torch.Tensor]:
    return [parameter.grad.detach() for parameter in module.parameters() if parameter.grad is not None]


def _trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _total_grad_norm(parameters: list[nn.Parameter]) -> float:
    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    return torch.linalg.vector_norm(torch.stack([gradient.norm() for gradient in gradients])).item()


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
        assert config.kind == "adamw"
        assert config.fused is True
        assert config.foreach is None
        assert config.eps == 1e-8
        assert config.adamw_lr is None
        assert config.muon_momentum == 0.95
        assert config.muon_nesterov is True
        assert config.muon_ns_coefficients == (3.4445, -4.7750, 2.0315)
        assert config.muon_eps == 1e-7
        assert config.muon_ns_steps == 5
        assert config.muon_adjust_lr_fn is None
        assert config.log_hybrid_adamw_parameters is False
        assert config.max_grad_norm is None
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
            kind="hybrid_muon",
            fused=False,
            foreach=True,
            eps=1e-6,
            adamw_lr=1e-4,
            muon_momentum=0.9,
            muon_nesterov=False,
            muon_ns_coefficients=(3.0, -4.0, 2.0),
            muon_eps=1e-6,
            muon_ns_steps=3,
            muon_adjust_lr_fn="match_rms_adamw",
            log_hybrid_adamw_parameters=True,
            max_grad_norm=1.5,
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
        assert config.kind == "hybrid_muon"
        assert config.fused is False
        assert config.foreach is True
        assert config.eps == 1e-6
        assert config.adamw_lr == 1e-4
        assert config.muon_momentum == 0.9
        assert config.muon_nesterov is False
        assert config.muon_ns_coefficients == (3.0, -4.0, 2.0)
        assert config.muon_eps == 1e-6
        assert config.muon_ns_steps == 3
        assert config.muon_adjust_lr_fn == "match_rms_adamw"
        assert config.log_hybrid_adamw_parameters is True
        assert config.max_grad_norm == 1.5
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

    @pytest.mark.parametrize("max_grad_norm", [0.0, -1.0])
    def test_invalid_max_grad_norm(self, max_grad_norm):
        """Test non-positive max_grad_norm values fail fast."""
        with pytest.raises(ValueError, match="max_grad_norm"):
            OptimizerConfig(
                lr=1e-3,
                weight_decay=0.01,
                betas=(0.9, 0.999),
                max_grad_norm=max_grad_norm,
            )


class TestOptimizerConfigInstantiate:
    """Test OptimizerConfig.instantiate() method."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return _build_simple_model()

    @pytest.fixture
    def model_with_head(self):
        class ModelWithHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(10, 20)
                self.head = nn.Linear(20, 5)

            def forward(self, x):
                return self.head(self.backbone(x))

        return ModelWithHead()

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

    def test_clip_grad_norm_returns_none_when_disabled(self, simple_model):
        """Test clip_grad_norm_ is a no-op when clipping is disabled."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            scheduled=False,
            fused=False,
        )
        optimizer, _ = config.instantiate(simple_model, total_steps=100)

        _backward_sum(simple_model, batch_size=8)
        gradients_before = [gradient.clone() for gradient in _grads(simple_model)]

        clipped_norm = config.clip_grad_norm_(optimizer)

        gradients_after = _grads(simple_model)
        assert clipped_norm is None
        assert len(gradients_before) == len(gradients_after)
        assert all(torch.allclose(before, after) for before, after in zip(gradients_before, gradients_after))

    def test_clip_grad_norm_clips_adamw_optimizer(self, simple_model):
        """Test clip_grad_norm_ enforces max_grad_norm on AdamW optimizers."""
        max_grad_norm = 0.1
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            scheduled=False,
            fused=False,
            max_grad_norm=max_grad_norm,
        )
        optimizer, _ = config.instantiate(simple_model, total_steps=100)

        _backward_sum(simple_model, batch_size=16, loss_scale=1000)
        parameters = _trainable_parameters(simple_model)
        grad_norm_before = _total_grad_norm(parameters)

        clipped_norm = config.clip_grad_norm_(optimizer)

        grad_norm_after = _total_grad_norm(parameters)
        assert clipped_norm is not None
        assert clipped_norm.item() == pytest.approx(grad_norm_before)
        assert grad_norm_after <= max_grad_norm + 1e-6

    def test_clip_grad_norm_clips_composite_optimizer_globally(self, simple_model):
        """Test composite optimizers are clipped once across all parameter groups."""
        max_grad_norm = 0.1
        optimizer = CompositeOptimizer(
            {
                "backbone": AdamW(simple_model[0].parameters(), lr=1e-3, fused=False),
                "head": AdamW(simple_model[2].parameters(), lr=1e-3, fused=False),
            }
        )
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            max_grad_norm=max_grad_norm,
        )

        _backward_sum(simple_model, batch_size=16, loss_scale=1000)
        parameters = _trainable_parameters(simple_model)
        grad_norm_before = _total_grad_norm(parameters)

        clipped_norm = config.clip_grad_norm_(optimizer)

        grad_norm_after = _total_grad_norm(parameters)
        assert clipped_norm is not None
        assert clipped_norm.item() == pytest.approx(grad_norm_before)
        assert grad_norm_after <= max_grad_norm + 1e-6

    @pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
    def test_instantiate_hybrid_muon(self, simple_model):
        """Test hybrid_muon returns composite optimizer and scheduler."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            kind=HYBRID_MUON_OPTIMIZER_KIND,
            scheduled=False,
            fused=False,
            muon_adjust_lr_fn="match_rms_adamw",
        )
        optimizer, scheduler = config.instantiate(simple_model, total_steps=1000)

        assert isinstance(optimizer, CompositeOptimizer)
        assert isinstance(scheduler, CompositeScheduler)
        assert MUON_COMPONENT_NAME in optimizer.optimizers
        assert ADAMW_COMPONENT_NAME in optimizer.optimizers

        muon_optimizer = optimizer.optimizers[MUON_COMPONENT_NAME]
        adamw_optimizer = optimizer.optimizers[ADAMW_COMPONENT_NAME]
        assert all(param.ndim == 2 for group in muon_optimizer.param_groups for param in group["params"])
        assert all(param.ndim != 2 for group in adamw_optimizer.param_groups for param in group["params"])

    @pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
    def test_instantiate_hybrid_muon_with_explicit_adamw_lr(self, simple_model):
        """Test hybrid mode supports a dedicated AdamW LR."""
        muon_lr = 1e-3
        adamw_lr = 2e-4
        config = OptimizerConfig(
            lr=muon_lr,
            adamw_lr=adamw_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            kind=HYBRID_MUON_OPTIMIZER_KIND,
            scheduled=False,
            fused=False,
        )
        optimizer, _ = config.instantiate(simple_model, total_steps=1000)

        assert isinstance(optimizer, CompositeOptimizer)
        assert optimizer.optimizers[MUON_COMPONENT_NAME].defaults["lr"] == muon_lr
        assert optimizer.optimizers[ADAMW_COMPONENT_NAME].defaults["lr"] == adamw_lr

    @pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
    def test_hybrid_muon_preserves_skip_weight_decay_on_1d(self, simple_model):
        """Test 1D params still receive zero weight decay in the AdamW branch."""
        base_weight_decay = 0.01
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=base_weight_decay,
            betas=(0.9, 0.999),
            kind=HYBRID_MUON_OPTIMIZER_KIND,
            scheduled=False,
            fused=False,
            skip_weight_decay_on_1d=True,
        )
        optimizer, _ = config.instantiate(simple_model, total_steps=1000)

        assert isinstance(optimizer, CompositeOptimizer)
        optimizer = cast(CompositeOptimizer, optimizer)
        adamw_optimizer = optimizer.optimizers[ADAMW_COMPONENT_NAME]
        weight_decay_by_param_id = {
            id(param): group["weight_decay"] for group in adamw_optimizer.param_groups for param in group["params"]
        }

        for param in simple_model.parameters():
            if param.ndim == 1:
                assert weight_decay_by_param_id[id(param)] == 0.0

    @pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
    def test_hybrid_muon_supports_explicit_adamw_parameter_group_assignment(self, model_with_head):
        """Test parameter groups can force AdamW assignment for selected 2D params."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            kind=HYBRID_MUON_OPTIMIZER_KIND,
            fused=False,
            parameter_groups=[{"params": ("head",), "optimizer": ADAMW_COMPONENT_NAME}],
        )
        optimizer, _ = config.instantiate(model_with_head, total_steps=1000)

        assert isinstance(optimizer, CompositeOptimizer)
        muon_param_ids = {
            id(param) for group in optimizer.optimizers[MUON_COMPONENT_NAME].param_groups for param in group["params"]
        }
        adamw_param_ids = {
            id(param) for group in optimizer.optimizers[ADAMW_COMPONENT_NAME].param_groups for param in group["params"]
        }
        assert id(model_with_head.head.weight) in adamw_param_ids
        assert id(model_with_head.head.weight) not in muon_param_ids

    @pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
    def test_hybrid_muon_rejects_non_2d_group_forced_to_muon(self, simple_model):
        """Test forcing 1D params to Muon raises a clear error."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            kind=HYBRID_MUON_OPTIMIZER_KIND,
            fused=False,
            parameter_groups=[{"params": ("bias",), "optimizer": MUON_COMPONENT_NAME}],
        )

        with pytest.raises(ValueError, match="optimizer='muon'"):
            config.instantiate(simple_model, total_steps=1000)

    @pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
    def test_hybrid_muon_rejects_unknown_parameter_group_optimizer_mode(self, simple_model):
        """Test invalid parameter-group optimizer routing mode fails fast."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            kind=HYBRID_MUON_OPTIMIZER_KIND,
            fused=False,
            parameter_groups=[{"params": ("weight",), "optimizer": "unknown"}],
        )

        with pytest.raises(ValueError, match="Unsupported parameter group optimizer mode"):
            config.instantiate(simple_model, total_steps=1000)

    @pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
    def test_hybrid_muon_logs_adamw_parameter_names_on_rank_zero(self, model_with_head, monkeypatch, caplog):
        """Test rank-zero debug logging for AdamW parameter assignments in hybrid mode."""
        monkeypatch.setattr(optimizer_module, "_is_rank_zero", lambda: True)
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            kind=HYBRID_MUON_OPTIMIZER_KIND,
            fused=False,
            log_hybrid_adamw_parameters=True,
            parameter_groups=[{"params": ("head",), "optimizer": ADAMW_COMPONENT_NAME}],
        )

        with caplog.at_level("INFO"):
            config.instantiate(model_with_head, total_steps=1000)
        assert any("Hybrid mode AdamW parameters" in record.message for record in caplog.records)
        assert any("head.weight" in record.message for record in caplog.records)

    @pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
    def test_hybrid_muon_does_not_log_on_non_zero_rank(self, model_with_head, monkeypatch, caplog):
        """Test hybrid AdamW parameter logging is suppressed on non-zero ranks."""
        monkeypatch.setattr(optimizer_module, "_is_rank_zero", lambda: False)
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            kind=HYBRID_MUON_OPTIMIZER_KIND,
            fused=False,
            log_hybrid_adamw_parameters=True,
            parameter_groups=[{"params": ("head",), "optimizer": ADAMW_COMPONENT_NAME}],
        )

        with caplog.at_level("INFO"):
            config.instantiate(model_with_head, total_steps=1000)
        assert not any("Hybrid mode AdamW parameters" in record.message for record in caplog.records)

    @pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
    def test_instantiate_hybrid_muon_with_onecycle_scheduler(self, simple_model):
        """Test hybrid_muon with scheduled=True applies OneCycleLR to each optimizer component."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            kind=HYBRID_MUON_OPTIMIZER_KIND,
            scheduled=True,
            fused=False,
        )
        optimizer, scheduler = config.instantiate(simple_model, total_steps=100)

        assert isinstance(optimizer, CompositeOptimizer)
        assert isinstance(scheduler, CompositeScheduler)
        assert isinstance(scheduler.schedulers[MUON_COMPONENT_NAME], OneCycleLR)
        assert isinstance(scheduler.schedulers[ADAMW_COMPONENT_NAME], OneCycleLR)

    @pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
    def test_hybrid_muon_optimizer_and_scheduler_state_roundtrip(self, simple_model):
        """Test state dict round-trip for hybrid optimizer and scheduler."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            kind=HYBRID_MUON_OPTIMIZER_KIND,
            scheduled=False,
            fused=False,
        )
        optimizer, scheduler = config.instantiate(simple_model, total_steps=100)

        inputs = torch.randn(8, 10)
        loss = simple_model(inputs).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()

        optimizer_copy, scheduler_copy = config.instantiate(simple_model, total_steps=100)
        optimizer_copy.load_state_dict(optimizer_state)
        scheduler_copy.load_state_dict(scheduler_state)
        assert optimizer_copy.state_dict()["_mjepa_optimizer_kind"] == HYBRID_MUON_OPTIMIZER_KIND
        assert scheduler_copy.state_dict()["_mjepa_scheduler_kind"] == HYBRID_MUON_OPTIMIZER_KIND

    @pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon unavailable")
    def test_hybrid_muon_rejects_legacy_state_dicts(self, simple_model):
        """Test hybrid optimizer/scheduler reject legacy non-hybrid state dicts."""
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            kind=HYBRID_MUON_OPTIMIZER_KIND,
            scheduled=False,
            fused=False,
        )
        optimizer, scheduler = config.instantiate(simple_model, total_steps=100)

        legacy_optimizer = AdamW(simple_model.parameters(), lr=1e-3, fused=False)
        legacy_scheduler = LinearLR(legacy_optimizer, start_factor=0.1, end_factor=1.0, total_iters=2)

        with pytest.raises(ValueError, match="Cannot load checkpoint optimizer kind"):
            optimizer.load_state_dict(legacy_optimizer.state_dict())
        with pytest.raises(ValueError, match="Cannot load checkpoint scheduler kind"):
            scheduler.load_state_dict(legacy_scheduler.state_dict())

    def test_hybrid_muon_requires_torch_muon(self, simple_model, monkeypatch):
        """Test hybrid_muon raises a clear error when torch.optim.Muon is unavailable."""
        monkeypatch.delattr(torch.optim, "Muon", raising=False)
        config = OptimizerConfig(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            kind=HYBRID_MUON_OPTIMIZER_KIND,
            scheduled=False,
            fused=False,
        )

        with pytest.raises(RuntimeError, match="torch.optim.Muon is unavailable"):
            config.instantiate(simple_model, total_steps=100)


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
max_grad_norm: 1.25
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
        assert config.max_grad_norm == 1.25
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
max_grad_norm: 0.75
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
        assert config.max_grad_norm == 0.75
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
max_grad_norm: 2.0
"""
        config = yaml.safe_load(yaml_str)

        assert isinstance(config, OptimizerConfig)
        assert config.lr == 0.002
        assert config.weight_decay == 0.05
        # YAML loads lists, not tuples
        assert list(config.betas) == [0.85, 0.95]
        assert config.max_grad_norm == 2.0
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
