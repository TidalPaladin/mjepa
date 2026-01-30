import yaml

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, OneCycleLR

from mjepa.optimizer import OptimizerConfig, _assign_parameter_groups, _match_parameters


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.block = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        self.extra = nn.Parameter(torch.ones(1))
        self.frozen = nn.Parameter(torch.zeros(1), requires_grad=False)


def _build_model() -> TinyModel:
    return TinyModel()


def test_match_parameters_by_class_name():
    model = _build_model()
    params = list(_match_parameters(model, ("Linear",)))

    expected = {
        model.linear.weight,
        model.linear.bias,
        model.block[0].weight,
        model.block[0].bias,
    }
    assert set(params) == expected


def test_match_parameters_by_param_name():
    model = _build_model()
    params = list(_match_parameters(model, ("bias",)))

    expected = {model.linear.bias, model.block[0].bias}
    assert set(params) == expected


def test_assign_parameter_groups_splits_params():
    model = _build_model()
    parameter_groups = [
        {"params": ("bias",), "weight_decay": 0.0},
        {"params": ("Linear",), "lr": 1e-4},
    ]

    assigned = _assign_parameter_groups(model, parameter_groups)

    all_params = []
    for group in assigned:
        all_params.extend(group["params"])

    assert len(all_params) == 5  # 4 Linear params + 1 extra param
    assert len(set(all_params)) == 5
    assert model.extra in assigned[-1]["params"]
    assert model.frozen not in set(all_params)


def test_optimizer_config_instantiate_linear_scheduler():
    model = _build_model()
    config = OptimizerConfig(
        lr=1e-3,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        fused=False,
        scheduled=False,
    )

    optimizer, scheduler = config.instantiate(model, total_steps=100)

    assert isinstance(optimizer, AdamW)
    assert isinstance(scheduler, LinearLR)


def test_optimizer_config_instantiate_onecycle_scheduler_with_group_lr():
    model = _build_model()
    config = OptimizerConfig(
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        fused=False,
        scheduled=True,
        parameter_groups=[
            {"params": ("bias",), "lr": 5e-4, "weight_decay": 0.0},
        ],
    )

    optimizer, scheduler = config.instantiate(model, total_steps=50)

    assert isinstance(optimizer, AdamW)
    assert isinstance(scheduler, OneCycleLR)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4 / config.div_factor)


def test_optimizer_config_from_yaml(tmp_path):
    config_path = tmp_path / "optimizer.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "lr": 0.01,
                "weight_decay": 0.1,
                "betas": [0.8, 0.99],
                "fused": False,
                "foreach": None,
            }
        )
    )

    config = OptimizerConfig.from_yaml(config_path)

    assert config.lr == 0.01
    assert config.weight_decay == 0.1
    assert tuple(config.betas) == (0.8, 0.99)
    assert config.fused is False
