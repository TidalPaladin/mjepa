import random

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mjepa.trainer import (
    ResolutionConfig,
    TrainerConfig,
    assert_all_ranks_synced,
    assert_all_trainable_params_have_grad,
    calculate_total_steps,
    format_large_number,
    format_pbar_description,
    rank_zero_only,
    scale_change,
    seed_everything,
    setup_logdir,
    should_step_optimizer,
    size_change,
)


class DummyMetric:
    def __init__(self, value: float):
        self._value = value

    def compute(self):
        return self._value


def test_seed_everything_reproducible():
    seed_everything(123)
    torch_a = torch.rand(3)
    np_a = np.random.rand(3)
    py_a = random.random()

    seed_everything(123)
    torch_b = torch.rand(3)
    np_b = np.random.rand(3)
    py_b = random.random()

    assert torch.allclose(torch_a, torch_b)
    assert np.allclose(np_a, np_b)
    assert py_a == py_b


def test_should_step_optimizer():
    assert should_step_optimizer(0, 1)
    assert not should_step_optimizer(0, 2)
    assert should_step_optimizer(1, 2)


def test_calculate_total_steps():
    dataset = TensorDataset(torch.arange(10))
    dataloader = DataLoader(dataset, batch_size=2)

    total_steps = calculate_total_steps(dataloader, num_epochs=3, accumulate_grad_batches=2)

    assert total_steps == 7


def test_format_large_number():
    assert format_large_number(999) == "999"
    assert format_large_number(1_000) == "1.0000K"
    assert format_large_number(1_000_000) == "1.0000M"
    assert format_large_number(1_000_000_000) == "1.0000B"


def test_rank_zero_only_runs_when_not_distributed():
    called = {"value": False}

    @rank_zero_only
    def _mark_called():
        called["value"] = True

    _mark_called()
    assert called["value"] is True


def test_setup_logdir_creates_subdir_and_copies_config(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("foo: bar\n")

    created = setup_logdir(log_dir, config_path, name="run")

    assert created is not None
    assert created.parent == log_dir
    assert (created / "config.yaml").is_file()
    assert (created / "run.log").exists()


def test_setup_logdir_none_returns_none():
    assert setup_logdir(None, None) is None


def test_format_pbar_description_includes_metrics():
    metric = DummyMetric(2.0)
    desc = format_pbar_description(step=10, microbatch=1, epoch=2, loss=metric)

    assert "Epoch: 2" in desc
    assert "Step: 10" in desc
    assert "Microbatch: 1" in desc
    assert "loss=2.0000" in desc


def test_assert_all_trainable_params_have_grad_behavior():
    model = nn.Linear(2, 2)

    with pytest.raises(AssertionError):
        assert_all_trainable_params_have_grad(model, step=0)

    for param in model.parameters():
        param.grad = torch.zeros_like(param)

    assert_all_trainable_params_have_grad(model, step=0)
    assert_all_trainable_params_have_grad(model, step=1)


def test_assert_all_ranks_synced_noop_when_not_distributed():
    model = nn.Linear(2, 2)
    assert_all_ranks_synced(model)


def test_size_change_and_scale_change():
    size_config = ResolutionConfig(size=[64, 64], batch_size=4)

    called = {}

    def train_loader_fn(size, batch_size):
        called["train"] = (tuple(size), batch_size)
        dataset = TensorDataset(torch.arange(8))
        return DataLoader(dataset, batch_size=batch_size)

    def val_loader_fn(size, batch_size):
        called["val"] = (tuple(size), batch_size)
        dataset = TensorDataset(torch.arange(4))
        return DataLoader(dataset, batch_size=batch_size)

    train_loader, val_loader, new_accumulate = size_change(
        size_config=size_config,
        batch_size=8,
        accumulate_grad_batches=2,
        train_dataloader_fn=train_loader_fn,
        val_dataloader_fn=val_loader_fn,
    )

    assert called["train"] == ((64, 64), 4)
    assert called["val"] == ((64, 64), 4)
    assert train_loader.batch_size == 4
    assert val_loader.batch_size == 4
    assert new_accumulate == 4

    assert scale_change(base_size=[256, 256], size_config=size_config, scale=4) == 1


def test_trainer_config_size_helpers():
    sizes = {
        5: ResolutionConfig(size=[128, 128], batch_size=4),
        10: ResolutionConfig(size=[256, 256], batch_size=2),
    }
    config = TrainerConfig(
        batch_size=8,
        num_workers=0,
        num_epochs=20,
        sizes=sizes,
    )

    assert config.is_size_change_epoch(5)
    assert not config.is_size_change_epoch(6)
    assert config.get_size_for_epoch(4) is None
    assert config.get_size_for_epoch(5) == sizes[5]
    assert config.get_size_for_epoch(7) == sizes[5]
    assert config.get_size_for_epoch(12) == sizes[10]
