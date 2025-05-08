import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from shutil import copyfile
from typing import Callable, Tuple, overload
from warnings import filterwarnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchmetrics as tm
import yaml
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    r"""Seed everything.

    Args:
        seed: The seed to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def should_step_optimizer(microbatch: int, accumulate_grad_batches: int) -> bool:
    r"""Determine if the optimizer should be stepped.

    Primarily used to account for gradient accumulation where an optimizer step is performed every ``accumulate_grad_batches`` steps.

    Args:
        microbatch: Current microbatch index.
        accumulate_grad_batches: Number of gradient accumulation steps.

    Returns:
        ``True`` if the optimizer should be stepped, ``False`` otherwise.
    """
    return (microbatch + 1) % accumulate_grad_batches == 0


def calculate_total_steps(dataloader: DataLoader, num_epochs: int, accumulate_grad_batches: int) -> int:
    """Calculate the total number of steps for the training loop.

    The computation is `num_epochs * len(dataloader) // accumulate_grad_batches`.

    Returns:
        The total number of steps.
    """
    return num_epochs * len(dataloader) // accumulate_grad_batches


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Count the total number of parameters in a module.

    Args:
        trainable_only: If True, only count trainable parameters.

    Returns:
        The total number of parameters.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def format_large_number(count: int) -> str:
    """Format a large number into a human-readable string.

    Returns:
        A string of the form "1.200M" or "1.200B" etc.
    """
    if count < 1_000:
        return count
    elif count < 1_000_000:
        return f"{count / 1_000:.4f}K"
    elif count < 1_000_000_000:
        return f"{count / 1_000_000:.4f}M"
    else:
        return f"{count / 1_000_000_000:.4f}B"


def is_rank_zero() -> bool:
    r"""Determine if the current rank is zero."""
    return not dist.is_initialized() or dist.get_rank() == 0


def rank_zero_only(fn: Callable) -> Callable:
    """Decorate a function to only run on rank zero."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if is_rank_zero():
            return fn(*args, **kwargs)

    return wrapper


@rank_zero_only
def rank_zero_print(message: str, tqdm_safe: bool = True) -> None:
    """Print a message if the current rank is zero.

    Args:
        message: The message to print.
    """
    if tqdm_safe:
        with tqdm.external_write_mode():
            print(message)
    else:
        print(message)


@rank_zero_only
def rank_zero_info(message: str, tqdm_safe: bool = True) -> None:
    """Log an info message if the current rank is zero.

    Args:
        message: The message to log.
    """
    if tqdm_safe:
        with tqdm.external_write_mode():
            logging.info(message)
    else:
        logging.info(message)


@rank_zero_only
def save_checkpoint(
    path: os.PathLike,
    backbone: nn.Module | None,
    predictor: nn.Module | None,
    probe: nn.Module | None,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    step: int,
    epoch: int,
) -> None:
    r"""Save a checkpoint.

    Runs only on rank zero.

    Args:
        path: Path to save the checkpoint.
        backbone: Backbone model or ``None`` if not present.
        predictor: Predictor model or ``None`` if not present.
        probe: Probe model or ``None`` if not present.
        optimizer: Optimizer.
        scheduler: Scheduler.
        step: Current step.
        epoch: Current epoch.
    """
    path = Path(path)
    if not path.parent.is_dir():
        raise NotADirectoryError(f"Checkpoint parent directory does not exist: {path.parent}")
    data = dict(
        backbone=backbone.state_dict() if backbone else None,
        predictor=predictor.state_dict() if predictor else None,
        probe=probe.state_dict() if probe else None,
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        step=step,
        epoch=epoch,
    )
    torch.save(data, path)


def load_checkpoint(
    path: os.PathLike,
    backbone: nn.Module | None,
    predictor: nn.Module | None,
    probe: nn.Module | None,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    strict: bool = True,
) -> Tuple[int, int]:
    r"""Load a checkpoint.

    Weights are loaded in-place.

    Args:
        path: Path to load the checkpoint.
        backbone: Backbone model or ``None`` if not present.
        predictor: Predictor model or ``None`` if not present.
        probe: Probe model or ``None`` if not present.
        optimizer: Optimizer.
        scheduler: Scheduler.
        strict: If ``True``, raise an error if any keys are missing in the checkpoint.

    Returns:
        Tuple of current step and epoch.
    """
    data = torch.load(path)
    if backbone:
        backbone.load_state_dict(data["backbone"], strict=strict)
    if predictor:
        predictor.load_state_dict(data["predictor"], strict=strict)
    if probe:
        probe.load_state_dict(data["probe"], strict=strict)
    optimizer.load_state_dict(data["optimizer"])
    scheduler.load_state_dict(data["scheduler"])
    return data["step"], data["epoch"]


@overload
def setup_logdir(log_dir: None, config_path: os.PathLike | None) -> None: ...


@overload
def setup_logdir(log_dir: os.PathLike, config_path: os.PathLike) -> Path: ...


def setup_logdir(log_dir: os.PathLike | None, config_path: os.PathLike | None, name: str | None = None) -> Path | None:
    r"""Setup a log directory.

    Does the following:
        - Creates a logging subdirectory with the current timestamp in the log directory.
        - Sets the logger to output to a file in the log subdirectory.
        - Copies the config file to the log subdirectory (if provided).

    If ``log_dir`` is ``None``, no setup is performed and ``None`` is returned.

    Args:
        log_dir: Path to the log directory.
        config_path: Path to the config file.
        name: Name of the run. Will be appended to the log subdirectory.

    Returns:
        The created logging subdirectory.
    """
    # Set up basic logging format
    if dist.is_initialized():
        rank = dist.get_rank() if dist.is_initialized() else 0
        fmt = f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s"
    else:
        fmt = "%(asctime)s - %(levelname)s - %(message)s"

    # Reset any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create formatters and handlers
    formatter = logging.Formatter(fmt)

    # Always add console handler for basic output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(console_handler)

    if log_dir is None:
        return None

    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        raise NotADirectoryError(f"Log directory does not exist: {log_dir}")

    # Create logging subdirectory
    subdir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if name:
        subdir = f"{subdir}_{name}"
    log_dir = log_dir / subdir
    log_dir.mkdir()

    # Add file handler when log directory is provided
    file_handler = logging.FileHandler(log_dir / "run.log")
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)

    # Copy config file to log directory
    if config_path:
        copyfile(config_path, log_dir / "config.yaml")
    return log_dir


def format_pbar_description(
    step: int,
    microbatch: int,
    epoch: int,
    **metrics: tm.Metric,
) -> str:
    r"""Format a progress bar description.

    Args:
        step: Current step.
        microbatch: Current microbatch.
        epoch: Current epoch.
        **metrics: Metrics to include in the description.
    """
    return (
        f"Epoch: {epoch}, "
        f"Step: {format_large_number(step)}, "
        f"Microbatch: {format_large_number(microbatch)}, "
        f"[{' '.join([f'{k}={v.compute():.4f}' for k, v in metrics.items()])}]"
    )


def assert_all_trainable_params_have_grad(model: nn.Module, step: int | None = None) -> None:
    r"""Assert that all trainable parameters have a gradient.

    If the current step is provided, the assertion is only performed if the step is zero.

    Args:
        model: The model to check.
        step: The current step.
    """
    if step != 0:
        return

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            raise AssertionError(f"Parameter {name} should have a gradient, but does not")


def ignore_warnings():
    filterwarnings("ignore", category=DeprecationWarning)
    filterwarnings("ignore", category=UserWarning)
    filterwarnings("ignore", category=FutureWarning)


@dataclass
class TrainerConfig:
    batch_size: int
    num_workers: int
    num_epochs: int
    accumulate_grad_batches: int = 1
    log_interval: int = 50
    check_val_every_n_epoch: int = 1


def config_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    return TrainerConfig(**values)


def register_constructors():
    tags = [
        "tag:yaml.org,2002:python/object:mjepa.TrainerConfig",
    ]
    loaders = [yaml.SafeLoader, yaml.FullLoader, yaml.UnsafeLoader]
    for tag in tags:
        for loader in loaders:
            loader.add_constructor(tag, config_constructor)


register_constructors()
