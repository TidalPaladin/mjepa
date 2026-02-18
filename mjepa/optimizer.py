from collections.abc import Iterator
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

import torch.nn as nn
import yaml
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler, OneCycleLR


@dataclass(frozen=True)
class OptimizerConfig:
    # Optimizer
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    fused: bool | None = True
    foreach: bool | None = None
    eps: float = 1e-8

    # Scheduler
    scheduled: bool = False
    pct_start: float = 0.05
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 10.0
    final_div_factor: float = 1e4

    # Parameter groups
    parameter_groups: list[dict[str, Any]] = field(default_factory=list)
    skip_weight_decay_on_1d: bool = False

    def instantiate(self, model: nn.Module, total_steps: int) -> tuple[Optimizer, LRScheduler]:
        parameter_groups = _assign_parameter_groups(
            model,
            self.parameter_groups,
            skip_weight_decay_on_1d=self.skip_weight_decay_on_1d,
        )
        optimizer = AdamW(
            parameter_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            fused=self.fused,
            foreach=self.foreach,
            eps=self.eps,
        )
        if self.scheduled:
            max_lrs = [group["lr"] for group in optimizer.param_groups]
            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=self.pct_start,
                base_momentum=self.base_momentum,
                max_momentum=self.max_momentum,
                div_factor=self.div_factor,
                final_div_factor=self.final_div_factor,
            )
        else:
            scheduler = LinearLR(
                optimizer,
                start_factor=1 / self.div_factor,
                end_factor=1,
                total_iters=int(total_steps * self.pct_start),
            )
        return optimizer, scheduler

    @classmethod
    def from_yaml(cls, path: PathLike) -> "OptimizerConfig":
        with open(path) as f:
            return cls(**yaml.full_load(f))


def _match_parameters(
    module: nn.Module,
    keys: tuple[str, ...],
    prefix: str = "",
) -> Iterator[nn.Parameter]:
    # Check direct parameters of the current module
    yield from (
        param
        for param_name, param in module.named_parameters(recurse=False, prefix=prefix)
        if (
            # Module class name matches a key exactly
            module.__class__.__name__ in keys
            # Or any part of the parameter name matches a key
            or any(part in keys for part in param_name.split("."))
        )
    )

    # Recurse into submodules
    for name, submodule in module.named_children():
        yield from _match_parameters(submodule, keys, prefix=f"{prefix}.{name}")


def _assign_parameter_groups(
    model: nn.Module,
    parameter_groups: list[dict[str, Any]],
    skip_weight_decay_on_1d: bool = False,
) -> list[dict[str, Any]]:
    assigned_groups: list[dict[str, Any]] = []
    assigned_params: set[nn.Parameter] = set()
    for config in parameter_groups:
        keys = config["params"]
        params = set(p for p in _match_parameters(model, keys) if p.requires_grad)
        params = params.difference(assigned_params)
        if params:
            assert params.isdisjoint(assigned_params)
            kwargs = {k: v for k, v in config.items() if k != "params"}
            assigned_groups.append({"params": list(params), **kwargs})
        assigned_params.update(params)

    remaining_trainable_params = [p for p in model.parameters() if p.requires_grad and p not in assigned_params]
    if skip_weight_decay_on_1d:
        no_decay_1d_params = [p for p in remaining_trainable_params if p.ndim == 1]
        if no_decay_1d_params:
            assigned_groups.append({"params": no_decay_1d_params, "weight_decay": 0.0})
            assigned_params.update(no_decay_1d_params)
            remaining_trainable_params = [p for p in remaining_trainable_params if p.ndim != 1]

    # Default param group
    assigned_groups.append({"params": remaining_trainable_params})
    return assigned_groups


def config_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    return OptimizerConfig(**values)


def register_constructors():
    tags = [
        "tag:yaml.org,2002:python/object:mjepa.OptimizerConfig",
    ]
    loaders = [yaml.SafeLoader, yaml.FullLoader, yaml.UnsafeLoader]
    for tag in tags:
        for loader in loaders:
            loader.add_constructor(tag, config_constructor)


register_constructors()
