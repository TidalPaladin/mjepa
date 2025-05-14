from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Dict, Iterator, List, Literal, Set, Tuple

import torch.nn as nn
import yaml
from pytorch_optimizer import SOAP
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR


@dataclass(frozen=True)
class OptimizerConfig:
    # Optimizer
    lr: float
    weight_decay: float
    betas: Tuple[float, float]
    fused: bool | None = True
    foreach: bool | None = None
    eps: float = 1e-8
    precondition_frequency: int = 10
    optimizer: Literal["adamw", "soap"] = "adamw"

    # Scheduler
    pct_start: float = 0.05
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 10.0
    final_div_factor: float = 1e4

    # Parameter groups
    parameter_groups: List[Dict[str, Any]] = field(default_factory=list)

    def instantiate(self, model: nn.Module, total_steps: int) -> Tuple[Optimizer, LRScheduler]:
        match self.optimizer:
            case "adamw":
                optimizer = self._instantiate_adamw(model)
            case "soap":
                optimizer = self._instantiate_soap(model)
            case _:
                raise ValueError(f"Invalid optimizer: {self.optimizer}")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=self.pct_start,
            base_momentum=self.base_momentum,
            max_momentum=self.max_momentum,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
        )
        return optimizer, scheduler

    def _instantiate_adamw(self, model: nn.Module) -> Optimizer:
        parameter_groups = _assign_parameter_groups(model, self.parameter_groups)
        return AdamW(
            parameter_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            fused=self.fused,
            foreach=self.foreach,
            eps=self.eps,
        )

    def _instantiate_soap(self, model: nn.Module) -> Optimizer:
        parameter_groups = _assign_parameter_groups(model, self.parameter_groups)
        return SOAP(
            parameter_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            precondition_frequency=self.precondition_frequency,
            eps=self.eps,
        )

    @classmethod
    def from_yaml(cls, path: PathLike) -> "OptimizerConfig":
        with open(path, "r") as f:
            return cls(**yaml.full_load(f))


def _match_parameters(
    module: nn.Module,
    keys: Tuple[str, ...],
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
    parameter_groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    assigned_groups: List[Dict[str, Any]] = []
    assigned_params: Set[nn.Parameter] = set()
    for config in parameter_groups:
        keys = config["params"]
        params = set(p for p in _match_parameters(model, keys) if p.requires_grad)
        params = params.difference(assigned_params)
        if params:
            assert params.isdisjoint(assigned_params)
            kwargs = {k: v for k, v in config.items() if k != "params"}
            assigned_groups.append({"params": list(params), **kwargs})
        assigned_params.update(params)
    # Default param group
    assigned_groups.append({"params": [p for p in model.parameters() if p not in assigned_params and p.requires_grad]})
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
