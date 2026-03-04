from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Literal, Protocol, cast

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler, OneCycleLR


ADAMW_OPTIMIZER_KIND = "adamw"
HYBRID_MUON_OPTIMIZER_KIND = "hybrid_muon"
ADAMW_SCHEDULER_KIND = "adamw"
HYBRID_MUON_SCHEDULER_KIND = "hybrid_muon"

MUON_COMPONENT_NAME = "muon"
ADAMW_COMPONENT_NAME = "adamw"

_MUON_GROUP_KEYS = {"lr", "weight_decay", "momentum", "nesterov", "ns_coefficients", "eps", "ns_steps", "adjust_lr_fn"}
_ADAMW_GROUP_KEYS = {
    "lr",
    "weight_decay",
    "betas",
    "eps",
    "amsgrad",
    "maximize",
    "foreach",
    "capturable",
    "differentiable",
    "fused",
}


class OptimizerLike(Protocol):
    param_groups: list[dict[str, Any]]

    def step(self, closure=None): ...
    def zero_grad(self, set_to_none: bool = True): ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]): ...


class SchedulerLike(Protocol):
    def step(self, epoch: int | None = None): ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]): ...


class CompositeOptimizer:
    def __init__(self, optimizers: dict[str, Optimizer]):
        if not optimizers:
            raise ValueError("CompositeOptimizer requires at least one optimizer")
        self.optimizers = optimizers
        self.defaults: dict[str, Any] = {}
        # Keep direct references so dynamic LR updates are visible.
        self.param_groups = [group for optimizer in self.optimizers.values() for group in optimizer.param_groups]
        self._mjepa_optimizer_kind = HYBRID_MUON_OPTIMIZER_KIND

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers.values():
            optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "_mjepa_optimizer_kind": self._mjepa_optimizer_kind,
            "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        checkpoint_kind = state_dict.get("_mjepa_optimizer_kind")
        if checkpoint_kind != HYBRID_MUON_OPTIMIZER_KIND:
            raise ValueError(
                f"Cannot load checkpoint optimizer kind {checkpoint_kind!r} into {HYBRID_MUON_OPTIMIZER_KIND!r}"
            )
        optimizer_state_dicts = state_dict.get("optimizers")
        if not isinstance(optimizer_state_dicts, Mapping):
            raise ValueError("Invalid composite optimizer state dict: missing 'optimizers' mapping")
        missing_components = [name for name in self.optimizers if name not in optimizer_state_dicts]
        if missing_components:
            raise ValueError(f"Missing optimizer state for components: {missing_components}")
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(cast(dict[str, Any], optimizer_state_dicts[name]))


class CompositeScheduler:
    def __init__(self, schedulers: dict[str, LRScheduler]):
        if not schedulers:
            raise ValueError("CompositeScheduler requires at least one scheduler")
        self.schedulers = schedulers
        self._mjepa_scheduler_kind = HYBRID_MUON_SCHEDULER_KIND
        self._last_lr = self.get_last_lr()

    def step(self, epoch: int | None = None):
        for scheduler in self.schedulers.values():
            scheduler.step(epoch)
        self._last_lr = self.get_last_lr()

    def get_last_lr(self) -> list[float]:
        return [lr for scheduler in self.schedulers.values() for lr in scheduler.get_last_lr()]

    def state_dict(self) -> dict[str, Any]:
        return {
            "_mjepa_scheduler_kind": self._mjepa_scheduler_kind,
            "schedulers": {name: scheduler.state_dict() for name, scheduler in self.schedulers.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        checkpoint_kind = state_dict.get("_mjepa_scheduler_kind")
        if checkpoint_kind != HYBRID_MUON_SCHEDULER_KIND:
            raise ValueError(
                f"Cannot load checkpoint scheduler kind {checkpoint_kind!r} into {HYBRID_MUON_SCHEDULER_KIND!r}"
            )
        scheduler_state_dicts = state_dict.get("schedulers")
        if not isinstance(scheduler_state_dicts, Mapping):
            raise ValueError("Invalid composite scheduler state dict: missing 'schedulers' mapping")
        missing_components = [name for name in self.schedulers if name not in scheduler_state_dicts]
        if missing_components:
            raise ValueError(f"Missing scheduler state for components: {missing_components}")
        for name, scheduler in self.schedulers.items():
            scheduler.load_state_dict(cast(dict[str, Any], scheduler_state_dicts[name]))
        self._last_lr = self.get_last_lr()


def get_optimizer_kind(optimizer: OptimizerLike) -> str:
    return cast(str, getattr(optimizer, "_mjepa_optimizer_kind", ADAMW_OPTIMIZER_KIND))


def get_scheduler_kind(scheduler: SchedulerLike) -> str:
    return cast(str, getattr(scheduler, "_mjepa_scheduler_kind", ADAMW_SCHEDULER_KIND))


def get_optimizer_kind_from_state_dict(state_dict: Mapping[str, Any]) -> str:
    return cast(str, state_dict.get("_mjepa_optimizer_kind", ADAMW_OPTIMIZER_KIND))


def get_scheduler_kind_from_state_dict(state_dict: Mapping[str, Any]) -> str:
    return cast(str, state_dict.get("_mjepa_scheduler_kind", ADAMW_SCHEDULER_KIND))


@dataclass(frozen=True)
class OptimizerConfig:
    # Optimizer
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    kind: Literal["adamw", "hybrid_muon"] = ADAMW_OPTIMIZER_KIND
    fused: bool | None = True
    foreach: bool | None = None
    eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
    muon_eps: float = 1e-7
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: Literal["original", "match_rms_adamw"] | None = None

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

    def _instantiate_scheduler(self, optimizer: Optimizer, total_steps: int) -> LRScheduler:
        if self.scheduled:
            max_lrs = [group["lr"] for group in optimizer.param_groups]
            return OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=self.pct_start,
                base_momentum=self.base_momentum,
                max_momentum=self.max_momentum,
                div_factor=self.div_factor,
                final_div_factor=self.final_div_factor,
            )
        return LinearLR(
            optimizer,
            start_factor=1 / self.div_factor,
            end_factor=1,
            total_iters=int(total_steps * self.pct_start),
        )

    def _instantiate_adamw(
        self, parameter_groups: list[dict[str, Any]], total_steps: int
    ) -> tuple[Optimizer, LRScheduler]:
        optimizer = AdamW(
            parameter_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            fused=self.fused,
            foreach=self.foreach,
            eps=self.eps,
        )
        scheduler = self._instantiate_scheduler(optimizer, total_steps)
        setattr(optimizer, "_mjepa_optimizer_kind", ADAMW_OPTIMIZER_KIND)
        setattr(scheduler, "_mjepa_scheduler_kind", ADAMW_SCHEDULER_KIND)
        return optimizer, scheduler

    def _instantiate_hybrid_muon(
        self, parameter_groups: list[dict[str, Any]], total_steps: int
    ) -> tuple[CompositeOptimizer, CompositeScheduler]:
        muon_cls = cast(Any, getattr(torch.optim, "Muon", None))
        if muon_cls is None:
            raise RuntimeError("torch.optim.Muon is unavailable. Upgrade PyTorch to a version that includes Muon.")

        muon_parameter_groups, adamw_parameter_groups = _split_parameter_groups_by_ndim(parameter_groups)

        optimizers: dict[str, Optimizer] = {}
        if muon_parameter_groups:
            optimizers[MUON_COMPONENT_NAME] = muon_cls(
                muon_parameter_groups,
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.muon_momentum,
                nesterov=self.muon_nesterov,
                ns_coefficients=self.muon_ns_coefficients,
                eps=self.muon_eps,
                ns_steps=self.muon_ns_steps,
                adjust_lr_fn=self.muon_adjust_lr_fn,
            )
        if adamw_parameter_groups:
            optimizers[ADAMW_COMPONENT_NAME] = AdamW(
                adamw_parameter_groups,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
                fused=self.fused,
                foreach=self.foreach,
                eps=self.eps,
            )
        if not optimizers:
            raise ValueError("No trainable parameters found for hybrid_muon optimizer")

        schedulers = {
            name: self._instantiate_scheduler(optimizer, total_steps) for name, optimizer in optimizers.items()
        }
        return CompositeOptimizer(optimizers), CompositeScheduler(schedulers)

    def instantiate(self, model: nn.Module, total_steps: int) -> tuple[OptimizerLike, SchedulerLike]:
        parameter_groups = _assign_parameter_groups(
            model,
            self.parameter_groups,
            skip_weight_decay_on_1d=self.skip_weight_decay_on_1d,
        )
        if self.kind == ADAMW_OPTIMIZER_KIND:
            return self._instantiate_adamw(parameter_groups, total_steps)
        if self.kind == HYBRID_MUON_OPTIMIZER_KIND:
            return self._instantiate_hybrid_muon(parameter_groups, total_steps)
        raise ValueError(f"Unsupported optimizer kind: {self.kind!r}")

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


def _split_parameter_groups_by_ndim(
    parameter_groups: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    muon_parameter_groups: list[dict[str, Any]] = []
    adamw_parameter_groups: list[dict[str, Any]] = []
    for group in parameter_groups:
        params = list(group["params"])
        kwargs = {k: v for k, v in group.items() if k != "params"}
        muon_params = [param for param in params if param.ndim == 2]
        adamw_params = [param for param in params if param.ndim != 2]
        if muon_params:
            muon_kwargs = {k: v for k, v in kwargs.items() if k in _MUON_GROUP_KEYS}
            muon_parameter_groups.append({"params": muon_params, **muon_kwargs})
        if adamw_params:
            adamw_kwargs = {k: v for k, v in kwargs.items() if k in _ADAMW_GROUP_KEYS}
            adamw_parameter_groups.append({"params": adamw_params, **adamw_kwargs})
    return muon_parameter_groups, adamw_parameter_groups


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
