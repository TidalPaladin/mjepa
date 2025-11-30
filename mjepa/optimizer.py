from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple

import torch.nn as nn
import yaml
from torch.optim import AdamW, Muon, Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler, OneCycleLR


def _is_internal_2d_parameter(name: str, param: nn.Parameter) -> bool:
    """Check if parameter is 2D and internal (not bias, norm, or embedding)."""
    return param.ndim == 2 and "blocks" in name


@dataclass(frozen=True)
class OptimizerConfig:
    # Optimizer
    lr: float
    weight_decay: float
    betas: Tuple[float, float]
    fused: bool | None = True
    foreach: bool | None = None
    eps: float = 1e-8

    # Muon optimizer
    use_muon: bool = False
    muon_lr: float | None = None  # If None, uses lr
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5

    # Scheduler
    scheduled: bool = False
    pct_start: float = 0.05
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 10.0
    final_div_factor: float = 1e4

    # Parameter groups
    parameter_groups: List[Dict[str, Any]] = field(default_factory=list)

    def instantiate(
        self,
        model: nn.Module,
        total_steps: int,
        split_test_fn: Callable[[str, nn.Parameter], bool] = _is_internal_2d_parameter,
    ) -> Tuple[List[Optimizer], List[LRScheduler]]:
        if self.use_muon:
            return self._instantiate_with_muon(model, total_steps, split_test_fn)
        else:
            return self._instantiate_adamw_only(model, total_steps)

    def _instantiate_adamw_only(self, model: nn.Module, total_steps: int) -> Tuple[List[Optimizer], List[LRScheduler]]:
        parameter_groups = _assign_parameter_groups(model, self.parameter_groups)
        optimizer = AdamW(
            parameter_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            fused=self.fused,
            foreach=self.foreach,
            eps=self.eps,
        )
        scheduler = self._create_scheduler(optimizer, self.lr, total_steps)
        return [optimizer], [scheduler]

    def _instantiate_with_muon(
        self,
        model: nn.Module,
        total_steps: int,
        split_test_fn: Callable[[str, nn.Parameter], bool] = _is_internal_2d_parameter,
    ) -> Tuple[List[Optimizer], List[LRScheduler]]:
        # Split parameters into Muon (2D internal) and AdamW (others)
        muon_params, adamw_params = _split_muon_adamw_parameters(model, self.parameter_groups, split_test_fn)

        optimizers: List[Optimizer] = []
        schedulers: List[LRScheduler] = []

        # Create Muon optimizer for 2D internal parameters
        if muon_params:
            assert Muon is not None, "Muon optimizer should be available at this point"
            muon_lr = self.muon_lr if self.muon_lr is not None else self.lr
            muon_optimizer = Muon(
                muon_params,
                lr=muon_lr,
                momentum=self.muon_momentum,
                nesterov=self.muon_nesterov,
                ns_steps=self.muon_ns_steps,
            )
            muon_scheduler = self._create_scheduler(muon_optimizer, muon_lr, total_steps, warmup=False)
            optimizers.append(muon_optimizer)
            schedulers.append(muon_scheduler)

        # Create AdamW optimizer for other parameters
        if adamw_params:
            adamw_optimizer = AdamW(
                adamw_params,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
                fused=self.fused,
                foreach=self.foreach,
                eps=self.eps,
            )
            adamw_scheduler = self._create_scheduler(adamw_optimizer, self.lr, total_steps, warmup=False)
            optimizers.append(adamw_optimizer)
            schedulers.append(adamw_scheduler)

        return optimizers, schedulers

    def _create_scheduler(self, optimizer: Optimizer, lr: float, total_steps: int, warmup: bool = True) -> LRScheduler:
        if self.scheduled:
            return OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=self.pct_start,
                base_momentum=self.base_momentum,
                max_momentum=self.max_momentum,
                div_factor=self.div_factor if warmup else 1.0,
                final_div_factor=self.final_div_factor,
            )
        else:
            return LinearLR(
                optimizer,
                start_factor=1 / self.div_factor if warmup else 1.0,
                end_factor=1,
                total_iters=int(total_steps * self.pct_start),
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


def _split_muon_adamw_parameters(
    model: nn.Module,
    parameter_groups: List[Dict[str, Any]],
    split_test_fn: Callable[[str, nn.Parameter], bool] = _is_internal_2d_parameter,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split parameters into Muon (2D internal) and AdamW (others) groups.

    Args:
        model: Model to extract parameters from
        parameter_groups: Custom parameter group configurations

    Returns:
        Tuple of (muon_param_groups, adamw_param_groups)
    """
    # First assign parameter groups as usual
    assigned_groups = _assign_parameter_groups(model, parameter_groups)

    # Now split each group into Muon and AdamW compatible parameters
    muon_groups: List[Dict[str, Any]] = []
    adamw_groups: List[Dict[str, Any]] = []

    for group in assigned_groups:
        muon_params = []
        adamw_params = []

        # Get parameter names for filtering
        param_to_name = {param: name for name, param in model.named_parameters()}

        for param in group["params"]:
            name = param_to_name.get(param, "")
            if split_test_fn(name, param):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        # Create new groups with split parameters
        group_config = {k: v for k, v in group.items() if k != "params"}

        if muon_params:
            muon_groups.append({"params": muon_params, **group_config})
        if adamw_params:
            adamw_groups.append({"params": adamw_params, **group_config})

    return muon_groups, adamw_groups


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
