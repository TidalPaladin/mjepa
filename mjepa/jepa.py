import math
from copy import deepcopy
from dataclasses import dataclass, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from vit import ViT
from vit.pos_enc import LearnablePosition
from vit.tokens import apply_mask, generate_non_overlapping_mask


torch._dynamo.config.cache_size_limit = 1024 * 1024 * 1024 * 1024

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}
PREDICTOR_PROJ_INIT_STD = 0.02
_AUTOCAST_DTYPES = {torch.bfloat16, torch.float16}
MSE_JEPA_LOSS_KIND = "mse"
COSINE_JEPA_LOSS_KIND = "cosine"
JEPALossKind: TypeAlias = Literal["mse", "cosine"]
JEPA_LOSS_KINDS: tuple[JEPALossKind, ...] = (MSE_JEPA_LOSS_KIND, COSINE_JEPA_LOSS_KIND)


def _normalize_dtype(dtype: torch.dtype | str | None, field_name: str) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        normalized = _DTYPE_MAP.get(dtype.removeprefix("torch."))
        if normalized is not None:
            return normalized
    raise ValueError(f"Unsupported dtype for {field_name}: {dtype!r}")


def _sync_module_config_dtype(model: nn.Module, dtype: torch.dtype) -> None:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "dtype"):
        return

    if is_dataclass(config):
        updated_config = replace(cast(Any, config), dtype=dtype)
        config_attr = "_config" if hasattr(model, "_config") else "config"
        setattr(model, config_attr, updated_config)
        return

    setattr(config, "dtype", dtype)


def autocast_context(device_type: str, dtype: torch.dtype):
    enabled = dtype in _AUTOCAST_DTYPES
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)


class CrossAttentionPredictor(nn.Module):
    r"""
    Predicts targets from context embeddings.

    The predictor consists of the following components:
        - Separate learnable position encodings for both context and target embeddings
        - A series of cross-attention layers between the target queries and the context.
        - A linear projection head to create the final output.

    Args:
        backbone: Backbone model with a :meth:`create_cross_attention_layer` method.
        depth: Depth of the predictor network.
        out_dim: Output dimension of the predictor.
            If ``None``, the output dimension will be the same as the input dimension.
        device: Device to place the predictor on.
        disable_predictor_regularizers: Whether to force predictor stochastic depth, hidden dropout,
            and attention dropout to ``0.0``.
    """

    def __init__(
        self,
        backbone: ViT,
        depth: int,
        out_dim: int | None = None,
        device: torch.device | None = None,
        disable_predictor_regularizers: bool = False,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": backbone.config.dtype}
        self.hidden_size = backbone.config.hidden_size
        self.out_dim = out_dim or backbone.config.hidden_size
        spatial_size = backbone.stem.tokenized_size(backbone.config.img_size)
        self.pos_enc_target = LearnablePosition(backbone.config.hidden_size, spatial_size, **factory_kwargs)
        self.rope = backbone.rope

        # Predictor blocks and output projection
        self.blocks = nn.ModuleList([self._create_block(backbone, device) for _ in range(depth)])
        if disable_predictor_regularizers:
            self._disable_predictor_regularizers()

        self.predictor_proj = self._create_projection(device=device, dtype=backbone.config.dtype)
        self.predictor_proj_shallow: nn.Linear | None = None

    @property
    def predictor_dtype(self) -> torch.dtype:
        return self.predictor_proj.weight.dtype

    @staticmethod
    def _create_block(backbone: ViT, device: torch.device | None) -> nn.Module:
        return backbone.create_cross_attention_layer(device=device, dtype=backbone.config.dtype)

    @staticmethod
    def _reset_projection_parameters(projection: nn.Linear) -> None:
        # Teacher targets are RMS-normalized and the predictor query starts at small positional scales,
        # so keep the output head small and zero-centered at initialization.
        nn.init.trunc_normal_(projection.weight, std=PREDICTOR_PROJ_INIT_STD)
        if projection.bias is not None:
            nn.init.zeros_(projection.bias)

    def _create_projection(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> nn.Linear:
        projection = nn.Linear(
            self.hidden_size,
            self.out_dim,
            device=device,
            dtype=dtype,
        )
        self._reset_projection_parameters(projection)
        return projection

    def enable_shallow_head(self) -> None:
        if self.predictor_proj_shallow is not None:
            return

        self.predictor_proj_shallow = self._create_projection(
            device=self.predictor_proj.weight.device,
            dtype=self.predictor_proj.weight.dtype,
        )

    @staticmethod
    def _project(query: Tensor, projection: nn.Linear | None) -> Tensor | None:
        return projection(query).float() if projection is not None else None

    def _disable_predictor_regularizers(self) -> None:
        # Force predictor regularizers to zero without changing the backbone configuration.
        no_regularization = 0.0
        for block in self.blocks:
            block = cast(Any, block)
            block.drop_path_rate = no_regularization
            block.cross_attention.dropout.p = no_regularization
            block.cross_attention.attention_dropout.p = no_regularization
            block.mlp.dropout.p = no_regularization

    def forward_features(
        self,
        tokenized_size: tuple[int, int],
        context: Tensor,
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
    ) -> Tensor:
        with autocast_context(context.device.type, self.predictor_dtype):
            # Create positional encodings
            B, _ = target_mask.shape
            pos_target = apply_mask(target_mask, self.pos_enc_target(tokenized_size).expand(B, -1, -1))

            # Prepare inputs
            query = pos_target.to(dtype=self.predictor_dtype)
            context = context.to(dtype=self.predictor_dtype)
            rope_q, rope_k = self.prepare_rope(tokenized_size, context_mask, target_mask, rope_seed=rope_seed)

            # Run query and context through predictor
            for block in self.blocks:
                query = block(query, context, rope_q=rope_q, rope_k=rope_k)

            return query

    def forward_heads(
        self,
        tokenized_size: tuple[int, int],
        context: Tensor,
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        query = self.forward_features(tokenized_size, context, context_mask, target_mask, rope_seed=rope_seed)
        with autocast_context(query.device.type, self.predictor_dtype):
            deep_output = self.predictor_proj(query).float()
            shallow_output = self._project(query, self.predictor_proj_shallow)
        return deep_output, shallow_output

    def forward(
        self,
        tokenized_size: tuple[int, int],
        context: Tensor,
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
    ) -> Tensor:
        deep_output, _ = self.forward_heads(tokenized_size, context, context_mask, target_mask, rope_seed=rope_seed)
        return deep_output

    def forward_shallow(
        self,
        tokenized_size: tuple[int, int],
        context: Tensor,
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
    ) -> Tensor:
        _, shallow_output = self.forward_heads(tokenized_size, context, context_mask, target_mask, rope_seed=rope_seed)
        if shallow_output is None:
            raise ValueError("Shallow predictor head is not enabled")
        return shallow_output

    if TYPE_CHECKING:

        def __call__(
            self,
            tokenized_size: tuple[int, int],
            context: Tensor,
            context_mask: Tensor | None,
            target_mask: Tensor,
            rope_seed: int | None = None,
        ) -> Tensor:
            return self.forward(tokenized_size, context, context_mask, target_mask, rope_seed)

    def prepare_rope(
        self,
        tokenized_size: tuple[int, int],
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.rope is None:
            return None, None

        H, W = tokenized_size
        rope = self.rope(H=H, W=W, rope_seed=rope_seed)
        sin, cos = rope
        B = target_mask.shape[0]

        sin_q = apply_mask(target_mask, sin[None].expand(B, -1, -1))
        cos_q = apply_mask(target_mask, cos[None].expand(B, -1, -1))
        rope_q = torch.stack([sin_q[:, None, ...], cos_q[:, None, ...]], dim=0)

        if context_mask is not None:
            sin_k = apply_mask(context_mask, sin[None].expand(B, -1, -1))
            cos_k = apply_mask(context_mask, cos[None].expand(B, -1, -1))
            rope_k = torch.stack([sin_k[:, None, ...], cos_k[:, None, ...]], dim=0)
        else:
            rope_k = None
        return rope_q, rope_k


@torch.no_grad()
def generate_masks(
    backbone: ViT,
    x: Tensor,
    context_ratio: float,
    target_ratio: float,
    scale: int,
    roll: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""Generate context masks and the corresponding prediction target masks.

    Args:
        backbone: Backbone implementing a :meth:`create_mask` method.
        x: Input tensor to generate the masks for.
        context_ratio: Ratio of the input to sample as context.
        target_ratio: Ratio of the input to sample as a prediction target.
        scale: Integer scale at which to mask contiguous blocks of tokens.

    Returns:
        Tuple of context and target masks.
    """
    # generate context mask - will always be non-ragged
    context_mask = backbone.create_mask(x, context_ratio, scale, roll=roll)

    if target_ratio == 1.0:
        # Full target coverage intentionally includes context tokens.
        target_mask = torch.ones_like(context_mask)
    else:
        # For partial targets, keep the existing non-overlapping behavior.
        target_mask = generate_non_overlapping_mask(context_mask, context_ratio, target_ratio)
    return context_mask, target_mask


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float = 0.0) -> None:
    r"""Update the teacher model with the student model using EMA.

    Args:
        student: Student model.
        teacher: Teacher model.
        momentum: Momentum for the EMA update.
    """
    assert 0 <= momentum <= 1.0, f"Momentum must be in the range [0, 1], got {momentum}"
    if momentum == 1.0:
        return
    weight = 1 - momentum
    teacher_params = [cast(Tensor, param) for param in teacher.parameters()]
    student_params = [cast(Tensor, param) for param in student.parameters()]
    matching_dtypes = all(
        teacher_param.dtype == student_param.dtype and teacher_param.device == student_param.device
        for teacher_param, student_param in zip(teacher_params, student_params)
    )
    if matching_dtypes:
        torch._foreach_lerp_(teacher_params, student_params, weight)
        return

    for teacher_param, student_param in zip(teacher_params, student_params):
        teacher_param.lerp_(student_param.to(device=teacher_param.device, dtype=teacher_param.dtype), weight)


def get_momentum(step: int, total_steps: int, momentum: float, scheduled: bool = False) -> float:
    r"""Linearly anneal momentum from the given value to 1.0 over the course of training.

    Args:
        step: Current step in the training loop.
        total_steps: Total number of steps in the training loop.
        momentum: The base momentum for the EMA update.
        scheduled: Whether to schedule the momentum.

    Returns:
        The current momentum value.
    """
    return min(momentum + (1 - momentum) * (step / total_steps), 1.0) if scheduled else momentum


M = TypeVar("M", bound=nn.Module)


def setup_teacher(backbone: M, dtype: torch.dtype | str | None = None) -> M:
    r"""Create a teacher model from a backbone model.

    The teacher will have parameters frozen and will be set to evaluation mode.

    Args:
        backbone: Backbone model to create a teacher from.
        dtype: Optional dtype override for the copied teacher weights.

    Returns:
        Teacher model.
    """
    teacher = deepcopy(backbone)
    teacher_dtype = _normalize_dtype(dtype, "teacher dtype")
    if teacher_dtype is not None:
        teacher = cast(M, teacher.to(dtype=teacher_dtype))
        _sync_module_config_dtype(teacher, teacher_dtype)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.compile(fullgraph=True)
def compute_gram_loss(student: Tensor, teacher: Tensor, normalize: bool = True, remove_neg: bool = True) -> Tensor:
    r"""Compute the Gram loss between student features and the Gram teacher's features.

    Args:
        student: Student features.
        teacher: Gram teacher features.
        normalize: Whether to normalize the features.
        remove_neg: Whether to remove negative values from the Gram matrix.

    Shapes:
        student: :math:`(*, L, D)`
        teacher: :math:`(*, L, D)`
        Output: Scalar

    Returns:
        The Gram loss.
    """
    student = F.normalize(student, dim=-1) if normalize else student
    teacher = F.normalize(teacher, dim=-1) if normalize else teacher

    student_sim = student.bmm(student.mT)
    teacher_sim = teacher.bmm(teacher.mT)
    if remove_neg:
        teacher_sim = teacher_sim.clamp(min=0.0)
        student_sim = student_sim.clamp(min=0.0)

    return F.mse_loss(student_sim, teacher_sim)


def compute_jepa_prediction_loss(
    student: Tensor,
    teacher: Tensor,
    kind: JEPALossKind = MSE_JEPA_LOSS_KIND,
) -> Tensor:
    r"""Compute the JEPA reconstruction loss between student predictions and teacher targets.

    Args:
        student: Student predictions.
        teacher: Teacher targets.
        kind: Reconstruction loss kind to apply.

    Shapes:
        student: :math:`(*, L, D)`
        teacher: :math:`(*, L, D)`
        Output: Scalar

    Returns:
        The reconstruction loss.
    """
    student = student.float()
    teacher = teacher.float()

    if kind == MSE_JEPA_LOSS_KIND:
        return F.mse_loss(student, teacher)
    if kind == COSINE_JEPA_LOSS_KIND:
        return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()
    raise ValueError(f"Unsupported JEPA loss kind: {kind!r}")


def is_gram_update_epoch(epoch: int, gram_start_epoch: int | None, gram_update_interval_epoch: int) -> bool:
    r"""Check if the current epoch is a Gram update epoch.

    Args:
        epoch: Current epoch.
        gram_start_epoch: The epoch at which to store a checkpoint and begin computing the Gram loss.
        gram_update_interval_epoch: The interval at which to update the Gram teacher after the initial setup.
    """
    if gram_start_epoch is None or gram_update_interval_epoch <= 0:
        return False
    return epoch > gram_start_epoch and (epoch - gram_start_epoch) % gram_update_interval_epoch == 0


def forward_gram_teacher(
    gram_teacher: ViT,
    img: Tensor,
    rope_seed: int | None = None,
    resolution_scale: float = 1.0,
) -> Tensor:
    r"""Forward pass through the Gram teacher.

    Applies image upsampling and downsampling to the input and output, respectively.

    Args:
        gram_teacher: Gram teacher model.
        img: Input image.
        rope_seed: Rope seed.
        resolution_scale: Resolution scale.

    Shapes:
        img: :math:`(*, C, H, W)`
        Output: :math:`(*, L, D)`

    Returns:
        Output features.
    """
    # Resize input according to resolution scale, tracking the tokenized size of the input and output
    target_tokenized_size = gram_teacher.stem.tokenized_size(img.shape[-2:])
    img = F.interpolate(img, scale_factor=resolution_scale, mode="bilinear", align_corners=False)
    output_tokenized_size = gram_teacher.stem.tokenized_size(img.shape[-2:])

    assert not gram_teacher.training, "Gram teacher must be in evaluation mode"

    # Forward pass and resize output features to original size
    gram_teacher_output = gram_teacher(img, rope_seed=rope_seed).visual_tokens
    B, _, D = gram_teacher_output.shape
    gram_teacher_output = gram_teacher_output.movedim(1, -1).reshape(B, D, *output_tokenized_size)
    gram_teacher_output = F.interpolate(
        gram_teacher_output, scale_factor=1 / resolution_scale, mode="bilinear", align_corners=False
    )
    gram_teacher_output = gram_teacher_output.flatten(2).movedim(-1, 1)

    assert gram_teacher_output.shape == (B, math.prod(target_tokenized_size), D)
    return gram_teacher_output


@torch.compile
def _compute_sigreg_loss_impl(x: Tensor, global_step: int, num_slices: int = 256) -> Tensor:
    r"""Compute the LeJEPA SigREG loss.

    This loss encourages features to follow an isotropic Gaussian distribution.

    Args:
        x: Input tensor.
        global_step: Global step.
        num_slices: Number of slices to use for the projection.

    Shapes:
        x: :math:`(*, L, D)`
        Output: Scalar

    Returns:
        The SigREG loss.
    """
    B, L, D = x.shape

    proj_shape = (1, D, num_slices)
    # Only pass CUDA devices to fork_rng to avoid CUDA initialization on CPU
    fork_devices = [x.device] if x.device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.random.manual_seed(global_step)
        A = torch.randn(proj_shape, device=x.device)
    A = F.normalize(A, dim=-2)

    t = torch.linspace(-5, 5, 17, device=x.device)
    exp_f = torch.exp(-0.5 * t**2)

    x_t = torch.bmm(x.type_as(A), A.expand(B, -1, -1)).unsqueeze(-1) * t
    ecf = (1j * x_t).exp().mean(-3)

    if dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(ecf, op=dist.ReduceOp.AVG)
        assert isinstance(ecf, Tensor)
    else:
        world_size = 1

    err = (ecf - exp_f).abs().square().mul(exp_f)
    N = L * world_size
    T = torch.trapz(err, t, dim=-1) * N
    assert T.shape == (B, num_slices)
    return T.mean()


def compute_sigreg_loss(x: Tensor, global_step: int, num_slices: int = 256) -> Tensor:
    r"""Compute the LeJEPA SigREG loss.

    Args:
        x: Input tensor with a non-empty token dimension.
        global_step: Global step.
        num_slices: Number of slices to use for the projection.

    Shapes:
        x: :math:`(*, L, D)` with :math:`L > 0`
        Output: Scalar

    Returns:
        The SigREG loss.
    """
    if x.shape[-2] <= 0:
        raise ValueError("x must contain at least one token")
    return _compute_sigreg_loss_impl(x, global_step, num_slices)


@dataclass
class JEPAConfig:
    """
    Configuration for JEPA related hyperparameters.

    Args:
        context_ratio: Ratio of the input to sample as context.
        target_ratio: Ratio of the input to sample as a prediction target.
        scale: Integer scale at which to sample contiguous blocks of context tokens.
            Increasing this ensures more adjacent tokens appear together in the context.
        momentum: The base momentum for the EMA update. Momentum will be linearly annealed
            from this value to 1.0 over the course of training if scheduled is ``True``.
        scheduled: Whether to schedule the momentum.
        predictor_depth: Depth of the predictor network.
        disable_predictor_regularizers: Whether to force predictor stochastic depth, hidden dropout,
            and attention dropout to ``0.0``.
        teacher_dtype: Optional dtype override for the teacher and Gram teacher weights.
        gram_teacher_epoch: Epoch when the Gram teacher is first synchronized from the teacher.
        gram_start_epoch: The epoch at which to begin computing the Gram loss.
            If ``None``, the Gram loss will not be computed.
        gram_update_interval_epoch: The interval at which to update the Gram teacher after the initial setup.
        gram_resolution_scale: The scale at which to feed inputs through the Gram teacher.
        gram_remove_neg: Whether to remove negative values from the Gram matrix.
        gram_loss_weight: The coefficient of the Gram loss.
        sigreg_loss_weight: The coefficient of the SigREG loss.
        stem_jepa_loss_weight: The coefficient of the stem-target JEPA loss.
        jepa_loss_kind: Reconstruction loss kind applied to both JEPA prediction losses.
    """

    context_ratio: float = 0.5
    target_ratio: float = 0.25
    scale: int = 4
    momentum: float = 0.99
    scheduled: bool = False
    predictor_depth: int = 4
    disable_predictor_regularizers: bool = False
    teacher_dtype: torch.dtype | str | None = None
    gram_teacher_epoch: int = 100
    gram_start_epoch: int | None = None
    gram_update_interval_epoch: int = 10
    gram_resolution_scale: float = 2.0
    gram_remove_neg: bool = False
    gram_loss_weight: float = 1.0
    sigreg_loss_weight: float = 1e-4
    stem_jepa_loss_weight: float = 0.0
    jepa_loss_kind: JEPALossKind = MSE_JEPA_LOSS_KIND

    def __post_init__(self) -> None:
        self.teacher_dtype = _normalize_dtype(self.teacher_dtype, "teacher_dtype")
        if not 0 < self.context_ratio <= 1:
            raise ValueError("context_ratio must be in the range (0, 1]")
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in the range (0, 1]")
        if self.gram_teacher_epoch <= 0:
            raise ValueError("gram_teacher_epoch must be a positive integer")
        if self.gram_start_epoch is not None and self.gram_start_epoch <= 0:
            raise ValueError("gram_start_epoch must be a positive integer or None")
        if self.gram_update_interval_epoch < 0:
            raise ValueError("gram_update_interval_epoch must be a non-negative integer")
        if self.gram_start_epoch is not None and self.gram_start_epoch < self.gram_teacher_epoch:
            raise ValueError("gram_start_epoch must be greater than or equal to gram_teacher_epoch")
        if self.gram_resolution_scale <= 0:
            raise ValueError("gram_resolution_scale must be a positive float")
        if self.gram_loss_weight <= 0:
            raise ValueError("gram_loss_weight must be a positive float")
        if self.sigreg_loss_weight < 0:
            raise ValueError("sigreg_loss_weight must be a non-negative float")
        if self.stem_jepa_loss_weight < 0:
            raise ValueError("stem_jepa_loss_weight must be a non-negative float")
        if self.jepa_loss_kind not in JEPA_LOSS_KINDS:
            raise ValueError(f"jepa_loss_kind must be one of {list(JEPA_LOSS_KINDS)}")


def config_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    return JEPAConfig(**values)


def register_constructors():
    tags = [
        "tag:yaml.org,2002:python/object:mjepa.JEPAConfig",
    ]
    loaders = [yaml.SafeLoader, yaml.FullLoader, yaml.UnsafeLoader]
    for tag in tags:
        for loader in loaders:
            loader.add_constructor(tag, config_constructor)


register_constructors()
