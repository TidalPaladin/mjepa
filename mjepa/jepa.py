from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

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
        spatial_size = backbone.stem.tokenized_size(backbone.config.img_size)
        self.pos_enc_target = LearnablePosition(
            backbone.config.hidden_size, spatial_size, device=device, dtype=backbone.config.dtype
        )
        self.rope = backbone.rope
        predictor_out_dim = out_dim or backbone.config.hidden_size

        # Predictor blocks and output projection
        self.blocks = nn.ModuleList([backbone.create_cross_attention_layer(device=device) for _ in range(depth)])
        if disable_predictor_regularizers:
            self._disable_predictor_regularizers()

        self.predictor_proj = nn.Linear(
            backbone.config.hidden_size,
            predictor_out_dim,
            device=device,
            dtype=backbone.config.dtype,
        )
        self.cls_global_head = nn.Linear(
            backbone.config.hidden_size,
            predictor_out_dim,
            device=device,
            dtype=backbone.config.dtype,
        )

    def _disable_predictor_regularizers(self) -> None:
        # Force predictor regularizers to zero without changing the backbone configuration.
        no_regularization = 0.0
        for block in self.blocks:
            block = cast(Any, block)
            block.drop_path_rate = no_regularization
            block.cross_attention.dropout.p = no_regularization
            block.cross_attention.attention_dropout.p = no_regularization
            block.mlp.dropout.p = no_regularization

    def forward(
        self,
        tokenized_size: tuple[int, int],
        context: Tensor,
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
        num_extra_context_tokens: int = 0,
    ) -> Tensor:
        # Create positional encodings
        B, _ = target_mask.shape
        pos_target = apply_mask(target_mask, self.pos_enc_target(tokenized_size).expand(B, -1, -1))

        # Prepare inputs
        query = pos_target
        rope_q, rope_k = self.prepare_rope(
            tokenized_size,
            context_mask,
            target_mask,
            rope_seed=rope_seed,
            num_extra_context_tokens=num_extra_context_tokens,
        )

        # Run query and context through predictor
        for block in self.blocks:
            query = block(query, context, rope_q=rope_q, rope_k=rope_k)

        return self.predictor_proj(query)

    def forward_cls_global(self, cls_tokens: Tensor) -> Tensor:
        return self.cls_global_head(cls_tokens.mean(dim=1))

    if TYPE_CHECKING:

        def __call__(
            self,
            tokenized_size: tuple[int, int],
            context: Tensor,
            context_mask: Tensor | None,
            target_mask: Tensor,
            rope_seed: int | None = None,
            num_extra_context_tokens: int = 0,
        ) -> Tensor:
            return self.forward(
                tokenized_size,
                context,
                context_mask,
                target_mask,
                rope_seed,
                num_extra_context_tokens,
            )

    def prepare_rope(
        self,
        tokenized_size: tuple[int, int],
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
        num_extra_context_tokens: int = 0,
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
            if num_extra_context_tokens > 0:
                identity_shape = (B, num_extra_context_tokens, sin_k.shape[-1])
                sin_extra = sin_k.new_zeros(identity_shape)
                cos_extra = cos_k.new_ones(identity_shape)
                sin_k = torch.cat([sin_k, sin_extra], dim=1)
                cos_k = torch.cat([cos_k, cos_extra], dim=1)
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
    r"""Generate non-overlapping and non-ragged context and target masks.

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

    # generate target mask - select non-ragged target mask from locations not in context mask
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
    torch._foreach_lerp_(list(teacher.parameters()), list(student.parameters()), weight)


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


def setup_teacher(backbone: M) -> M:
    r"""Create a teacher model from a backbone model.

    The teacher will have parameters frozen and will be set to evaluation mode.

    Args:
        backbone: Backbone model to create a teacher from.

    Returns:
        Teacher model.
    """
    teacher = deepcopy(backbone)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.compile(fullgraph=True)
def compute_gram_loss(student: Tensor, teacher: Tensor, normalize: bool = True, remove_neg: bool = True) -> Tensor:
    r"""Compute the Gram loss between student features and Gram target features.

    Args:
        student: Student features.
        teacher: Gram target features.
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


@torch.compile
def compute_sigreg_loss(x: Tensor, global_step: int, num_slices: int = 256) -> Tensor:
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
        gram_warmup_epochs: Number of epochs over which the Gram loss weight is linearly annealed
            from ``0`` to ``gram_loss_weight``.
        gram_remove_neg: Whether to remove negative values from the Gram matrix.
        gram_loss_weight: The coefficient of the Gram loss.
        sigreg_loss_weight: The coefficient of the SigREG loss.
    """

    context_ratio: float = 0.5
    target_ratio: float = 0.25
    scale: int = 4
    momentum: float = 0.99
    scheduled: bool = False
    predictor_depth: int = 4
    disable_predictor_regularizers: bool = False
    gram_warmup_epochs: int = 700
    gram_remove_neg: bool = False
    gram_loss_weight: float = 1.0
    sigreg_loss_weight: float = 1e-4

    def __post_init__(self) -> None:
        if not 0 < self.context_ratio <= 1:
            raise ValueError("context_ratio must be in the range (0, 1]")
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in the range (0, 1]")
        if self.gram_warmup_epochs < 0:
            raise ValueError("gram_warmup_epochs must be a non-negative integer")
        if self.gram_loss_weight <= 0:
            raise ValueError("gram_loss_weight must be a positive float")
        if self.sigreg_loss_weight < 0:
            raise ValueError("sigreg_loss_weight must be a non-negative float")


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
