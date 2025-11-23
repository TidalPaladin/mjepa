import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, TypeVar
import torch.distributed as dist

import torch
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
        context_pos_emb: Whether to introduce positional encoding to the context.
        out_dim: Output dimension of the predictor.
            If ``None``, the output dimension will be the same as the input dimension.
    """

    def __init__(
        self,
        backbone: ViT,
        depth: int,
        out_dim: int | None = None,
    ):
        super().__init__()
        # NOTE: This specific setup seems very important to high CIFAR-10 probe performance.
        # Changes should be made with care.
        spatial_size = backbone.stem.tokenized_size(backbone.config.img_size)
        self.pos_enc_target = LearnablePosition(backbone.config.hidden_size, spatial_size)
        self.rope = backbone.rope

        # Predictor blocks and output projection
        self.blocks = nn.ModuleList([backbone.create_cross_attention_layer() for _ in range(depth)])

        self.predictor_proj = nn.Linear(backbone.config.hidden_size, out_dim or backbone.config.hidden_size)

    def forward(
        self,
        tokenized_size: Tuple[int, int],
        context: Tensor,
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
    ) -> Tensor:
        # Create positional encodings
        B, _ = target_mask.shape
        pos_target = apply_mask(target_mask, self.pos_enc_target(tokenized_size).expand(B, -1, -1))

        # Prepare inputs
        query = pos_target
        rope_q, rope_k = self.prepare_rope(tokenized_size, context_mask, target_mask, rope_seed=rope_seed)

        # Run query and context through predictor
        for block in self.blocks:
            query = block(query, context, rope_q=rope_q, rope_k=rope_k)

        return self.predictor_proj(query)

    if TYPE_CHECKING:

        def __call__(
            self,
            tokenized_size: Tuple[int, int],
            context: Tensor,
            context_mask: Tensor | None,
            target_mask: Tensor,
            rope_seed: int | None = None,
        ) -> Tensor:
            return self.forward(tokenized_size, context, context_mask, target_mask, rope_seed)

    def prepare_rope(
        self, tokenized_size: Tuple[int, int], context_mask: Tensor | None, target_mask: Tensor, rope_seed: int | None = None
    ) -> Tuple[Tensor | None, Tensor | None]:
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
) -> Tuple[Tensor, Tensor]:
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
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.lerp_(student_param, weight)


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


def is_gram_update_epoch(epoch: int, gram_start_epoch: int | None, gram_update_interval_epoch: int) -> bool:
    r"""Check if the current epoch is a Gram update epoch.

    Args:
        epoch: Current epoch.
        gram_start_epoch: The epoch at which to store a checkpoint and begin computing the Gram loss.
        gram_update_interval_epoch: The interval at which to update the Gram teacher after the initial setup.
    """
    if gram_start_epoch is None:
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
def compute_sigreg_loss(x: Tensor, global_step: int, num_slices: int = 256) -> Tensor:
    r"""Compute the LeJEPA SigREG loss.

    This loss encourages features to follow an isotropic Gaussian distribution.

    Args:
        x: Input tensor.
        global_step: Global step.
        num_slices: Number of slices to use for the projection.

    Returns:
        The SigREG loss.
    """
    B, L, D = x.shape

    proj_shape = (1, D, num_slices)
    with torch.random.fork_rng(devices=[x.device]):
        torch.random.manual_seed(global_step)
        A = torch.randn(proj_shape, device=x.device)
    A = F.normalize(A, dim=-2)

    t = torch.linspace(-5, 5, 17, device=x.device)
    exp_f = torch.exp(-0.5 * t**2)

    x_t = torch.bmm(x.type_as(A), A.expand(B, -1, -1)).unsqueeze(-1) * t
    ecf = (1j * x_t).exp().mean(-3)

    if dist.is_initialized():
        world_size = dist.get_world_size()
        ecf = dist.all_reduce(ecf, op=dist.ReduceOp.AVG)
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
        gram_epoch: The epoch at which to store a checkpoint and begin computing the Gram loss.
            If ``None``, the Gram loss will not be computed.
        gram_epoch: The epoch at which to store a checkpoint and begin computing the Gram loss.
            If ``None``, the Gram loss will not be computed.
        gram_update_interval_epoch: The interval at which to update the Gram teacher after the initial setup.
        gram_resolution_scale: The scale at which to feed inputs through the Gram teacher.
        gram_remove_neg: Whether to remove negative values from the Gram matrix.
    """

    context_ratio: float = 0.5
    target_ratio: float = 0.25
    scale: int = 4
    momentum: float = 0.99
    scheduled: bool = False
    predictor_depth: int = 4
    gram_teacher_epoch: int = 100
    gram_start_epoch: int | None = None
    gram_update_interval_epoch: int = 10
    gram_resolution_scale: float = 2.0
    gram_remove_neg: bool = False

    def __post_init__(self) -> None:
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
