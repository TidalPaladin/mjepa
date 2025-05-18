from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import yaml
from torch import Tensor
from vit import ViT
from vit.attention import AttentivePool
from vit.tokens import apply_mask, generate_non_overlapping_mask
from vit.pos_enc import create_grid, LearnablePosition, RelativeFactorizedPosition, LearnableFourierFeatures

class CrossAttentionPredictor(nn.Module):
    r"""
    Predicts targets from context embeddings.

    The predictor consists of the following components:
        - A positional encoding layer using relative factorized position encoding with a MLP to
          initialize the target queries. This is the same as the positional encoding used in the
          backbone.
        - A series of cross-attention layers between the target queries and the context.
        - A linear projection head to create the final output.

    Args:
        backbone: Backbone model with a :meth:`create_cross_attention_layer` method.
        depth: Depth of the predictor network.
        context_pos_emb: Whether to introduce positional encoding to the context.
        shared_pos_emb: Whether to use the backbone's positional encoding in the predictor.
        out_dim: Output dimension of the predictor.
            If ``None``, the output dimension will be the same as the input dimension.
    """

    def __init__(
        self,
        backbone: ViT,
        depth: int,
        shared_pos_emb: bool = False,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.pos_enc = backbone.stem.pos_enc if shared_pos_emb else deepcopy(backbone.stem.pos_enc)
        self.query = nn.Parameter(torch.randn(backbone.config.hidden_size))
        self.context_norm = nn.RMSNorm(backbone.config.hidden_size)

        # Predictor blocks and output projection
        self.blocks = nn.ModuleList([backbone.create_cross_attention_layer() for i in range(depth)])
        self.predictor_proj = nn.Linear(backbone.config.hidden_size, out_dim or backbone.config.hidden_size)

    def forward(
        self,
        tokenized_size: Tuple[int, int],
        context: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        # Create positional encodings
        B, L = target_mask.shape
        pos = apply_mask(target_mask, self.pos_enc(tokenized_size).expand(B, -1, -1))

        # Combine positional encodings with context and query
        context = self.context_norm(context)
        query = pos + self.query

        # Run query and context through predictor
        for block in self.blocks:
            query = block(query, context)

        return self.predictor_proj(query)


class AttentiveProbe(nn.Module):

    def __init__(self, hidden_size: int, out_dim: int, num_attention_heads: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.pool = AttentivePool(hidden_size, num_attention_heads)
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.pool(x)
        return self.proj(x)


class LinearProbe(nn.Module):

    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.proj(x)


@torch.no_grad()
def generate_masks(
    backbone: ViT,
    x: Tensor,
    context_ratio: float,
    target_ratio: float,
    scale: int,
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
    context_mask = backbone.create_mask(x, context_ratio, scale)

    # generate target mask - select non-ragged target mask from locations not in context mask
    target_mask = generate_non_overlapping_mask(context_mask, context_ratio, target_ratio)
    return context_mask, target_mask


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
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


def get_momentum(step: int, total_steps: int, momentum: float) -> float:
    r"""Linearly anneal momentum from the given value to 1.0 over the course of training.

    Args:
        step: Current step in the training loop.
        total_steps: Total number of steps in the training loop.
        momentum: The base momentum for the EMA update.

    Returns:
        The current momentum value.
    """
    return momentum + (1 - momentum) * (step / total_steps)


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
            from this value to 1.0 over the course of training.
        predictor_depth: Depth of the predictor network.
        probe_type: Type of probe to use (linear or attentive)
        context_pos_emb: Whether to introduce positional encoding to the context
            as part of the predictor network.
        shared_pos_emb: Whether to use the backbone's positional encoding in the predictor.
    """

    context_ratio: float = 0.5
    target_ratio: float = 0.25
    scale: int = 4
    momentum: float = 0.99
    predictor_depth: int = 4
    probe_type: Literal["attentive", "linear"] = "linear"
    context_pos_emb: bool = False
    shared_pos_emb: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.context_ratio <= 1:
            raise ValueError("context_ratio must be in the range (0, 1]")
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in the range (0, 1]")


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
