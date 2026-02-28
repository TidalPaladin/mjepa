from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from vit import ViT, ViTFeatures
from vit.tokens import apply_mask

from .jepa import (
    CrossAttentionPredictor,
    JEPAConfig,
    compute_sigreg_loss,
    generate_masks,
    get_momentum,
    setup_teacher,
    update_teacher,
)
from .trainer import assert_all_ranks_synced, assert_all_trainable_params_have_grad


SIGREG_NUM_SLICES = 256


@dataclass
class MJEPALosses:
    jepa_loss: Tensor
    jepa_loss_cls: Tensor | float
    sigreg_loss: Tensor | float
    dense_pred_loss: Tensor | float

    dense_pred_loss_weight: float = 1.0
    sigreg_loss_weight: float = 1e-4

    def reduce(self) -> Tensor:
        loss = (
            self.jepa_loss
            + self.jepa_loss_cls
            + self.dense_pred_loss * self.dense_pred_loss_weight
            + self.sigreg_loss * self.sigreg_loss_weight
        )
        assert isinstance(loss, Tensor)
        return loss


@dataclass
class MJEPAPredictions:
    pred: Tensor
    pred_with_cls: Tensor | None
    dense_pred: Tensor
    student_output: ViTFeatures
    teacher_output: ViTFeatures
    teacher_stem_output: Tensor
    context_mask: Tensor
    target_mask: Tensor

    probes: dict[str, Tensor] = field(default_factory=dict)


class MJEPA(nn.Module):
    def __init__(
        self,
        config: JEPAConfig,
        backbone: ViT,
        predictor: CrossAttentionPredictor,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.config = config
        self.student = backbone
        self.teacher = setup_teacher(backbone)
        self.predictor = predictor
        self.dense_pred_head = nn.Linear(
            backbone.config.hidden_size,
            backbone.config.hidden_size,
            dtype=backbone.config.dtype,
        )
        self.dtype = dtype

    @property
    def img_size(self) -> tuple[int, int]:
        return cast(tuple[int, int], self.student.config.img_size)

    def forward_teacher(self, x: Tensor) -> ViTFeatures:
        self.teacher.eval()
        with torch.autocast(device_type=x.device.type, dtype=self.dtype), torch.inference_mode():
            output = self.teacher(x)
        return ViTFeatures(
            output.dense_features.clone(), output.num_register_tokens, output.num_cls_tokens, output.tokenized_size
        )

    def forward_student(self, x: Tensor, context_mask: Tensor, rope_seed: int | None = None) -> ViTFeatures:
        with torch.autocast(device_type=x.device.type, dtype=self.dtype):
            return self.student(x, mask=context_mask, rope_seed=rope_seed)

    def forward_predictor(
        self,
        tokenized_size: tuple[int, int],
        context: Tensor,
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
    ) -> Tensor:
        with torch.autocast(device_type=context.device.type, dtype=self.dtype):
            return self.predictor(tokenized_size, context, context_mask, target_mask, rope_seed=rope_seed)

    def forward_probe(self, features: ViTFeatures) -> dict[str, Tensor]:
        return dict()

    def forward_teacher_stem(self, x: Tensor, context_mask: Tensor) -> Tensor:
        self.teacher.eval()
        with torch.autocast(device_type=x.device.type, dtype=self.dtype), torch.inference_mode():
            teacher_stem_output = self.teacher.stem(x)
            teacher_stem_output = apply_mask(context_mask, teacher_stem_output, fill_value=None)
        return teacher_stem_output.clone()

    def forward_dense_pred(self, student_visual_tokens: Tensor) -> Tensor:
        with torch.autocast(device_type=student_visual_tokens.device.type, dtype=self.dtype):
            return self.dense_pred_head(student_visual_tokens)

    def compute_losses(self, output: MJEPAPredictions, step: int, epoch: int) -> MJEPALosses:
        del epoch  # Kept in the signature for call-site compatibility with the training loop.
        # Compute JEPA loss
        target = apply_mask(output.target_mask, output.teacher_output.visual_tokens, fill_value=None).float()
        jepa_loss = F.mse_loss(output.pred.float(), target)
        jepa_loss_cls = F.mse_loss(output.pred_with_cls.float(), target) if output.pred_with_cls is not None else 0.0

        # Compute SigREG loss
        sigreg_loss = (
            compute_sigreg_loss(
                output.student_output.cls_tokens.transpose(0, 1).float(), step, num_slices=SIGREG_NUM_SLICES
            )
            if self.config.sigreg_loss_weight > 0
            else 0.0
        )

        # Compute dense prediction loss
        dense_pred_loss = F.mse_loss(output.dense_pred.float(), output.teacher_stem_output.float())

        return MJEPALosses(
            jepa_loss=jepa_loss,
            jepa_loss_cls=jepa_loss_cls,
            sigreg_loss=sigreg_loss,
            dense_pred_loss=dense_pred_loss,
            dense_pred_loss_weight=self.config.dense_pred_loss_weight,
            sigreg_loss_weight=self.config.sigreg_loss_weight,
        )

    def forward(self, x: Tensor, jepa_scale: int, epoch: int) -> MJEPAPredictions:
        # NOTE: For DDP to work, all components must execute in the forward pass when training
        context_mask, target_mask = generate_masks(
            self.student, x, self.config.context_ratio, self.config.target_ratio, jepa_scale
        )

        rope_seed = int(torch.randint(0, 1000000, (1,)).item())
        # Teacher forward pass
        teacher_output = self.forward_teacher(x)
        teacher_stem_output = self.forward_teacher_stem(x, context_mask)

        Ht, Wt = cast(tuple[int, int], self.student.stem.tokenized_size(x.shape[-2:]))

        student_output = self.forward_student(x, context_mask, rope_seed=rope_seed)
        dense_pred = self.forward_dense_pred(student_output.visual_tokens)
        pred = self.forward_predictor(
            (Ht, Wt), student_output.visual_tokens, context_mask, target_mask, rope_seed=rope_seed
        )
        pred_with_cls = (
            self.forward_predictor((Ht, Wt), student_output.cls_tokens, None, target_mask, rope_seed=rope_seed)
            if student_output.cls_tokens.numel()
            else None
        )

        with torch.autocast(device_type=pred.device.type, dtype=self.dtype):
            probes = self.forward_probe(teacher_output)

        return MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            dense_pred=dense_pred,
            student_output=student_output,
            teacher_output=teacher_output,
            teacher_stem_output=teacher_stem_output,
            context_mask=context_mask,
            target_mask=target_mask,
            probes=probes,
        )

    def update_teacher(self, step: int, total_steps: int) -> None:
        current_momentum = get_momentum(step, total_steps, self.config.momentum, self.config.scheduled)
        update_teacher(self.student, self.teacher, current_momentum)

    def assert_student_params_have_grad(self, step: int | None = None) -> None:
        assert_all_trainable_params_have_grad(self.student, step)

    def assert_predictor_params_have_grad(self, step: int | None = None) -> None:
        assert_all_trainable_params_have_grad(self.predictor, step)

    def assert_dense_pred_params_have_grad(self, step: int | None = None) -> None:
        assert_all_trainable_params_have_grad(self.dense_pred_head, step)

    def assert_student_params_synced(self, atol: float = 1e-4, rtol: float = 0) -> None:
        assert_all_ranks_synced(self.student, atol, rtol)

    def assert_predictor_params_synced(self, atol: float = 1e-4, rtol: float = 0) -> None:
        assert_all_ranks_synced(self.predictor, atol, rtol)

    def assert_dense_pred_params_synced(self, atol: float = 1e-4, rtol: float = 0) -> None:
        assert_all_ranks_synced(self.dense_pred_head, atol, rtol)

    if TYPE_CHECKING:

        def __call__(self, x: Tensor, jepa_scale: int, epoch: int) -> MJEPAPredictions:
            return self.forward(x, jepa_scale, epoch)
