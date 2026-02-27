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
    compute_gram_loss,
    compute_sigreg_loss,
    generate_masks,
    get_momentum,
    setup_teacher,
    update_teacher,
)
from .trainer import assert_all_ranks_synced, assert_all_trainable_params_have_grad


@dataclass
class MJEPALosses:
    jepa_loss: Tensor
    jepa_loss_cls: Tensor | float
    sigreg_loss: Tensor | float
    gram_loss: Tensor | float

    gram_loss_weight: float = 1.0
    sigreg_loss_weight: float = 1e-4

    def reduce(self) -> Tensor:
        loss = (
            self.jepa_loss
            + self.jepa_loss_cls
            + self.gram_loss * self.gram_loss_weight
            + self.sigreg_loss * self.sigreg_loss_weight
        )
        assert isinstance(loss, Tensor)
        return loss


@dataclass
class MJEPAPredictions:
    pred: Tensor
    cls_global_pred: Tensor | None
    student_output: ViTFeatures
    teacher_output: ViTFeatures
    context_mask: Tensor
    target_mask: Tensor

    gram_target_output: Tensor | None = None
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
        num_extra_context_tokens: int = 0,
    ) -> Tensor:
        with torch.autocast(device_type=context.device.type, dtype=self.dtype):
            return self.predictor(
                tokenized_size,
                context,
                context_mask,
                target_mask,
                rope_seed=rope_seed,
                num_extra_context_tokens=num_extra_context_tokens,
            )

    def forward_predictor_cls_global(self, cls_tokens: Tensor) -> Tensor:
        with torch.autocast(device_type=cls_tokens.device.type, dtype=self.dtype):
            return self.predictor.forward_cls_global(cls_tokens)

    def forward_probe(self, features: ViTFeatures) -> dict[str, Tensor]:
        return dict()

    def forward_gram_target(self, x: Tensor, context_mask: Tensor) -> Tensor:
        self.teacher.eval()
        with torch.autocast(device_type=x.device.type, dtype=self.dtype), torch.inference_mode():
            stem_tokens = self.teacher.stem(x)
            gram_target_output = apply_mask(context_mask, stem_tokens, fill_value=None)
        return gram_target_output.detach().clone()

    def gram_loss_weight_for_epoch(self, epoch: int) -> float:
        warmup_epochs = self.config.gram_warmup_epochs
        if warmup_epochs <= 0:
            return self.config.gram_loss_weight

        return self.config.gram_loss_weight * min(max(epoch, 0), warmup_epochs) / warmup_epochs

    def compute_losses(self, output: MJEPAPredictions, step: int, epoch: int) -> MJEPALosses:
        # Compute JEPA loss
        target = apply_mask(output.target_mask, output.teacher_output.visual_tokens, fill_value=None).float()
        jepa_loss = F.mse_loss(output.pred.float(), target)
        cls_target = output.teacher_output.visual_tokens.float().mean(dim=1)
        jepa_loss_cls = (
            F.mse_loss(output.cls_global_pred.float(), cls_target) if output.cls_global_pred is not None else 0.0
        )

        # Compute SigREG loss
        sigreg_loss = (
            compute_sigreg_loss(output.student_output.cls_tokens.transpose(0, 1).float(), step, num_slices=256)
            if self.config.sigreg_loss_weight > 0
            else 0.0
        )

        # Compute Gram loss.
        gram_target_output = output.gram_target_output
        assert gram_target_output is not None
        gram_loss = compute_gram_loss(
            output.student_output.visual_tokens.float(),
            gram_target_output.float(),
            remove_neg=self.config.gram_remove_neg,
        )
        gram_loss_weight = self.gram_loss_weight_for_epoch(epoch)

        return MJEPALosses(
            jepa_loss=jepa_loss,
            jepa_loss_cls=jepa_loss_cls,
            sigreg_loss=sigreg_loss,
            gram_loss=gram_loss,
            gram_loss_weight=gram_loss_weight,
            sigreg_loss_weight=self.config.sigreg_loss_weight,
        )

    def forward(self, x: Tensor, jepa_scale: int, epoch: int) -> MJEPAPredictions:
        # NOTE: For DDP to work, all components must execute in the forward pass when training
        context_mask, target_mask = generate_masks(
            self.student, x, self.config.context_ratio, self.config.target_ratio, jepa_scale
        )

        rope_seed = int(torch.randint(0, 1000000, (1,)).item())
        # Teacher / Gram target forward pass
        teacher_output = self.forward_teacher(x)
        gram_target_output = self.forward_gram_target(x, context_mask)

        Ht, Wt = cast(tuple[int, int], self.student.stem.tokenized_size(x.shape[-2:]))

        student_output = self.forward_student(x, context_mask, rope_seed=rope_seed)
        cls_tokens = student_output.cls_tokens
        num_cls_tokens = cls_tokens.shape[1]
        context_tokens = (
            torch.cat([student_output.visual_tokens, cls_tokens], dim=1)
            if num_cls_tokens
            else student_output.visual_tokens
        )
        pred = self.forward_predictor(
            (Ht, Wt),
            context_tokens,
            context_mask,
            target_mask,
            rope_seed=rope_seed,
            num_extra_context_tokens=num_cls_tokens,
        )
        cls_global_pred = self.forward_predictor_cls_global(cls_tokens) if cls_tokens.numel() else None

        with torch.autocast(device_type=pred.device.type, dtype=self.dtype):
            probes = self.forward_probe(teacher_output)

        return MJEPAPredictions(
            pred=pred,
            cls_global_pred=cls_global_pred,
            student_output=student_output,
            teacher_output=teacher_output,
            context_mask=context_mask,
            target_mask=target_mask,
            gram_target_output=gram_target_output,
            probes=probes,
        )

    def update_teacher(self, step: int, total_steps: int) -> None:
        current_momentum = get_momentum(step, total_steps, self.config.momentum, self.config.scheduled)
        update_teacher(self.student, self.teacher, current_momentum)

    def assert_student_params_have_grad(self, step: int | None = None) -> None:
        assert_all_trainable_params_have_grad(self.student, step)

    def assert_predictor_params_have_grad(self, step: int | None = None) -> None:
        assert_all_trainable_params_have_grad(self.predictor, step)

    def assert_student_params_synced(self, atol: float = 1e-4, rtol: float = 0) -> None:
        assert_all_ranks_synced(self.student, atol, rtol)

    def assert_predictor_params_synced(self, atol: float = 1e-4, rtol: float = 0) -> None:
        assert_all_ranks_synced(self.predictor, atol, rtol)

    if TYPE_CHECKING:

        def __call__(self, x: Tensor, jepa_scale: int, epoch: int) -> MJEPAPredictions:
            return self.forward(x, jepa_scale, epoch)
