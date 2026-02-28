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
    forward_gram_teacher,
    generate_masks,
    get_momentum,
    is_gram_update_epoch,
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
    pred_with_cls: Tensor | None
    student_output: ViTFeatures
    teacher_output: ViTFeatures
    context_mask: Tensor
    target_mask: Tensor

    gram_teacher_output: Tensor | None = None
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
        self.gram_teacher = setup_teacher(backbone) if config.gram_start_epoch is not None else None
        self.dtype = dtype
        self._gram_cooldown_end_epoch: int | None = None

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

    def forward_gram_teacher(self, x: Tensor, context_mask: Tensor, rope_seed: int | None = None) -> Tensor:
        if self.gram_teacher is None:
            raise ValueError("Gram teacher is not initialized")

        self.gram_teacher.eval()
        with torch.autocast(device_type=x.device.type, dtype=self.dtype), torch.inference_mode():
            gram_teacher_output = forward_gram_teacher(
                self.gram_teacher,
                x,
                rope_seed=rope_seed,
                resolution_scale=self.config.gram_resolution_scale,
            )
            gram_teacher_output = apply_mask(context_mask, gram_teacher_output, fill_value=None)
        return gram_teacher_output.clone()

    def _is_gram_cooldown_active(self, epoch: int) -> bool:
        if self._gram_cooldown_end_epoch is None:
            return False
        return epoch < self._gram_cooldown_end_epoch

    def _is_gram_enabled_for_epoch(self, epoch: int) -> bool:
        if self.config.gram_start_epoch is None:
            return False
        return epoch >= self.config.gram_start_epoch and not self._is_gram_cooldown_active(epoch)

    def update_gram_teacher(self, current_epoch: int, resolution_changed: bool = False):
        if self.gram_teacher is None:
            return

        should_sync_gram_teacher = current_epoch == self.config.gram_teacher_epoch

        # Initial Gram teacher setup (if necessary)
        # Pause gram anchoring updates after a resolution transition.
        interval = self.config.gram_update_interval_epoch
        if resolution_changed and interval > 0:
            self._gram_cooldown_end_epoch = current_epoch + interval

        if self._gram_cooldown_end_epoch is not None:
            if current_epoch < self._gram_cooldown_end_epoch:
                return
            should_sync_gram_teacher = should_sync_gram_teacher or current_epoch >= self.config.gram_teacher_epoch
            self._gram_cooldown_end_epoch = None

        # Gram teacher update
        if is_gram_update_epoch(current_epoch, self.config.gram_start_epoch, self.config.gram_update_interval_epoch):
            should_sync_gram_teacher = True

        if should_sync_gram_teacher:
            update_teacher(self.teacher, self.gram_teacher)

    def compute_losses(self, output: MJEPAPredictions, step: int, epoch: int) -> MJEPALosses:
        # Compute JEPA loss
        target = apply_mask(output.target_mask, output.teacher_output.visual_tokens, fill_value=None).float()
        jepa_loss = F.mse_loss(output.pred.float(), target)
        jepa_loss_cls = F.mse_loss(output.pred_with_cls.float(), target) if output.pred_with_cls is not None else 0.0

        # Compute SigREG loss
        sigreg_loss = (
            compute_sigreg_loss(output.student_output.cls_tokens.transpose(0, 1).float(), step, num_slices=256)
            if self.config.sigreg_loss_weight > 0
            else 0.0
        )

        # Compute Gram loss (if necessary)
        if self._is_gram_enabled_for_epoch(epoch):
            assert output.gram_teacher_output is not None
            assert self.config.gram_loss_weight > 0
            gram_loss = compute_gram_loss(
                output.student_output.visual_tokens.float(),
                output.gram_teacher_output.float(),
                remove_neg=self.config.gram_remove_neg,
            )
        else:
            gram_loss = 0.0

        return MJEPALosses(
            jepa_loss=jepa_loss,
            jepa_loss_cls=jepa_loss_cls,
            sigreg_loss=sigreg_loss,
            gram_loss=gram_loss,
            gram_loss_weight=self.config.gram_loss_weight,
            sigreg_loss_weight=self.config.sigreg_loss_weight,
        )

    def forward(self, x: Tensor, jepa_scale: int, epoch: int) -> MJEPAPredictions:
        # NOTE: For DDP to work, all components must execute in the forward pass when training
        context_mask, target_mask = generate_masks(
            self.student, x, self.config.context_ratio, self.config.target_ratio, jepa_scale
        )

        rope_seed = int(torch.randint(0, 1000000, (1,)).item())
        # Teacher / Gram teacher forward pass
        teacher_output = self.forward_teacher(x)
        gram_teacher_output = (
            self.forward_gram_teacher(x, context_mask, rope_seed) if self._is_gram_enabled_for_epoch(epoch) else None
        )

        Ht, Wt = cast(tuple[int, int], self.student.stem.tokenized_size(x.shape[-2:]))

        student_output = self.forward_student(x, context_mask, rope_seed=rope_seed)
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
            student_output=student_output,
            teacher_output=teacher_output,
            context_mask=context_mask,
            target_mask=target_mask,
            gram_teacher_output=gram_teacher_output,
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
