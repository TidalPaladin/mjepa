from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from torch import Tensor
from vit import ViT, ViTFeatures
from vit.tokens import apply_mask

from .jepa import (
    ADALN_BLIND_CLS_PREDICTION_MODE,
    ADALN_CLS_PREDICTION_MODES,
    ADALN_SHARED_CLS_PREDICTION_MODE,
    DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODES,
    EXPANDED_CLS_PREDICTION_MODES,
    JOINT_CONTEXT_CLS_PREDICTION_MODES,
    PACKED_ADALN_HARD_BLIND_CLS_PREDICTION_MODES,
    SINGLE_CLS_PREDICTION_MODES,
    CrossAttentionPredictor,
    JEPAConfig,
    autocast_context,
    compute_gram_loss,
    compute_jepa_prediction_loss,
    compute_sigreg_loss,
    create_joint_context_attention_mask,
    create_packed_dual_route_mask,
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

    stem_jepa_loss: Tensor | float = 0.0
    stem_jepa_loss_weight: float = 0.0
    jepa_loss_weight: float = 1.0
    gram_loss_weight: float = 1.0
    sigreg_loss_weight: float = 1e-4

    def reduce(self) -> Tensor:
        loss = (
            self.jepa_loss * self.jepa_loss_weight
            + self.jepa_loss_cls
            + self.stem_jepa_loss * self.stem_jepa_loss_weight
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

    stem_pred: Tensor | None = None
    teacher_stem_output: Tensor | None = None
    gram_teacher_output: Tensor | None = None
    probes: dict[str, Tensor] = field(default_factory=dict)
    context_attention_mask: Tensor | None = None


class MJEPA(nn.Module):
    def __init__(
        self,
        config: JEPAConfig,
        backbone: ViT,
        predictor: CrossAttentionPredictor,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.config = config
        self.student = backbone
        teacher_dtype = config.teacher_dtype
        if predictor.attention_mode != config.predictor_attention_mode:
            raise ValueError(
                "Predictor attention mode does not match JEPAConfig: "
                f"{predictor.attention_mode!r} != {config.predictor_attention_mode!r}"
            )
        if predictor.cls_prediction_mode != config.cls_prediction_mode:
            raise ValueError(
                "Predictor CLS prediction mode does not match JEPAConfig: "
                f"{predictor.cls_prediction_mode!r} != {config.cls_prediction_mode!r}"
            )
        if predictor.cls_context_tokens != config.cls_context_tokens:
            raise ValueError(
                "Predictor CLS context count does not match JEPAConfig: "
                f"{predictor.cls_context_tokens} != {config.cls_context_tokens}"
            )
        if config.cls_prediction_mode in SINGLE_CLS_PREDICTION_MODES and backbone.config.num_cls_tokens != 1:
            raise ValueError(
                f"{config.cls_prediction_mode} CLS prediction requires exactly one CLS token, "
                f"got {backbone.config.num_cls_tokens}"
            )
        self.teacher = setup_teacher(backbone, dtype=teacher_dtype)
        self.predictor = predictor
        if self._use_stem_targets():
            self.predictor.enable_shallow_head()
        self.gram_teacher = setup_teacher(backbone, dtype=teacher_dtype) if self.config.use_gram_anchoring else None
        self.autocast_dtype = autocast_dtype
        self._gram_cooldown_end_epoch: int | None = None

    @property
    def img_size(self) -> tuple[int, int]:
        return cast(tuple[int, int], self.student.config.img_size)

    def forward_teacher(self, x: Tensor, mask: Tensor | None = None) -> ViTFeatures:
        self.teacher.eval()
        with autocast_context(x.device.type, self.autocast_dtype), torch.inference_mode():
            output = self.teacher(x, mask=mask)
        return ViTFeatures(
            output.dense_features.clone(),
            output.num_register_tokens,
            output.num_cls_tokens,
            output.tokenized_size,
            output.visual_mask.clone() if output.visual_mask is not None else None,
        )

    def forward_teacher_stem(self, x: Tensor) -> Tensor:
        self.teacher.eval()
        with autocast_context(x.device.type, self.autocast_dtype), torch.inference_mode():
            stem_output = self.teacher.stem(x)
            stem_output = self.teacher.normalize_patch_embeddings(stem_output)
        return stem_output.clone()

    def forward_student(self, x: Tensor, context_mask: Tensor, rope_seed: int | None = None) -> ViTFeatures:
        with autocast_context(x.device.type, self.autocast_dtype):
            return self.student(x, mask=context_mask, rope_seed=rope_seed)

    def forward_predictor(
        self,
        tokenized_size: tuple[int, int],
        context: Tensor,
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
        conditioning: Tensor | None = None,
        context_attention_mask: Tensor | None = None,
    ) -> Tensor:
        with autocast_context(context.device.type, self.autocast_dtype):
            return self.predictor(
                tokenized_size,
                context,
                context_mask,
                target_mask,
                rope_seed=rope_seed,
                conditioning=conditioning,
                context_attention_mask=context_attention_mask,
            )

    def forward_predictor_heads(
        self,
        tokenized_size: tuple[int, int],
        context: Tensor,
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
        conditioning: Tensor | None = None,
        context_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        with autocast_context(context.device.type, self.autocast_dtype):
            return self.predictor.forward_heads(
                tokenized_size,
                context,
                context_mask,
                target_mask,
                rope_seed=rope_seed,
                conditioning=conditioning,
                context_attention_mask=context_attention_mask,
            )

    def forward_blind_cls_predictor(
        self,
        tokenized_size: tuple[int, int],
        conditioning: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        with autocast_context(conditioning.device.type, self.autocast_dtype):
            return self.predictor.forward_blind(tokenized_size, conditioning, target_mask)

    def forward_cls_predictor(
        self,
        tokenized_size: tuple[int, int],
        cls_tokens: Tensor,
        target_mask: Tensor,
        rope_seed: int | None = None,
    ) -> Tensor:
        """Run the configured visually blinded auxiliary predictor from CLS tokens."""
        if self.config.cls_prediction_mode in EXPANDED_CLS_PREDICTION_MODES:
            cls_tokens = self.predictor.expand_cls_context(cls_tokens)
        elif self.config.cls_prediction_mode in ADALN_CLS_PREDICTION_MODES:
            return self.forward_blind_cls_predictor(tokenized_size, cls_tokens[:, 0], target_mask)
        return self.forward_predictor(tokenized_size, cls_tokens, None, target_mask, rope_seed=rope_seed)

    def forward_joint_context_predictor_heads(
        self,
        tokenized_size: tuple[int, int],
        visual_context: Tensor,
        cls_tokens: Tensor,
        context_mask: Tensor,
        target_mask: Tensor,
        rope_seed: int | None = None,
        context_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Predict from visible visual tokens and one appended CLS token in a single forward pass."""
        if self.config.cls_prediction_mode not in JOINT_CONTEXT_CLS_PREDICTION_MODES:
            raise ValueError("joint context prediction requires a joint-context CLS prediction mode")
        if cls_tokens.ndim != 3 or cls_tokens.shape[1] != 1:
            raise ValueError("joint context prediction requires exactly one CLS token")

        joint_context = torch.cat([visual_context, cls_tokens], dim=1)
        return self.forward_predictor_heads(
            tokenized_size,
            joint_context,
            context_mask,
            target_mask,
            rope_seed=rope_seed,
            context_attention_mask=context_attention_mask,
        )

    def forward_packed_adaln_hard_blind_predictor_heads(
        self,
        tokenized_size: tuple[int, int],
        visual_context: Tensor,
        cls_tokens: Tensor,
        context_mask: Tensor,
        target_mask: Tensor,
        rope_seed: int | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Predict visual and hard-blind target copies in one configured predictor forward."""
        if self.config.cls_prediction_mode not in PACKED_ADALN_HARD_BLIND_CLS_PREDICTION_MODES:
            raise ValueError("packed AdaLN prediction requires a packed hard-blind CLS prediction mode")
        if cls_tokens.ndim != 3 or cls_tokens.shape[1] != 1:
            raise ValueError("packed AdaLN prediction requires exactly one CLS token")
        return self.forward_predictor_heads(
            tokenized_size,
            visual_context,
            context_mask,
            target_mask,
            rope_seed=rope_seed,
            conditioning=cls_tokens[:, 0],
        )

    @staticmethod
    def _has_cls_tokens(features: ViTFeatures) -> bool:
        return features.num_cls_tokens > 0

    @classmethod
    def _get_probe_tokens(cls, features: ViTFeatures) -> Tensor:
        return features.cls_tokens if cls._has_cls_tokens(features) else features.visual_tokens

    @staticmethod
    def _masked_target(target_mask: Tensor, target_features: Tensor) -> Tensor:
        return apply_mask(target_mask, target_features, fill_value=None)

    def _use_stem_targets(self) -> bool:
        return self.config.stem_jepa_loss_weight > 0

    def forward_probe(self, features: ViTFeatures) -> dict[str, Tensor]:
        """Override in subclasses; use ``_get_probe_tokens`` for zero-CLS-safe inputs."""
        return dict()

    def forward_gram_teacher(self, x: Tensor, context_mask: Tensor, rope_seed: int | None = None) -> Tensor:
        if self.gram_teacher is None:
            raise ValueError("Gram teacher is not initialized")

        self.gram_teacher.eval()
        with autocast_context(x.device.type, self.autocast_dtype), torch.inference_mode():
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
        if not self.config.use_gram_anchoring or self.config.gram_start_epoch is None:
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
        student_output = output.student_output
        target = self._masked_target(output.target_mask, output.teacher_output.restored_visual_tokens)
        if self.config.cls_prediction_mode in DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODES:
            jepa_loss, jepa_loss_cls = self._compute_dual_route_losses(
                output.pred,
                target,
                output.context_attention_mask,
            )
        else:
            jepa_loss = compute_jepa_prediction_loss(output.pred, target, kind=self.config.jepa_loss_kind)
            jepa_loss_cls = (
                compute_jepa_prediction_loss(output.pred_with_cls, target, kind=self.config.jepa_loss_kind)
                if output.pred_with_cls is not None
                else 0.0
            )
        if output.stem_pred is not None and output.teacher_stem_output is not None:
            stem_target = self._masked_target(output.target_mask, output.teacher_stem_output)
            if self.config.cls_prediction_mode in DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODES:
                stem_visual_loss, stem_cls_loss = self._compute_dual_route_losses(
                    output.stem_pred,
                    stem_target,
                    output.context_attention_mask,
                )
                stem_jepa_loss = stem_visual_loss + stem_cls_loss
            else:
                stem_jepa_loss = compute_jepa_prediction_loss(
                    output.stem_pred,
                    stem_target,
                    kind=self.config.jepa_loss_kind,
                )
        else:
            stem_jepa_loss = 0.0

        sigreg_loss = (
            compute_sigreg_loss(student_output.cls_tokens.transpose(0, 1).float(), step, num_slices=256)
            if self.config.sigreg_loss_weight > 0 and self._has_cls_tokens(student_output)
            else 0.0
        )

        if self._is_gram_enabled_for_epoch(epoch):
            assert output.gram_teacher_output is not None
            assert self.config.gram_loss_weight > 0
            gram_loss = compute_gram_loss(
                student_output.visual_tokens.float(),
                output.gram_teacher_output.float(),
                remove_neg=self.config.gram_remove_neg,
            )
        else:
            gram_loss = 0.0

        return MJEPALosses(
            jepa_loss=jepa_loss,
            jepa_loss_cls=jepa_loss_cls,
            stem_jepa_loss=stem_jepa_loss,
            sigreg_loss=sigreg_loss,
            gram_loss=gram_loss,
            stem_jepa_loss_weight=self.config.stem_jepa_loss_weight,
            jepa_loss_weight=self.config.jepa_loss_weight,
            gram_loss_weight=self.config.gram_loss_weight,
            sigreg_loss_weight=self.config.sigreg_loss_weight,
        )

    def _compute_dual_route_losses(
        self,
        prediction: Tensor,
        target: Tensor,
        context_attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if context_attention_mask is None or context_attention_mask.dtype != torch.bool:
            raise ValueError("dual-routed prediction requires its boolean context attention mask")
        if prediction.shape[1] % target.shape[1] != 0:
            raise ValueError("dual-routed prediction count must be a multiple of the teacher target count")
        query_multiplier = prediction.shape[1] // target.shape[1]
        aligned_target = target.repeat(1, query_multiplier, 1)
        route_mask = context_attention_mask[:, 0]
        if route_mask.shape[:2] != prediction.shape[:2]:
            raise ValueError("dual-routed context attention mask must align with predictor queries")
        visual_only = route_mask[..., :-1].all(dim=-1) & ~route_mask[..., -1]
        cls_only = ~route_mask[..., :-1].any(dim=-1) & route_mask[..., -1]
        if not visual_only.any() or not cls_only.any() or (visual_only & cls_only).any():
            raise ValueError("dual-routed prediction requires non-overlapping visual-only and CLS-only queries")
        return (
            compute_jepa_prediction_loss(
                prediction[visual_only],
                aligned_target[visual_only],
                kind=self.config.jepa_loss_kind,
            ),
            compute_jepa_prediction_loss(
                prediction[cls_only],
                aligned_target[cls_only],
                kind=self.config.jepa_loss_kind,
            ),
        )

    def forward(
        self,
        x: Tensor,
        jepa_scale: int,
        epoch: int,
        compute_mask: Tensor | None = None,
    ) -> MJEPAPredictions:
        # NOTE: For DDP to work, all components must execute in the forward pass when training
        context_mask, target_mask = generate_masks(
            self.student,
            x,
            self.config.context_ratio,
            self.config.target_ratio,
            jepa_scale,
            eligible_mask=compute_mask,
        )

        rope_seed = int(torch.randint(0, 1000000, (1,)).item())
        # Teacher / Gram teacher forward pass
        teacher_output = self.forward_teacher(x, mask=compute_mask)
        teacher_stem_output = self.forward_teacher_stem(x) if self._use_stem_targets() else None
        gram_teacher_output = (
            self.forward_gram_teacher(x, context_mask, rope_seed) if self._is_gram_enabled_for_epoch(epoch) else None
        )

        Ht, Wt = cast(tuple[int, int], self.student.stem.tokenized_size(x.shape[-2:]))

        student_output = self.forward_student(x, context_mask, rope_seed=rope_seed)
        student_has_cls_tokens = self._has_cls_tokens(student_output)
        cls_conditioning = student_output.cls_tokens[:, 0] if student_has_cls_tokens else None
        main_conditioning = None
        if self.config.cls_prediction_mode == ADALN_BLIND_CLS_PREDICTION_MODE:
            assert cls_conditioning is not None
            main_conditioning = torch.zeros_like(cls_conditioning)
        elif self.config.cls_prediction_mode == ADALN_SHARED_CLS_PREDICTION_MODE:
            assert cls_conditioning is not None
            main_conditioning = cls_conditioning
        context_attention_mask = None
        if self.config.cls_prediction_mode in PACKED_ADALN_HARD_BLIND_CLS_PREDICTION_MODES:
            target_tokens_per_example = target_mask.sum(dim=1)
            if not torch.equal(
                target_tokens_per_example,
                target_tokens_per_example[:1].expand_as(target_tokens_per_example),
            ):
                raise ValueError("packed AdaLN prediction requires the same target-token count in every example")
            context_attention_mask = create_packed_dual_route_mask(
                target_mask.shape[0],
                int(target_tokens_per_example[0].item()),
                student_output.visual_tokens.shape[1],
                device=student_output.visual_tokens.device,
            )
            pred, stem_pred = self.forward_packed_adaln_hard_blind_predictor_heads(
                (Ht, Wt),
                student_output.visual_tokens,
                student_output.cls_tokens,
                context_mask,
                target_mask,
                rope_seed=rope_seed,
            )
            pred_with_cls = None
        elif self.config.cls_prediction_mode in JOINT_CONTEXT_CLS_PREDICTION_MODES:
            target_tokens_per_example = target_mask.sum(dim=1)
            if not torch.equal(
                target_tokens_per_example,
                target_tokens_per_example[:1].expand_as(target_tokens_per_example),
            ):
                raise ValueError("joint-context prediction requires the same target-token count in every example")
            context_attention_mask = create_joint_context_attention_mask(
                self.config.cls_prediction_mode,
                target_mask.shape[0],
                int(target_tokens_per_example[0].item()),
                student_output.visual_tokens.shape[1],
                device=student_output.visual_tokens.device,
                dtype=student_output.visual_tokens.dtype,
            )
            pred, stem_pred = self.forward_joint_context_predictor_heads(
                (Ht, Wt),
                student_output.visual_tokens,
                student_output.cls_tokens,
                context_mask,
                target_mask,
                rope_seed=rope_seed,
                context_attention_mask=context_attention_mask,
            )
            pred_with_cls = None
        else:
            pred, stem_pred = self.forward_predictor_heads(
                (Ht, Wt),
                student_output.visual_tokens,
                context_mask,
                target_mask,
                rope_seed=rope_seed,
                conditioning=main_conditioning,
            )
            pred_with_cls = (
                self.forward_cls_predictor((Ht, Wt), student_output.cls_tokens, target_mask, rope_seed=rope_seed)
                if student_has_cls_tokens
                else None
            )

        with autocast_context(pred.device.type, self.autocast_dtype):
            probes = self.forward_probe(teacher_output)

        return MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            stem_pred=stem_pred,
            student_output=student_output,
            teacher_output=teacher_output,
            context_mask=context_mask,
            target_mask=target_mask,
            teacher_stem_output=teacher_stem_output,
            gram_teacher_output=gram_teacher_output,
            probes=probes,
            context_attention_mask=context_attention_mask,
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

        def __call__(
            self,
            x: Tensor,
            jepa_scale: int,
            epoch: int,
            compute_mask: Tensor | None = None,
        ) -> MJEPAPredictions:
            return self.forward(x, jepa_scale, epoch, compute_mask)
