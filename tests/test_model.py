import pytest
import torch
from vit import ViT, ViTConfig, ViTFeatures
from vit.tokens import apply_mask

from mjepa.jepa import CrossAttentionPredictor, JEPAConfig
from mjepa.model import MJEPA, MJEPALosses, MJEPAPredictions


BATCH_SIZE = 2
IN_CHANNELS = 3
IMG_SIZE = 32
PATCH_SIZE = [4, 4]
HIDDEN_SIZE = 64
DEPTH = 2
NUM_ATTENTION_HEADS = 4
FFN_HIDDEN_SIZE = 128
NUM_VISUAL_TOKENS = 64
NUM_REGISTER_TOKENS = 2
NUM_CLS_TOKENS = 2


def _create_small_vit_config(
    *,
    dtype: torch.dtype = torch.float32,
    num_register_tokens: int = NUM_REGISTER_TOKENS,
    num_cls_tokens: int = NUM_CLS_TOKENS,
) -> ViTConfig:
    return ViTConfig(
        in_channels=IN_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        patch_size=PATCH_SIZE,
        img_size=[IMG_SIZE, IMG_SIZE],
        depth=DEPTH,
        num_attention_heads=NUM_ATTENTION_HEADS,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        activation="gelu",
        num_register_tokens=num_register_tokens,
        num_cls_tokens=num_cls_tokens,
        pos_enc="rope",
        rope_base=100,
        dtype=dtype,
    )


@pytest.fixture
def jepa_config():
    """Create a JEPA configuration for testing."""
    return JEPAConfig(
        context_ratio=0.5,
        target_ratio=0.25,
        scale=2,
        momentum=0.98,
        predictor_depth=2,
        scheduled=False,
        dense_pred_loss_weight=1.0,
        sigreg_loss_weight=0.0001,
    )


@pytest.fixture
def jepa_config_no_sigreg():
    """Create a JEPA configuration with SigREG disabled."""
    return JEPAConfig(
        context_ratio=0.5,
        target_ratio=0.25,
        scale=2,
        momentum=0.98,
        predictor_depth=2,
        scheduled=False,
        dense_pred_loss_weight=1.0,
        sigreg_loss_weight=0.0,
    )


@pytest.fixture
def small_vit_config():
    """Create a small ViT configuration for testing."""
    return _create_small_vit_config(dtype=torch.float32)


@pytest.fixture
def small_vit(small_vit_config):
    """Create a small ViT backbone for testing."""
    return small_vit_config.instantiate()


@pytest.fixture
def predictor(small_vit):
    """Create a predictor for testing."""
    return CrossAttentionPredictor(small_vit, depth=2, out_dim=None)


@pytest.fixture
def mjepa_model(jepa_config, small_vit, predictor):
    """Create an MJEPA model for testing."""
    return MJEPA(jepa_config, small_vit, predictor, dtype=torch.float32)


@pytest.fixture
def mjepa_model_no_sigreg(jepa_config_no_sigreg, small_vit, predictor):
    """Create an MJEPA model with SigREG disabled."""
    return MJEPA(jepa_config_no_sigreg, small_vit, predictor, dtype=torch.float32)


@pytest.fixture
def dummy_batch():
    """Create a dummy batch of images."""
    return torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE)


@pytest.fixture
def dummy_vit_features():
    """Create dummy ViT features for testing."""
    dense_features = torch.randn(
        BATCH_SIZE,
        NUM_VISUAL_TOKENS + NUM_CLS_TOKENS + NUM_REGISTER_TOKENS,
        HIDDEN_SIZE,
    )
    return ViTFeatures(dense_features, NUM_REGISTER_TOKENS, NUM_CLS_TOKENS)


class TestMJEPALosses:
    """Test the MJEPALosses dataclass."""

    def test_initialization(self):
        """Test that MJEPALosses can be initialized."""
        jepa_loss = torch.tensor(1.0)
        jepa_loss_cls = torch.tensor(0.5)
        sigreg_loss = torch.tensor(0.01)
        dense_pred_loss = torch.tensor(0.1)

        losses = MJEPALosses(
            jepa_loss=jepa_loss,
            jepa_loss_cls=jepa_loss_cls,
            sigreg_loss=sigreg_loss,
            dense_pred_loss=dense_pred_loss,
            dense_pred_loss_weight=1.0,
            sigreg_loss_weight=1e-4,
        )

        assert losses.jepa_loss == jepa_loss
        assert losses.jepa_loss_cls == jepa_loss_cls
        assert losses.sigreg_loss == sigreg_loss
        assert losses.dense_pred_loss == dense_pred_loss
        assert losses.dense_pred_loss_weight == 1.0
        assert losses.sigreg_loss_weight == 1e-4

    @pytest.mark.parametrize(
        "dense_pred_weight,sigreg_weight",
        [
            (1.0, 1e-4),
            (0.5, 1e-3),
            (2.0, 0.0),
        ],
    )
    def test_reduce(self, dense_pred_weight, sigreg_weight):
        """Test that reduce combines losses correctly."""
        jepa_loss = torch.tensor(1.0)
        jepa_loss_cls = torch.tensor(0.5)
        sigreg_loss = torch.tensor(0.01)
        dense_pred_loss = torch.tensor(0.1)

        losses = MJEPALosses(
            jepa_loss=jepa_loss,
            jepa_loss_cls=jepa_loss_cls,
            sigreg_loss=sigreg_loss,
            dense_pred_loss=dense_pred_loss,
            dense_pred_loss_weight=dense_pred_weight,
            sigreg_loss_weight=sigreg_weight,
        )

        reduced = losses.reduce()
        expected = jepa_loss + jepa_loss_cls + dense_pred_loss * dense_pred_weight + sigreg_loss * sigreg_weight

        assert isinstance(reduced, torch.Tensor)
        assert torch.allclose(reduced, expected)


class TestMJEPAPredictions:
    """Test the MJEPAPredictions dataclass."""

    def test_initialization(self, dummy_vit_features):
        """Test that MJEPAPredictions can be initialized."""
        pred = torch.randn(2, 16, 64)
        pred_with_cls = torch.randn(2, 16, 64)
        dense_pred = torch.randn(2, 32, 64)
        teacher_stem_output = torch.randn(2, 32, 64)
        context_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)
        target_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)

        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            dense_pred=dense_pred,
            student_output=dummy_vit_features,
            teacher_output=dummy_vit_features,
            teacher_stem_output=teacher_stem_output,
            context_mask=context_mask,
            target_mask=target_mask,
        )

        assert torch.allclose(predictions.pred, pred)
        assert predictions.pred_with_cls is not None
        assert torch.allclose(predictions.pred_with_cls, pred_with_cls)
        assert torch.allclose(predictions.dense_pred, dense_pred)
        assert torch.allclose(predictions.teacher_stem_output, teacher_stem_output)
        assert predictions.student_output == dummy_vit_features
        assert predictions.teacher_output == dummy_vit_features
        assert torch.all(predictions.context_mask == context_mask)
        assert torch.all(predictions.target_mask == target_mask)
        assert predictions.probes == {}


class TestMJEPAInitialization:
    """Test MJEPA model initialization."""

    def test_initialization(
        self,
        mjepa_model: MJEPA,
        jepa_config: JEPAConfig,
        small_vit: ViT,
        predictor: CrossAttentionPredictor,
    ):
        """Test that MJEPA model is initialized correctly."""
        assert mjepa_model.config == jepa_config
        assert mjepa_model.student == small_vit
        assert mjepa_model.predictor == predictor
        assert mjepa_model.dtype == torch.float32
        assert mjepa_model.teacher is not None
        assert mjepa_model.dense_pred_head is not None

    def test_teacher_frozen(self, mjepa_model):
        """Test that teacher parameters are frozen."""
        for param in mjepa_model.teacher.parameters():
            assert not param.requires_grad

    def test_dense_pred_head_trainable(self, mjepa_model):
        """Test that dense prediction head parameters are trainable."""
        for param in mjepa_model.dense_pred_head.parameters():
            assert param.requires_grad

    def test_img_size_property(self, mjepa_model):
        """Test the img_size property."""
        img_size = mjepa_model.img_size
        assert img_size == (32, 32) or img_size == [32, 32]
        assert len(img_size) == 2
        assert img_size[0] == 32
        assert img_size[1] == 32


class TestMJEPAForwardPasses:
    """Test MJEPA forward pass helper methods."""

    def test_forward_teacher(self, mjepa_model: MJEPA, dummy_batch):
        """Test forward pass through teacher."""
        output = mjepa_model.forward_teacher(dummy_batch)

        assert isinstance(output, ViTFeatures)
        assert output.dense_features.shape[0] == dummy_batch.shape[0]
        assert output.dense_features.dtype == torch.float32

    def test_forward_teacher_eval_mode(self, mjepa_model: MJEPA, dummy_batch):
        """Test that teacher is in eval mode during forward pass."""
        mjepa_model.teacher.train()
        _ = mjepa_model.forward_teacher(dummy_batch)
        assert not mjepa_model.teacher.training

    def test_forward_student(self, mjepa_model: MJEPA, dummy_batch):
        """Test forward pass through student."""
        context_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)
        output = mjepa_model.forward_student(dummy_batch, context_mask)

        assert isinstance(output, ViTFeatures)
        assert output.dense_features.shape[0] == dummy_batch.shape[0]

    def test_forward_predictor(self, mjepa_model: MJEPA):
        """Test forward pass through predictor."""
        tokenized_size = (8, 8)
        num_tokens = tokenized_size[0] * tokenized_size[1]
        context = torch.randn(2, 32, 64)
        context_mask = torch.zeros(2, num_tokens, dtype=torch.bool)
        context_mask[:, :32] = True
        target_mask = torch.zeros(2, num_tokens, dtype=torch.bool)
        target_mask[:, 32:48] = True

        output = mjepa_model.forward_predictor(tokenized_size, context, context_mask, target_mask)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2

    def test_forward_probe(self, mjepa_model: MJEPA, dummy_vit_features):
        """Test forward pass through probe (currently returns empty dict)."""
        output = mjepa_model.forward_probe(dummy_vit_features)

        assert isinstance(output, dict)
        assert len(output) == 0

    def test_forward_teacher_stem_masks_context(self, mjepa_model: MJEPA, dummy_batch):
        """Teacher stem output should be masked to context tokens."""
        context_mask = torch.zeros(2, 64, dtype=torch.bool)
        context_mask[:, :24] = True

        output = mjepa_model.forward_teacher_stem(dummy_batch, context_mask)

        assert output.shape == (2, 24, 64)
        expected = apply_mask(context_mask, mjepa_model.teacher.stem(dummy_batch), fill_value=None)
        assert torch.allclose(output, expected)

    def test_forward_dense_pred(self, mjepa_model: MJEPA):
        """Dense prediction head should preserve token count and hidden dimension."""
        student_visual_tokens = torch.randn(2, 24, 64)
        dense_pred = mjepa_model.forward_dense_pred(student_visual_tokens)

        assert dense_pred.shape == student_visual_tokens.shape


class TestMJEPATeacherUpdate:
    """Test MJEPA teacher update methods."""

    def test_update_teacher(self, mjepa_model: MJEPA):
        """Test updating the teacher model."""
        initial_weight = mjepa_model.teacher.stem.patch.weight.data.clone()

        mjepa_model.student.stem.patch.weight.data.fill_(1.0)

        step = 0
        total_steps = 100
        mjepa_model.update_teacher(step, total_steps)

        assert not torch.allclose(mjepa_model.teacher.stem.patch.weight.data, initial_weight)

    def test_update_teacher_scheduled(self):
        """Test updating teacher with scheduled momentum."""
        config = JEPAConfig(
            momentum=0.9,
            scheduled=True,
            dense_pred_loss_weight=1.0,
            sigreg_loss_weight=0.0,
        )
        vit_config = _create_small_vit_config(dtype=torch.float32)
        backbone = vit_config.instantiate()
        predictor = CrossAttentionPredictor(backbone, depth=2)
        model = MJEPA(config, backbone, predictor)

        initial_weight = model.teacher.stem.patch.weight.data.clone()

        model.student.stem.patch.weight.data.fill_(1.0)

        model.update_teacher(0, 100)
        weight_step_0 = model.teacher.stem.patch.weight.data.clone()

        model.update_teacher(50, 100)
        weight_step_50 = model.teacher.stem.patch.weight.data.clone()

        assert not torch.allclose(weight_step_0, initial_weight)
        assert not torch.allclose(weight_step_50, weight_step_0)


class TestMJEPAComputeLosses:
    """Test MJEPA loss computation."""

    @staticmethod
    def _build_feature_pair(
        batch_size: int,
        num_tokens: int,
        hidden_size: int,
        num_register_tokens: int,
        num_cls_tokens: int,
    ) -> tuple[ViTFeatures, ViTFeatures]:
        dense_features = torch.randn(batch_size, num_tokens + num_register_tokens + num_cls_tokens, hidden_size)
        student_output = ViTFeatures(dense_features, num_register_tokens, num_cls_tokens)
        teacher_output = ViTFeatures(dense_features.clone(), num_register_tokens, num_cls_tokens)
        return student_output, teacher_output

    @staticmethod
    def _build_target_mask(batch_size: int, num_tokens: int, num_target_tokens: int) -> torch.Tensor:
        target_mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool)
        target_mask[:, :num_target_tokens] = True
        return target_mask

    def test_compute_losses_basic(self, mjepa_model_no_sigreg: MJEPA):
        """Test basic loss computation with dense prediction loss."""
        pred = torch.randn(2, 16, 64)
        pred_with_cls = torch.randn(2, 16, 64)

        student_output, teacher_output = self._build_feature_pair(
            batch_size=BATCH_SIZE,
            num_tokens=NUM_VISUAL_TOKENS,
            hidden_size=HIDDEN_SIZE,
            num_register_tokens=NUM_REGISTER_TOKENS,
            num_cls_tokens=NUM_CLS_TOKENS,
        )
        target_mask = self._build_target_mask(
            batch_size=BATCH_SIZE,
            num_tokens=NUM_VISUAL_TOKENS,
            num_target_tokens=16,
        )

        teacher_stem_output = torch.randn(2, 32, 64)
        dense_pred = torch.randn(2, 32, 64)

        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            dense_pred=dense_pred,
            student_output=student_output,
            teacher_output=teacher_output,
            teacher_stem_output=teacher_stem_output,
            target_mask=target_mask,
            context_mask=target_mask,
        )
        losses = mjepa_model_no_sigreg.compute_losses(
            output=predictions,
            step=0,
            epoch=0,
        )

        assert isinstance(losses, MJEPALosses)
        assert isinstance(losses.jepa_loss, torch.Tensor)
        assert isinstance(losses.jepa_loss_cls, torch.Tensor)
        assert isinstance(losses.dense_pred_loss, torch.Tensor)
        assert losses.sigreg_loss == 0.0

    def test_compute_losses_dense_pred_matches_mse(self, mjepa_model_no_sigreg: MJEPA):
        """Dense prediction loss should match direct MSE."""
        pred = torch.zeros(1, 4, 2)
        pred_with_cls = torch.zeros(1, 4, 2)
        dense_pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        teacher_stem_output = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])

        dense_features = torch.zeros(1, 4, 2)
        student_output = ViTFeatures(dense_features, 0, 0)
        teacher_output = ViTFeatures(dense_features.clone(), 0, 0)

        target_mask = torch.ones(1, 4, dtype=torch.bool)

        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            dense_pred=dense_pred,
            student_output=student_output,
            teacher_output=teacher_output,
            teacher_stem_output=teacher_stem_output,
            target_mask=target_mask,
            context_mask=target_mask,
        )

        losses = mjepa_model_no_sigreg.compute_losses(output=predictions, step=0, epoch=0)
        expected = torch.nn.functional.mse_loss(dense_pred.float(), teacher_stem_output.float())
        assert isinstance(losses.dense_pred_loss, torch.Tensor)
        assert torch.allclose(losses.dense_pred_loss, expected)

    def test_compute_losses_with_sigreg(self, mjepa_model: MJEPA):
        """Test loss computation with SigREG loss enabled."""
        pred = torch.randn(2, 16, 64)
        pred_with_cls = torch.randn(2, 16, 64)
        dense_pred = torch.randn(2, 32, 64)
        teacher_stem_output = torch.randn(2, 32, 64)

        student_output, teacher_output = self._build_feature_pair(
            batch_size=BATCH_SIZE,
            num_tokens=NUM_VISUAL_TOKENS,
            hidden_size=HIDDEN_SIZE,
            num_register_tokens=0,
            num_cls_tokens=NUM_CLS_TOKENS,
        )
        target_mask = self._build_target_mask(
            batch_size=BATCH_SIZE,
            num_tokens=NUM_VISUAL_TOKENS,
            num_target_tokens=16,
        )

        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            dense_pred=dense_pred,
            student_output=student_output,
            teacher_output=teacher_output,
            teacher_stem_output=teacher_stem_output,
            target_mask=target_mask,
            context_mask=target_mask,
        )
        losses = mjepa_model.compute_losses(
            output=predictions,
            step=0,
            epoch=0,
        )

        assert isinstance(losses.sigreg_loss, torch.Tensor)
        assert losses.sigreg_loss.item() > 0


class TestMJEPAForward:
    """Test the main MJEPA forward pass."""

    @pytest.mark.parametrize("jepa_scale", [1, 2, 4])
    def test_forward_shapes_consistent(self, mjepa_model: MJEPA, dummy_batch, jepa_scale):
        """Forward should produce consistent token-aligned outputs."""
        mjepa_model.eval()

        with torch.no_grad():
            predictions = mjepa_model(dummy_batch, jepa_scale=jepa_scale, epoch=0)

        assert isinstance(predictions, MJEPAPredictions)
        assert predictions.pred_with_cls is not None
        assert predictions.pred.shape[0] == dummy_batch.shape[0]
        assert predictions.dense_pred.shape == predictions.teacher_stem_output.shape
        assert predictions.context_mask.shape == predictions.target_mask.shape

        overlap = predictions.context_mask & predictions.target_mask
        assert not overlap.any()

    def test_forward_training_mode(self, mjepa_model: MJEPA, dummy_batch):
        """Forward in train mode should preserve gradients for prediction paths."""
        mjepa_model.train()
        predictions = mjepa_model(dummy_batch, jepa_scale=2, epoch=0)

        assert predictions.pred.requires_grad
        assert predictions.dense_pred.requires_grad

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_forward_different_batch_sizes(self, mjepa_model: MJEPA, batch_size):
        """Forward should work for different batch sizes."""
        mjepa_model.eval()

        batch = torch.randn(batch_size, 3, 32, 32)
        with torch.no_grad():
            predictions = mjepa_model(batch, jepa_scale=2, epoch=0)

        assert predictions.pred.shape[0] == batch_size
        assert predictions.context_mask.shape[0] == batch_size
        assert predictions.target_mask.shape[0] == batch_size
        assert predictions.dense_pred.shape[0] == batch_size


class TestMJEPAEndToEnd:
    """Test end-to-end MJEPA training workflow."""

    def test_full_training_step(self, mjepa_model_no_sigreg: MJEPA, dummy_batch):
        """Test a full training step including loss computation and backprop."""
        mjepa_model_no_sigreg.train()

        predictions = mjepa_model_no_sigreg(dummy_batch, jepa_scale=2, epoch=0)

        losses = mjepa_model_no_sigreg.compute_losses(
            output=predictions,
            step=0,
            epoch=0,
        ).reduce()

        losses.backward()

        student_has_grad = any(
            param.requires_grad and param.grad is not None for param in mjepa_model_no_sigreg.student.parameters()
        )
        predictor_has_grad = any(
            param.requires_grad and param.grad is not None for param in mjepa_model_no_sigreg.predictor.parameters()
        )
        dense_head_has_grad = any(
            param.requires_grad and param.grad is not None
            for param in mjepa_model_no_sigreg.dense_pred_head.parameters()
        )

        assert student_has_grad
        assert predictor_has_grad
        assert dense_head_has_grad

    def test_teacher_update_workflow(self, mjepa_model_no_sigreg: MJEPA):
        """Test the teacher update workflow."""
        initial_weight = mjepa_model_no_sigreg.teacher.stem.patch.weight.data.clone()

        mjepa_model_no_sigreg.student.stem.patch.weight.data.fill_(1.0)

        mjepa_model_no_sigreg.update_teacher(step=0, total_steps=100)

        updated_weight = mjepa_model_no_sigreg.teacher.stem.patch.weight.data
        assert not torch.allclose(updated_weight, initial_weight)

    @pytest.mark.cuda
    def test_full_workflow_cuda(self):
        """Smoke test on CUDA when available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = JEPAConfig(dense_pred_loss_weight=1.0, sigreg_loss_weight=0.0)
        vit_config = _create_small_vit_config(dtype=torch.float32)
        backbone = vit_config.instantiate()
        predictor = CrossAttentionPredictor(backbone, depth=2)
        model = MJEPA(config, backbone, predictor).to("cuda")
        model.train()

        dummy_batch = torch.randn(2, 3, 32, 32, device="cuda")

        predictions = model(dummy_batch, jepa_scale=2, epoch=0)
        losses = model.compute_losses(
            output=predictions,
            step=0,
            epoch=0,
        )

        loss = losses.reduce()
        loss.backward()

        assert torch.isfinite(loss)
        assert loss.item() > 0
