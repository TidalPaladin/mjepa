import pytest
import torch
from vit import ViT, ViTConfig, ViTFeatures
from vit.tokens import apply_mask

from mjepa.jepa import CrossAttentionPredictor, JEPAConfig
from mjepa.model import MJEPA, MJEPALosses, MJEPAPredictions


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
        gram_start_epoch=20,
        gram_remove_neg=False,
        gram_loss_weight=1.0,
        sigreg_loss_weight=0.0001,
    )


@pytest.fixture
def jepa_config_no_gram():
    """Create a JEPA configuration without Gram loss for testing."""
    return JEPAConfig(
        context_ratio=0.5,
        target_ratio=0.25,
        scale=2,
        momentum=0.98,
        predictor_depth=2,
        scheduled=False,
        gram_start_epoch=None,
        gram_loss_weight=1.0,
        sigreg_loss_weight=0.0001,
    )


@pytest.fixture
def small_vit_config():
    """Create a small ViT configuration for testing."""
    return ViTConfig(
        in_channels=3,
        hidden_size=64,
        patch_size=[4, 4],
        img_size=[32, 32],
        depth=2,
        num_attention_heads=4,
        ffn_hidden_size=128,
        activation="gelu",
        num_register_tokens=2,
        num_cls_tokens=2,
        pos_enc="rope",
        rope_base=100,
        dtype=torch.float32,
    )


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
def mjepa_model_no_gram(jepa_config_no_gram, small_vit, predictor):
    """Create an MJEPA model without Gram loss for testing."""
    return MJEPA(jepa_config_no_gram, small_vit, predictor, dtype=torch.float32)


@pytest.fixture
def dummy_batch():
    """Create a dummy batch of images."""
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def dummy_vit_features():
    """Create dummy ViT features for testing."""
    batch_size = 2
    num_tokens = 64
    hidden_size = 64
    num_cls_tokens = 2
    num_register_tokens = 2

    # dense_features includes cls, register, and visual tokens
    dense_features = torch.randn(batch_size, num_tokens + num_cls_tokens + num_register_tokens, hidden_size)
    return ViTFeatures(dense_features, num_register_tokens, num_cls_tokens)


class TestMJEPALosses:
    """Test the MJEPALosses dataclass."""

    def test_initialization(self):
        """Test that MJEPALosses can be initialized."""
        jepa_loss = torch.tensor(1.0)
        jepa_loss_cls = torch.tensor(0.5)
        sigreg_loss = torch.tensor(0.01)
        gram_loss = torch.tensor(0.1)

        losses = MJEPALosses(
            jepa_loss=jepa_loss,
            jepa_loss_cls=jepa_loss_cls,
            sigreg_loss=sigreg_loss,
            gram_loss=gram_loss,
            gram_loss_weight=1.0,
            sigreg_loss_weight=1e-4,
        )

        assert losses.jepa_loss == jepa_loss
        assert losses.jepa_loss_cls == jepa_loss_cls
        assert losses.sigreg_loss == sigreg_loss
        assert losses.gram_loss == gram_loss
        assert losses.gram_loss_weight == 1.0
        assert losses.sigreg_loss_weight == 1e-4

    def test_initialization_with_float_losses(self):
        """Test that MJEPALosses can be initialized with float values."""
        jepa_loss = torch.tensor(1.0)

        losses = MJEPALosses(
            jepa_loss=jepa_loss,
            jepa_loss_cls=0.5,
            sigreg_loss=0.01,
            gram_loss=0.1,
        )

        assert losses.jepa_loss == jepa_loss
        assert losses.jepa_loss_cls == 0.5
        assert losses.sigreg_loss == 0.01
        assert losses.gram_loss == 0.1

    @pytest.mark.parametrize(
        "gram_weight,sigreg_weight",
        [
            (1.0, 1e-4),
            (0.5, 1e-3),
            (2.0, 0.0),
        ],
    )
    def test_reduce(self, gram_weight, sigreg_weight):
        """Test that reduce combines losses correctly."""
        jepa_loss = torch.tensor(1.0)
        jepa_loss_cls = torch.tensor(0.5)
        sigreg_loss = torch.tensor(0.01)
        gram_loss = torch.tensor(0.1)

        losses = MJEPALosses(
            jepa_loss=jepa_loss,
            jepa_loss_cls=jepa_loss_cls,
            sigreg_loss=sigreg_loss,
            gram_loss=gram_loss,
            gram_loss_weight=gram_weight,
            sigreg_loss_weight=sigreg_weight,
        )

        reduced = losses.reduce()
        expected = jepa_loss + jepa_loss_cls + gram_loss * gram_weight + sigreg_loss * sigreg_weight

        assert isinstance(reduced, torch.Tensor)
        assert torch.allclose(reduced, expected)

    def test_reduce_with_zero_weights(self):
        """Test that reduce works with zero weights."""
        jepa_loss = torch.tensor(1.0)
        jepa_loss_cls = torch.tensor(0.5)
        sigreg_loss = 0.01
        gram_loss = 0.1

        losses = MJEPALosses(
            jepa_loss=jepa_loss,
            jepa_loss_cls=jepa_loss_cls,
            sigreg_loss=sigreg_loss,
            gram_loss=gram_loss,
            gram_loss_weight=0.0,
            sigreg_loss_weight=0.0,
        )

        reduced = losses.reduce()
        expected = jepa_loss + jepa_loss_cls

        assert isinstance(reduced, torch.Tensor)
        assert torch.allclose(reduced, expected)


class TestMJEPAPredictions:
    """Test the MJEPAPredictions dataclass."""

    def test_initialization(self, dummy_vit_features):
        """Test that MJEPAPredictions can be initialized."""
        pred = torch.randn(2, 16, 64)
        pred_with_cls = torch.randn(2, 16, 64)
        context_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)
        target_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)

        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            student_output=dummy_vit_features,
            teacher_output=dummy_vit_features,
            context_mask=context_mask,
            target_mask=target_mask,
        )

        assert torch.allclose(predictions.pred, pred)
        assert predictions.pred_with_cls is not None
        assert torch.allclose(predictions.pred_with_cls, pred_with_cls)
        assert predictions.student_output == dummy_vit_features
        assert predictions.teacher_output == dummy_vit_features
        assert torch.all(predictions.context_mask == context_mask)
        assert torch.all(predictions.target_mask == target_mask)
        assert predictions.gram_anchor_output is None
        assert predictions.probes == {}

    def test_initialization_with_optional_fields(self, dummy_vit_features):
        """Test that MJEPAPredictions can be initialized with optional fields."""
        pred = torch.randn(2, 16, 64)
        pred_with_cls = torch.randn(2, 16, 64)
        context_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)
        target_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)
        gram_anchor_output = torch.randn(2, 64, 64)
        probes = {"probe1": torch.randn(2, 10)}

        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            student_output=dummy_vit_features,
            teacher_output=dummy_vit_features,
            context_mask=context_mask,
            target_mask=target_mask,
            gram_anchor_output=gram_anchor_output,
            probes=probes,
        )

        assert predictions.gram_anchor_output is not None
        assert torch.allclose(predictions.gram_anchor_output, gram_anchor_output)
        assert "probe1" in predictions.probes
        assert torch.allclose(predictions.probes["probe1"], probes["probe1"])


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

    def test_initialization_no_gram(self, mjepa_model_no_gram):
        """Test that MJEPA model without Gram loss is initialized correctly."""
        assert mjepa_model_no_gram.config.gram_start_epoch is None

    def test_teacher_frozen(self, mjepa_model):
        """Test that teacher parameters are frozen."""
        for param in mjepa_model.teacher.parameters():
            assert not param.requires_grad

    def test_img_size_property(self, mjepa_model):
        """Test the img_size property."""
        img_size = mjepa_model.img_size
        assert img_size == (32, 32) or img_size == [32, 32]
        assert len(img_size) == 2
        assert img_size[0] == 32
        assert img_size[1] == 32


class TestMJEPAForwardPasses:
    """Test MJEPA forward pass methods."""

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

    def test_forward_student_with_rope_seed(self, mjepa_model: MJEPA, dummy_batch):
        """Test forward pass through student with rope seed."""
        context_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)
        rope_seed = 12345
        output = mjepa_model.forward_student(dummy_batch, context_mask, rope_seed=rope_seed)

        assert isinstance(output, ViTFeatures)
        assert output.dense_features.shape[0] == dummy_batch.shape[0]

    def test_forward_predictor(self, mjepa_model: MJEPA):
        """Test forward pass through predictor."""
        tokenized_size = (8, 8)
        num_tokens = tokenized_size[0] * tokenized_size[1]
        context = torch.randn(2, 32, 64)
        context_mask = torch.zeros(2, num_tokens, dtype=torch.bool)
        context_mask[:, :32] = True  # Mark first 32 as context
        target_mask = torch.zeros(2, num_tokens, dtype=torch.bool)
        target_mask[:, 32:48] = True  # Mark next 16 as target (non-overlapping)

        output = mjepa_model.forward_predictor(tokenized_size, context, context_mask, target_mask)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2

    def test_forward_probe(self, mjepa_model: MJEPA, dummy_vit_features):
        """Test forward pass through probe (currently returns empty dict)."""
        output = mjepa_model.forward_probe(dummy_vit_features)

        assert isinstance(output, dict)
        assert len(output) == 0

    def test_forward_gram_anchor(self, mjepa_model: MJEPA, dummy_batch):
        """Test forward pass through detached stem-based Gram anchor."""
        context_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)
        output = mjepa_model.forward_gram_anchor(dummy_batch, context_mask)
        expected = apply_mask(context_mask, mjepa_model.student.stem(dummy_batch), fill_value=None)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == dummy_batch.shape[0]
        assert torch.allclose(output, expected)
        assert not output.requires_grad


class TestMJEPATeacherUpdate:
    """Test MJEPA teacher update methods."""

    def test_update_teacher(self, mjepa_model: MJEPA):
        """Test updating the teacher model."""
        # Store initial teacher weights
        initial_weight = mjepa_model.teacher.stem.patch.weight.data.clone()

        # Modify student weights
        mjepa_model.student.stem.patch.weight.data.fill_(1.0)

        # Update teacher
        step = 0
        total_steps = 100
        mjepa_model.update_teacher(step, total_steps)

        # Teacher weights should have changed
        assert not torch.allclose(mjepa_model.teacher.stem.patch.weight.data, initial_weight)

    def test_update_teacher_scheduled(self):
        """Test updating teacher with scheduled momentum."""

        # Create config with scheduled momentum
        config = JEPAConfig(
            momentum=0.9,
            scheduled=True,
            gram_start_epoch=None,
        )
        vit_config = ViTConfig(
            in_channels=3,
            hidden_size=64,
            patch_size=[4, 4],
            img_size=[32, 32],
            depth=2,
            num_attention_heads=4,
            ffn_hidden_size=128,
        )
        backbone = vit_config.instantiate()
        predictor = CrossAttentionPredictor(backbone, depth=2)
        model = MJEPA(config, backbone, predictor)

        # Store initial teacher weights
        initial_weight = model.teacher.stem.patch.weight.data.clone()

        # Modify student weights
        model.student.stem.patch.weight.data.fill_(1.0)

        # Update teacher at different steps
        model.update_teacher(0, 100)
        weight_step_0 = model.teacher.stem.patch.weight.data.clone()

        model.update_teacher(50, 100)
        weight_step_50 = model.teacher.stem.patch.weight.data.clone()

        # Weights should be different at different steps
        assert not torch.allclose(weight_step_0, initial_weight)
        assert not torch.allclose(weight_step_50, weight_step_0)


class TestMJEPAComputeLosses:
    """Test MJEPA loss computation."""

    def test_compute_losses_basic(self, mjepa_model_no_gram: MJEPA):
        """Test basic loss computation without gram loss."""
        pred = torch.randn(2, 16, 64)
        pred_with_cls = torch.randn(2, 16, 64)

        batch_size = 2
        num_tokens = 64
        hidden_size = 64
        dense_features = torch.randn(batch_size, num_tokens + 4, hidden_size)
        student_output = ViTFeatures(dense_features, 2, 2)
        teacher_output = ViTFeatures(dense_features.clone(), 2, 2)

        target_mask = torch.zeros(2, 64, dtype=torch.bool)
        target_mask[:, :16] = True

        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            student_output=student_output,
            teacher_output=teacher_output,
            target_mask=target_mask,
            context_mask=target_mask,
        )
        losses = mjepa_model_no_gram.compute_losses(
            output=predictions,
            step=0,
            epoch=0,
        )

        assert isinstance(losses, MJEPALosses)
        assert isinstance(losses.jepa_loss, torch.Tensor)
        assert isinstance(losses.jepa_loss_cls, torch.Tensor)
        # sigreg_loss can be a Tensor or 0.0 depending on config
        assert isinstance(losses.sigreg_loss, (torch.Tensor, float))
        assert losses.gram_loss == 0.0  # no gram anchor

    def test_compute_losses_with_sigreg(self):
        """Test loss computation with SigREG loss enabled."""
        config = JEPAConfig(
            sigreg_loss_weight=1e-4,
            gram_start_epoch=None,
        )
        vit_config = ViTConfig(
            in_channels=3,
            hidden_size=64,
            patch_size=[4, 4],
            img_size=[32, 32],
            depth=2,
            num_attention_heads=4,
            ffn_hidden_size=128,
            num_cls_tokens=2,
        )
        backbone = vit_config.instantiate()
        predictor = CrossAttentionPredictor(backbone, depth=2)
        model = MJEPA(config, backbone, predictor)

        pred = torch.randn(2, 16, 64)
        pred_with_cls = torch.randn(2, 16, 64)

        batch_size = 2
        num_tokens = 64
        hidden_size = 64
        dense_features = torch.randn(batch_size, num_tokens + 2, hidden_size)
        student_output = ViTFeatures(dense_features, 0, 2)
        teacher_output = ViTFeatures(dense_features.clone(), 0, 2)

        target_mask = torch.zeros(2, 64, dtype=torch.bool)
        target_mask[:, :16] = True

        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            student_output=student_output,
            teacher_output=teacher_output,
            target_mask=target_mask,
            context_mask=target_mask,
        )
        losses = model.compute_losses(
            output=predictions,
            step=0,
            epoch=0,
        )

        assert isinstance(losses.sigreg_loss, torch.Tensor)
        assert losses.sigreg_loss.item() > 0

    def test_compute_losses_with_gram(self, mjepa_model: MJEPA):
        """Test loss computation with Gram loss."""
        pred = torch.randn(2, 16, 64)
        pred_with_cls = torch.randn(2, 16, 64)

        batch_size = 2
        num_tokens = 64
        hidden_size = 64
        dense_features = torch.randn(batch_size, num_tokens + 4, hidden_size)
        student_output = ViTFeatures(dense_features, 2, 2)
        teacher_output = ViTFeatures(dense_features.clone(), 2, 2)

        target_mask = torch.zeros(2, 64, dtype=torch.bool)
        target_mask[:, :16] = True

        gram_anchor_output = torch.randn(2, 64, 64)

        # Test with epoch >= gram_start_epoch
        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            student_output=student_output,
            teacher_output=teacher_output,
            target_mask=target_mask,
            context_mask=target_mask,
            gram_anchor_output=gram_anchor_output,
        )
        losses = mjepa_model.compute_losses(
            output=predictions,
            step=0,
            epoch=20,  # >= gram_start_epoch
        )

        assert isinstance(losses.gram_loss, torch.Tensor)
        assert losses.gram_loss.item() > 0

    def test_compute_losses_gram_before_start_epoch(self, mjepa_model: MJEPA):
        """Test that gram loss is zero before gram_start_epoch."""
        pred = torch.randn(2, 16, 64)
        pred_with_cls = torch.randn(2, 16, 64)

        batch_size = 2
        num_tokens = 64
        hidden_size = 64
        dense_features = torch.randn(batch_size, num_tokens + 4, hidden_size)
        student_output = ViTFeatures(dense_features, 2, 2)
        teacher_output = ViTFeatures(dense_features.clone(), 2, 2)

        target_mask = torch.zeros(2, 64, dtype=torch.bool)
        target_mask[:, :16] = True

        # Test with epoch < gram_start_epoch
        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            student_output=student_output,
            teacher_output=teacher_output,
            target_mask=target_mask,
            context_mask=target_mask,
        )
        losses = mjepa_model.compute_losses(
            output=predictions,
            step=0,
            epoch=15,  # < gram_start_epoch (20)
        )

        assert losses.gram_loss == 0.0

    @pytest.mark.parametrize("epoch", [20, 25, 30])
    def test_compute_losses_gram_different_epochs(self, mjepa_model: MJEPA, epoch):
        """Test gram loss computation at different epochs."""
        pred = torch.randn(2, 16, 64)
        pred_with_cls = torch.randn(2, 16, 64)

        batch_size = 2
        num_tokens = 64
        hidden_size = 64
        dense_features = torch.randn(batch_size, num_tokens + 4, hidden_size)
        student_output = ViTFeatures(dense_features, 2, 2)
        teacher_output = ViTFeatures(dense_features.clone(), 2, 2)

        target_mask = torch.zeros(2, 64, dtype=torch.bool)
        target_mask[:, :16] = True

        gram_anchor_output = torch.randn(2, 64, 64)

        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            student_output=student_output,
            teacher_output=teacher_output,
            target_mask=target_mask,
            context_mask=target_mask,
            gram_anchor_output=gram_anchor_output,
        )
        losses = mjepa_model.compute_losses(
            output=predictions,
            step=0,
            epoch=epoch,
        )

        assert isinstance(losses.gram_loss, torch.Tensor)


class TestMJEPAForward:
    """Test the main MJEPA forward pass."""

    def test_forward_basic(self, mjepa_model_no_gram: MJEPA, dummy_batch):
        """Test basic forward pass without Gram anchor."""
        mjepa_model_no_gram.eval()

        with torch.no_grad():
            predictions = mjepa_model_no_gram(dummy_batch, jepa_scale=2, epoch=0)

        assert isinstance(predictions, MJEPAPredictions)
        assert predictions.pred_with_cls is not None
        assert predictions.pred.shape[0] == dummy_batch.shape[0]
        assert predictions.pred_with_cls.shape[0] == dummy_batch.shape[0]
        assert isinstance(predictions.student_output, ViTFeatures)
        assert isinstance(predictions.teacher_output, ViTFeatures)
        assert predictions.context_mask.shape[0] == dummy_batch.shape[0]
        assert predictions.target_mask.shape[0] == dummy_batch.shape[0]
        assert predictions.gram_anchor_output is None

    def test_forward_with_gram(self, mjepa_model, dummy_batch):
        """Test forward pass with detached stem Gram anchor."""
        mjepa_model.eval()

        # Test before gram_start_epoch
        with torch.no_grad():
            predictions_before = mjepa_model(dummy_batch, jepa_scale=2, epoch=15)
        assert predictions_before.gram_anchor_output is None

        # Test after gram_start_epoch
        with torch.no_grad():
            predictions_after = mjepa_model(dummy_batch, jepa_scale=2, epoch=20)
        assert predictions_after.gram_anchor_output is not None
        assert isinstance(predictions_after.gram_anchor_output, torch.Tensor)

    @pytest.mark.parametrize("jepa_scale", [1, 2, 4])
    def test_forward_different_scales(self, mjepa_model_no_gram, dummy_batch, jepa_scale):
        """Test forward pass with different JEPA scales."""
        mjepa_model_no_gram.eval()

        with torch.no_grad():
            predictions = mjepa_model_no_gram(dummy_batch, jepa_scale=jepa_scale, epoch=0)

        assert isinstance(predictions, MJEPAPredictions)
        assert predictions.pred.shape[0] == dummy_batch.shape[0]

    def test_forward_training_mode(self, mjepa_model_no_gram, dummy_batch):
        """Test forward pass in training mode."""
        mjepa_model_no_gram.train()

        predictions = mjepa_model_no_gram(dummy_batch, jepa_scale=2, epoch=0)

        assert isinstance(predictions, MJEPAPredictions)
        assert predictions.pred.requires_grad

    def test_forward_shapes_consistency(self, mjepa_model_no_gram, dummy_batch):
        """Test that output shapes are consistent."""
        mjepa_model_no_gram.eval()

        with torch.no_grad():
            predictions = mjepa_model_no_gram(dummy_batch, jepa_scale=2, epoch=0)

        # Check that masks have same shape (batch_size, num_tokens)
        assert predictions.context_mask.shape == predictions.target_mask.shape

        # Check that context and target masks don't overlap
        overlap = predictions.context_mask & predictions.target_mask
        assert not overlap.any()

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_forward_different_batch_sizes(self, mjepa_model_no_gram, batch_size):
        """Test forward pass with different batch sizes."""
        mjepa_model_no_gram.eval()

        dummy_batch = torch.randn(batch_size, 3, 32, 32)

        with torch.no_grad():
            predictions = mjepa_model_no_gram(dummy_batch, jepa_scale=2, epoch=0)

        assert predictions.pred.shape[0] == batch_size
        assert predictions.context_mask.shape[0] == batch_size
        assert predictions.target_mask.shape[0] == batch_size


class TestMJEPAEndToEnd:
    """Test end-to-end MJEPA training workflow."""

    def test_full_training_step(self, mjepa_model_no_gram: MJEPA, dummy_batch):
        """Test a full training step including loss computation and backprop."""
        mjepa_model_no_gram.train()

        # Forward pass
        predictions = mjepa_model_no_gram(dummy_batch, jepa_scale=2, epoch=0)

        # Compute losses
        losses = mjepa_model_no_gram.compute_losses(
            output=predictions,
            step=0,
            epoch=0,
        ).reduce()

        # Reduce and backward
        losses.backward()

        # Check that student has gradients
        for param in mjepa_model_no_gram.student.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break

    def test_teacher_update_workflow(self, mjepa_model_no_gram: MJEPA):
        """Test the teacher update workflow."""
        # Store initial teacher weight
        initial_weight = mjepa_model_no_gram.teacher.stem.patch.weight.data.clone()

        # Simulate training by modifying student
        mjepa_model_no_gram.student.stem.patch.weight.data.fill_(1.0)

        # Update teacher
        mjepa_model_no_gram.update_teacher(step=0, total_steps=100)

        # Check that teacher was updated
        updated_weight = mjepa_model_no_gram.teacher.stem.patch.weight.data
        assert not torch.allclose(updated_weight, initial_weight)

    @pytest.mark.cuda
    def test_full_workflow(self):
        config = JEPAConfig(gram_start_epoch=None, sigreg_loss_weight=0.0)
        vit_config = ViTConfig(
            in_channels=3,
            hidden_size=64,
            patch_size=[4, 4],
            img_size=[32, 32],
            depth=2,
            num_attention_heads=4,
            ffn_hidden_size=128,
        )
        backbone = vit_config.instantiate()
        predictor = CrossAttentionPredictor(backbone, depth=2)
        model = MJEPA(config, backbone, predictor).to("cuda")
        model.train()

        dummy_batch = torch.randn(2, 3, 32, 32, device="cuda")

        # Forward pass
        predictions = model(dummy_batch, jepa_scale=2, epoch=0)

        # Compute losses
        losses = model.compute_losses(
            output=predictions,
            step=0,
            epoch=0,
        )

        # Reduce and backward
        loss = losses.reduce()
        loss.backward()

        # Check that loss is finite
        assert torch.isfinite(loss)
        assert loss.item() > 0
