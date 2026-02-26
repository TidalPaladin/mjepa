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
        gram_teacher="backbone",
        gram_teacher_epoch=10,
        gram_start_epoch=20,
        gram_update_interval_epoch=5,
        gram_resolution_scale=1.0,
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
def jepa_config_stem():
    """Create a JEPA configuration using stem-based Gram supervision."""
    return JEPAConfig(
        context_ratio=0.5,
        target_ratio=0.25,
        scale=2,
        momentum=0.98,
        predictor_depth=2,
        scheduled=False,
        gram_teacher="stem",
        gram_teacher_epoch=10,
        gram_start_epoch=20,
        gram_update_interval_epoch=5,
        gram_resolution_scale=1.0,
        gram_remove_neg=False,
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
    """Create an MJEPA model without Gram teacher for testing."""
    return MJEPA(jepa_config_no_gram, small_vit, predictor, dtype=torch.float32)


@pytest.fixture
def mjepa_model_stem(jepa_config_stem, small_vit, predictor):
    """Create an MJEPA model with stem-based Gram supervision."""
    return MJEPA(jepa_config_stem, small_vit, predictor, dtype=torch.float32)


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
        assert predictions.gram_teacher_output is None
        assert predictions.probes == {}

    def test_initialization_with_optional_fields(self, dummy_vit_features):
        """Test that MJEPAPredictions can be initialized with optional fields."""
        pred = torch.randn(2, 16, 64)
        pred_with_cls = torch.randn(2, 16, 64)
        context_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)
        target_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)
        gram_teacher_output = torch.randn(2, 64, 64)
        probes = {"probe1": torch.randn(2, 10)}

        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            student_output=dummy_vit_features,
            teacher_output=dummy_vit_features,
            context_mask=context_mask,
            target_mask=target_mask,
            gram_teacher_output=gram_teacher_output,
            probes=probes,
        )

        assert predictions.gram_teacher_output is not None
        assert torch.allclose(predictions.gram_teacher_output, gram_teacher_output)
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
        assert mjepa_model.gram_teacher is not None

    def test_initialization_no_gram(self, mjepa_model_no_gram):
        """Test that MJEPA model without Gram teacher is initialized correctly."""
        assert mjepa_model_no_gram.gram_teacher is None
        assert mjepa_model_no_gram.config.gram_start_epoch is None

    def test_initialization_stem_mode(self, mjepa_model_stem):
        """Test that stem mode does not allocate a separate Gram teacher model."""
        assert mjepa_model_stem.config.gram_teacher == "stem"
        assert mjepa_model_stem.gram_teacher is None

    def test_teacher_frozen(self, mjepa_model):
        """Test that teacher parameters are frozen."""
        for param in mjepa_model.teacher.parameters():
            assert not param.requires_grad

    def test_gram_teacher_frozen(self, mjepa_model):
        """Test that gram teacher parameters are frozen."""
        assert mjepa_model.gram_teacher is not None
        for param in mjepa_model.gram_teacher.parameters():
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

    def test_forward_gram_teacher(self, mjepa_model: MJEPA, dummy_batch):
        """Test forward pass through gram teacher."""
        context_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)
        output = mjepa_model.forward_gram_teacher(dummy_batch, context_mask)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == dummy_batch.shape[0]

    def test_forward_gram_teacher_with_rope_seed(self, mjepa_model: MJEPA, dummy_batch):
        """Test forward pass through gram teacher with rope seed."""
        context_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)
        rope_seed = 12345
        output = mjepa_model.forward_gram_teacher(dummy_batch, context_mask, rope_seed=rope_seed)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == dummy_batch.shape[0]

    def test_forward_gram_teacher_not_initialized(self, mjepa_model_no_gram: MJEPA, dummy_batch):
        """Test that forward_gram_teacher raises error when gram teacher is not initialized."""
        context_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)

        with pytest.raises(ValueError, match="Gram teacher is not initialized"):
            mjepa_model_no_gram.forward_gram_teacher(dummy_batch, context_mask)

    def test_forward_gram_teacher_eval_mode(self, mjepa_model: MJEPA, dummy_batch):
        """Test that gram teacher is in eval mode during forward pass."""
        context_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)
        assert mjepa_model.gram_teacher is not None
        mjepa_model.gram_teacher.train()
        _ = mjepa_model.forward_gram_teacher(dummy_batch, context_mask)
        assert not mjepa_model.gram_teacher.training

    def test_forward_gram_anchor(self, mjepa_model_stem: MJEPA, dummy_batch):
        """Test forward pass through detached stem-based Gram anchor."""
        context_mask = torch.randint(0, 2, (2, 64), dtype=torch.bool)
        output = mjepa_model_stem.forward_gram_anchor(dummy_batch, context_mask)
        expected = apply_mask(context_mask, mjepa_model_stem.teacher.stem(dummy_batch), fill_value=None)

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

    def test_update_gram_teacher_initial_setup(self, mjepa_model: MJEPA):
        """Test gram teacher initial setup at gram_teacher_epoch."""
        # Modify regular teacher
        mjepa_model.teacher.stem.patch.weight.data.fill_(1.0)

        # Store initial gram teacher weights
        assert mjepa_model.gram_teacher is not None
        initial_gram_weight = mjepa_model.gram_teacher.stem.patch.weight.data.clone()

        # Update gram teacher at gram_teacher_epoch (should copy from teacher)
        mjepa_model.update_gram_teacher(current_epoch=10)

        # Gram teacher should now match regular teacher
        assert torch.allclose(
            mjepa_model.gram_teacher.stem.patch.weight.data,
            mjepa_model.teacher.stem.patch.weight.data,
        )
        assert not torch.allclose(
            mjepa_model.gram_teacher.stem.patch.weight.data,
            initial_gram_weight,
        )

    def test_update_gram_teacher_periodic_update(self, mjepa_model: MJEPA):
        """Test gram teacher periodic update."""
        # First initialize gram teacher at epoch 10
        mjepa_model.update_gram_teacher(current_epoch=10)

        # Modify teacher weights
        mjepa_model.teacher.stem.patch.weight.data.fill_(2.0)

        # Store gram teacher weights before update
        assert mjepa_model.gram_teacher is not None
        weights_before = mjepa_model.gram_teacher.stem.patch.weight.data.clone()

        # Update at epoch 25 (gram_start_epoch=20, interval=5, so 20+5=25)
        mjepa_model.update_gram_teacher(current_epoch=25)

        # Gram teacher should be updated
        assert not torch.allclose(
            mjepa_model.gram_teacher.stem.patch.weight.data,
            weights_before,
        )

    def test_update_gram_teacher_no_update_between_intervals(self, mjepa_model: MJEPA):
        """Test that gram teacher is not updated between intervals."""
        # Initialize at epoch 10
        mjepa_model.update_gram_teacher(current_epoch=10)

        # Store weights
        assert mjepa_model.gram_teacher is not None
        weights_after_init = mjepa_model.gram_teacher.stem.patch.weight.data.clone()

        # Modify teacher
        mjepa_model.teacher.stem.patch.weight.data.fill_(2.0)

        # Update at epoch 22 (not at interval boundary)
        mjepa_model.update_gram_teacher(current_epoch=22)

        # Gram teacher should not have changed
        assert torch.allclose(
            mjepa_model.gram_teacher.stem.patch.weight.data,
            weights_after_init,
        )

    def test_update_gram_teacher_no_gram(self, mjepa_model_no_gram):
        """Test that update_gram_teacher does nothing when gram teacher is None."""
        # Should not raise any errors
        mjepa_model_no_gram.update_gram_teacher(current_epoch=10)
        mjepa_model_no_gram.update_gram_teacher(current_epoch=20)

    def test_update_gram_teacher_resolution_change_starts_cooldown(self, mjepa_model: MJEPA):
        """Test that resolution changes pause gram teacher updates until cooldown ends."""
        resolution_change_epoch = 20
        cooldown_probe_epoch = 22
        expected_cooldown_end_epoch = resolution_change_epoch + mjepa_model.config.gram_update_interval_epoch

        mjepa_model.update_gram_teacher(current_epoch=10)
        assert mjepa_model.gram_teacher is not None
        weights_after_init = mjepa_model.gram_teacher.stem.patch.weight.data.clone()

        mjepa_model.teacher.stem.patch.weight.data.fill_(2.0)
        mjepa_model.update_gram_teacher(current_epoch=resolution_change_epoch, resolution_changed=True)
        mjepa_model.update_gram_teacher(current_epoch=cooldown_probe_epoch)

        assert mjepa_model._gram_cooldown_end_epoch == expected_cooldown_end_epoch
        assert torch.allclose(mjepa_model.gram_teacher.stem.patch.weight.data, weights_after_init)

    def test_update_gram_teacher_resets_at_cooldown_end(self, mjepa_model: MJEPA):
        """Test that gram teacher is reset from teacher when cooldown reaches its end epoch."""
        resolution_change_epoch = 20
        cooldown_end_epoch = resolution_change_epoch + mjepa_model.config.gram_update_interval_epoch

        mjepa_model.update_gram_teacher(current_epoch=10)
        assert mjepa_model.gram_teacher is not None

        mjepa_model.teacher.stem.patch.weight.data.fill_(3.0)
        mjepa_model.update_gram_teacher(current_epoch=resolution_change_epoch, resolution_changed=True)
        mjepa_model.update_gram_teacher(current_epoch=cooldown_end_epoch)

        assert torch.allclose(
            mjepa_model.gram_teacher.stem.patch.weight.data,
            mjepa_model.teacher.stem.patch.weight.data,
        )
        assert mjepa_model._gram_cooldown_end_epoch is None

    def test_update_gram_teacher_overlapping_resolution_changes_restart_cooldown(self, mjepa_model: MJEPA):
        """Test that a new resolution change restarts the cooldown window."""
        first_change_epoch = 20
        second_change_epoch = 23
        interval = mjepa_model.config.gram_update_interval_epoch

        mjepa_model.update_gram_teacher(current_epoch=10)
        assert mjepa_model.gram_teacher is not None
        gram_before = mjepa_model.gram_teacher.stem.patch.weight.data.clone()

        mjepa_model.update_gram_teacher(current_epoch=first_change_epoch, resolution_changed=True)
        assert mjepa_model._gram_cooldown_end_epoch == first_change_epoch + interval

        mjepa_model.update_gram_teacher(current_epoch=second_change_epoch, resolution_changed=True)
        assert mjepa_model._gram_cooldown_end_epoch == second_change_epoch + interval

        mjepa_model.teacher.stem.patch.weight.data.fill_(4.0)
        mjepa_model.update_gram_teacher(current_epoch=25)
        assert torch.allclose(mjepa_model.gram_teacher.stem.patch.weight.data, gram_before)

        mjepa_model.update_gram_teacher(current_epoch=second_change_epoch + interval)
        assert torch.allclose(
            mjepa_model.gram_teacher.stem.patch.weight.data,
            mjepa_model.teacher.stem.patch.weight.data,
        )
        assert mjepa_model._gram_cooldown_end_epoch is None

    def test_update_gram_teacher_interval_zero_keeps_initial_snapshot(self, small_vit, predictor):
        """Test that interval zero disables periodic updates and resolution-triggered resets."""
        config = JEPAConfig(
            context_ratio=0.5,
            target_ratio=0.25,
            scale=2,
            momentum=0.98,
            predictor_depth=2,
            scheduled=False,
            gram_teacher_epoch=10,
            gram_start_epoch=20,
            gram_update_interval_epoch=0,
            gram_resolution_scale=1.0,
            gram_remove_neg=False,
            gram_loss_weight=1.0,
            sigreg_loss_weight=0.0001,
        )
        model = MJEPA(config, small_vit, predictor, dtype=torch.float32)
        assert model.gram_teacher is not None

        model.teacher.stem.patch.weight.data.fill_(1.0)
        model.update_gram_teacher(current_epoch=10)
        initial_snapshot = model.gram_teacher.stem.patch.weight.data.clone()

        model.teacher.stem.patch.weight.data.fill_(2.0)
        model.update_gram_teacher(current_epoch=25)
        model.update_gram_teacher(current_epoch=30, resolution_changed=True)
        model.update_gram_teacher(current_epoch=40)

        assert torch.allclose(model.gram_teacher.stem.patch.weight.data, initial_snapshot)
        assert model._gram_cooldown_end_epoch is None

    def test_update_gram_teacher_cooldown_end_before_teacher_epoch_does_not_resync(self, mjepa_model: MJEPA):
        """Test cooldown completion does not bypass the gram_teacher_epoch gate."""
        assert mjepa_model.gram_teacher is not None
        initial_gram_weight = mjepa_model.gram_teacher.stem.patch.weight.data.clone()
        gram_teacher_epoch = mjepa_model.config.gram_teacher_epoch
        cooldown_interval = mjepa_model.config.gram_update_interval_epoch

        resolution_change_epoch = 3
        cooldown_end_epoch = resolution_change_epoch + cooldown_interval
        assert cooldown_end_epoch < gram_teacher_epoch

        mjepa_model.teacher.stem.patch.weight.data.fill_(5.0)
        mjepa_model.update_gram_teacher(current_epoch=resolution_change_epoch, resolution_changed=True)
        mjepa_model.update_gram_teacher(current_epoch=cooldown_end_epoch)

        # Gram teacher should remain unchanged until gram_teacher_epoch is reached.
        assert torch.allclose(mjepa_model.gram_teacher.stem.patch.weight.data, initial_gram_weight)

        mjepa_model.update_gram_teacher(current_epoch=gram_teacher_epoch)
        assert torch.allclose(
            mjepa_model.gram_teacher.stem.patch.weight.data,
            mjepa_model.teacher.stem.patch.weight.data,
        )

    def test_update_gram_teacher_stem_mode_tracks_cooldown(self, mjepa_model_stem: MJEPA):
        """Test that stem mode still honors cooldown windows for Gram usage."""
        interval = mjepa_model_stem.config.gram_update_interval_epoch
        resolution_change_epoch = 20
        cooldown_end_epoch = resolution_change_epoch + interval

        mjepa_model_stem.update_gram_teacher(current_epoch=resolution_change_epoch, resolution_changed=True)
        assert mjepa_model_stem._gram_cooldown_end_epoch == cooldown_end_epoch

        mjepa_model_stem.update_gram_teacher(current_epoch=cooldown_end_epoch)
        assert mjepa_model_stem._gram_cooldown_end_epoch is None


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
        assert losses.gram_loss == 0.0  # no gram teacher

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

        gram_teacher_output = torch.randn(2, 64, 64)

        # Test with epoch >= gram_start_epoch
        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            student_output=student_output,
            teacher_output=teacher_output,
            target_mask=target_mask,
            context_mask=target_mask,
            gram_teacher_output=gram_teacher_output,
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

    def test_compute_losses_gram_during_cooldown_is_zero(self, mjepa_model: MJEPA):
        """Test that gram loss is disabled during the resolution cooldown window."""
        resolution_change_epoch = 20
        cooldown_epoch = 22

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

        mjepa_model.update_gram_teacher(current_epoch=resolution_change_epoch, resolution_changed=True)

        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            student_output=student_output,
            teacher_output=teacher_output,
            target_mask=target_mask,
            context_mask=target_mask,
            gram_teacher_output=None,
        )
        losses = mjepa_model.compute_losses(
            output=predictions,
            step=0,
            epoch=cooldown_epoch,
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

        gram_teacher_output = torch.randn(2, 64, 64)

        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            student_output=student_output,
            teacher_output=teacher_output,
            target_mask=target_mask,
            context_mask=target_mask,
            gram_teacher_output=gram_teacher_output,
        )
        losses = mjepa_model.compute_losses(
            output=predictions,
            step=0,
            epoch=epoch,
        )

        assert isinstance(losses.gram_loss, torch.Tensor)

    def test_compute_losses_with_stem_gram(self, mjepa_model_stem: MJEPA):
        """Test loss computation with stem-based Gram source."""
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
            gram_teacher_output=torch.randn(2, 64, 64),
        )
        losses = mjepa_model_stem.compute_losses(output=predictions, step=0, epoch=20)
        assert isinstance(losses.gram_loss, torch.Tensor)
        assert losses.gram_loss.item() > 0


class TestMJEPAForward:
    """Test the main MJEPA forward pass."""

    def test_forward_basic(self, mjepa_model_no_gram: MJEPA, dummy_batch):
        """Test basic forward pass without gram teacher."""
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
        assert predictions.gram_teacher_output is None

    def test_forward_with_gram(self, mjepa_model, dummy_batch):
        """Test forward pass with gram teacher."""
        mjepa_model.eval()

        # Test before gram_start_epoch
        with torch.no_grad():
            predictions_before = mjepa_model(dummy_batch, jepa_scale=2, epoch=15)
        assert predictions_before.gram_teacher_output is None

        # Test after gram_start_epoch
        with torch.no_grad():
            predictions_after = mjepa_model(dummy_batch, jepa_scale=2, epoch=20)
        assert predictions_after.gram_teacher_output is not None
        assert isinstance(predictions_after.gram_teacher_output, torch.Tensor)

    def test_forward_disables_gram_during_resolution_cooldown(self, mjepa_model, dummy_batch):
        """Test that gram teacher forward is skipped during cooldown and resumes at cooldown end."""
        resolution_change_epoch = 20
        cooldown_epoch = 22
        cooldown_end_epoch = resolution_change_epoch + mjepa_model.config.gram_update_interval_epoch

        mjepa_model.eval()

        mjepa_model.update_gram_teacher(current_epoch=resolution_change_epoch, resolution_changed=True)
        with torch.no_grad():
            predictions_during = mjepa_model(dummy_batch, jepa_scale=2, epoch=cooldown_epoch)
        assert predictions_during.gram_teacher_output is None

        mjepa_model.update_gram_teacher(current_epoch=cooldown_end_epoch)
        with torch.no_grad():
            predictions_after = mjepa_model(dummy_batch, jepa_scale=2, epoch=cooldown_end_epoch)
        assert predictions_after.gram_teacher_output is not None

    def test_forward_with_stem_gram(self, mjepa_model_stem, dummy_batch):
        """Test forward pass with stem-based Gram source."""
        mjepa_model_stem.eval()

        with torch.no_grad():
            predictions_before = mjepa_model_stem(dummy_batch, jepa_scale=2, epoch=15)
        assert predictions_before.gram_teacher_output is None

        with torch.no_grad():
            predictions_after = mjepa_model_stem(dummy_batch, jepa_scale=2, epoch=20)
        assert predictions_after.gram_teacher_output is not None
        assert isinstance(predictions_after.gram_teacher_output, torch.Tensor)
        assert not predictions_after.gram_teacher_output.requires_grad

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

    def test_gram_teacher_workflow(self, mjepa_model: MJEPA):
        """Test the gram teacher update workflow."""
        # Initialize gram teacher at epoch 10
        mjepa_model.update_gram_teacher(current_epoch=10)
        assert mjepa_model.gram_teacher is not None
        initial_weight = mjepa_model.gram_teacher.stem.patch.weight.data.clone()

        # Modify regular teacher
        mjepa_model.teacher.stem.patch.weight.data.fill_(2.0)

        # Update gram teacher at epoch 25
        mjepa_model.update_gram_teacher(current_epoch=25)

        # Check that gram teacher was updated
        updated_weight = mjepa_model.gram_teacher.stem.patch.weight.data
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
