import pytest
import torch
from vit import ViT, ViTConfig, ViTFeatures
from vit.tokens import apply_mask

from mjepa.jepa import CrossAttentionPredictor, JEPAConfig
from mjepa.model import MJEPA, MJEPALosses, MJEPAPredictions


BATCH_SIZE = 2
IMAGE_SIZE = 32
PATCH_SIZE = 4
HIDDEN_SIZE = 64
NUM_TOKENS = 64
NUM_CLS_TOKENS = 2
NUM_REGISTER_TOKENS = 2
WARMUP_EPOCHS = 20
TOKENIZED_SIZE = (8, 8)
NUM_CONTEXT_TOKENS = 32
NUM_TARGET_TOKENS = 16


@pytest.fixture
def jepa_config():
    return JEPAConfig(
        context_ratio=0.5,
        target_ratio=0.25,
        scale=2,
        momentum=0.98,
        predictor_depth=2,
        scheduled=False,
        gram_warmup_epochs=WARMUP_EPOCHS,
        gram_remove_neg=False,
        gram_loss_weight=1.0,
        sigreg_loss_weight=0.0001,
    )


@pytest.fixture
def jepa_config_instant_gram():
    return JEPAConfig(
        context_ratio=0.5,
        target_ratio=0.25,
        scale=2,
        momentum=0.98,
        predictor_depth=2,
        scheduled=False,
        gram_warmup_epochs=0,
        gram_remove_neg=False,
        gram_loss_weight=1.0,
        sigreg_loss_weight=0.0001,
    )


@pytest.fixture
def small_vit_config():
    return ViTConfig(
        in_channels=3,
        hidden_size=HIDDEN_SIZE,
        patch_size=[PATCH_SIZE, PATCH_SIZE],
        img_size=[IMAGE_SIZE, IMAGE_SIZE],
        depth=2,
        num_attention_heads=4,
        ffn_hidden_size=128,
        activation="gelu",
        num_register_tokens=NUM_REGISTER_TOKENS,
        num_cls_tokens=NUM_CLS_TOKENS,
        pos_enc="rope",
        rope_base=100,
        dtype=torch.float32,
    )


@pytest.fixture
def small_vit(small_vit_config):
    return small_vit_config.instantiate()


@pytest.fixture
def predictor(small_vit):
    return CrossAttentionPredictor(small_vit, depth=2, out_dim=None)


@pytest.fixture
def mjepa_model(jepa_config, small_vit, predictor):
    return MJEPA(jepa_config, small_vit, predictor, dtype=torch.float32)


@pytest.fixture
def mjepa_model_instant_gram(jepa_config_instant_gram, small_vit, predictor):
    return MJEPA(jepa_config_instant_gram, small_vit, predictor, dtype=torch.float32)


@pytest.fixture
def dummy_batch():
    return torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def dummy_vit_features():
    dense_features = torch.randn(BATCH_SIZE, NUM_TOKENS + NUM_CLS_TOKENS + NUM_REGISTER_TOKENS, HIDDEN_SIZE)
    return ViTFeatures(dense_features, NUM_REGISTER_TOKENS, NUM_CLS_TOKENS)


class TestMJEPALosses:
    def test_reduce(self):
        losses = MJEPALosses(
            jepa_loss=torch.tensor(1.0),
            jepa_loss_cls=torch.tensor(0.5),
            sigreg_loss=torch.tensor(0.01),
            gram_loss=torch.tensor(0.1),
            gram_loss_weight=0.4,
            sigreg_loss_weight=1e-3,
        )

        reduced = losses.reduce()
        expected = torch.tensor(1.0) + torch.tensor(0.5) + torch.tensor(0.1) * 0.4 + torch.tensor(0.01) * 1e-3
        assert torch.allclose(reduced, expected)


class TestMJEPAPredictions:
    def test_initialization(self, dummy_vit_features):
        pred = torch.randn(BATCH_SIZE, NUM_TARGET_TOKENS, HIDDEN_SIZE)
        pred_with_cls = torch.randn(BATCH_SIZE, NUM_TARGET_TOKENS, HIDDEN_SIZE)
        context_mask = torch.randint(0, 2, (BATCH_SIZE, NUM_TOKENS), dtype=torch.bool)
        target_mask = torch.randint(0, 2, (BATCH_SIZE, NUM_TOKENS), dtype=torch.bool)
        gram_target_output = torch.randn(BATCH_SIZE, NUM_TOKENS, HIDDEN_SIZE)

        predictions = MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            student_output=dummy_vit_features,
            teacher_output=dummy_vit_features,
            context_mask=context_mask,
            target_mask=target_mask,
            gram_target_output=gram_target_output,
        )

        assert predictions.gram_target_output is not None
        assert torch.allclose(predictions.gram_target_output, gram_target_output)


class TestMJEPAInitialization:
    def test_initialization(
        self, mjepa_model: MJEPA, jepa_config: JEPAConfig, small_vit: ViT, predictor: CrossAttentionPredictor
    ):
        assert mjepa_model.config == jepa_config
        assert mjepa_model.student == small_vit
        assert mjepa_model.teacher is not None
        assert mjepa_model.predictor == predictor
        assert not hasattr(mjepa_model, "gram_teacher")

    def test_teacher_frozen(self, mjepa_model: MJEPA):
        for param in mjepa_model.teacher.parameters():
            assert not param.requires_grad


class TestMJEPAForwardPasses:
    def test_forward_teacher(self, mjepa_model: MJEPA, dummy_batch):
        output = mjepa_model.forward_teacher(dummy_batch)
        assert isinstance(output, ViTFeatures)
        assert output.dense_features.shape[0] == dummy_batch.shape[0]

    def test_forward_student(self, mjepa_model: MJEPA, dummy_batch):
        context_mask = torch.randint(0, 2, (BATCH_SIZE, NUM_TOKENS), dtype=torch.bool)
        output = mjepa_model.forward_student(dummy_batch, context_mask)
        assert isinstance(output, ViTFeatures)
        assert output.dense_features.shape[0] == dummy_batch.shape[0]

    def test_forward_predictor(self, mjepa_model: MJEPA):
        num_tokens = TOKENIZED_SIZE[0] * TOKENIZED_SIZE[1]
        context = torch.randn(BATCH_SIZE, NUM_CONTEXT_TOKENS, HIDDEN_SIZE)
        context_mask = torch.zeros(BATCH_SIZE, num_tokens, dtype=torch.bool)
        context_mask[:, :NUM_CONTEXT_TOKENS] = True
        target_mask = torch.zeros(BATCH_SIZE, num_tokens, dtype=torch.bool)
        target_mask[:, NUM_CONTEXT_TOKENS : NUM_CONTEXT_TOKENS + NUM_TARGET_TOKENS] = True

        output = mjepa_model.forward_predictor(TOKENIZED_SIZE, context, context_mask, target_mask)
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == BATCH_SIZE

    def test_forward_gram_target(self, mjepa_model: MJEPA, dummy_batch):
        context_mask = torch.randint(0, 2, (BATCH_SIZE, NUM_TOKENS), dtype=torch.bool)
        output = mjepa_model.forward_gram_target(dummy_batch, context_mask)
        expected = apply_mask(context_mask, mjepa_model.teacher.stem(dummy_batch), fill_value=None)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == dummy_batch.shape[0]
        assert torch.allclose(output, expected)
        assert not output.requires_grad


class TestMJEPAWarmup:
    @pytest.mark.parametrize(
        "epoch,expected",
        [
            (0, 0.0),
            (10, 0.5),
            (WARMUP_EPOCHS, 1.0),
            (40, 1.0),
        ],
    )
    def test_gram_loss_weight_for_epoch(self, mjepa_model: MJEPA, epoch: int, expected: float):
        assert mjepa_model.gram_loss_weight_for_epoch(epoch) == pytest.approx(expected)

    def test_gram_loss_weight_for_epoch_without_warmup(self, mjepa_model_instant_gram: MJEPA):
        assert mjepa_model_instant_gram.gram_loss_weight_for_epoch(0) == pytest.approx(1.0)
        assert mjepa_model_instant_gram.gram_loss_weight_for_epoch(100) == pytest.approx(1.0)


class TestMJEPAComputeLosses:
    def _build_predictions(self) -> MJEPAPredictions:
        pred = torch.randn(BATCH_SIZE, NUM_TARGET_TOKENS, HIDDEN_SIZE)
        pred_with_cls = torch.randn(BATCH_SIZE, NUM_TARGET_TOKENS, HIDDEN_SIZE)
        dense_features = torch.randn(BATCH_SIZE, NUM_TOKENS + NUM_CLS_TOKENS + NUM_REGISTER_TOKENS, HIDDEN_SIZE)
        student_output = ViTFeatures(dense_features, NUM_REGISTER_TOKENS, NUM_CLS_TOKENS)
        teacher_output = ViTFeatures(dense_features.clone(), NUM_REGISTER_TOKENS, NUM_CLS_TOKENS)

        target_mask = torch.zeros(BATCH_SIZE, NUM_TOKENS, dtype=torch.bool)
        target_mask[:, :NUM_TARGET_TOKENS] = True

        return MJEPAPredictions(
            pred=pred,
            pred_with_cls=pred_with_cls,
            student_output=student_output,
            teacher_output=teacher_output,
            target_mask=target_mask,
            context_mask=target_mask,
            gram_target_output=torch.randn(BATCH_SIZE, NUM_TOKENS, HIDDEN_SIZE),
        )

    def test_compute_losses_with_gram(self, mjepa_model: MJEPA):
        losses = mjepa_model.compute_losses(output=self._build_predictions(), step=0, epoch=WARMUP_EPOCHS)
        assert isinstance(losses.gram_loss, torch.Tensor)
        assert losses.gram_loss.item() > 0
        assert losses.gram_loss_weight == pytest.approx(1.0)

    def test_compute_losses_with_warmup_weight(self, mjepa_model: MJEPA):
        losses = mjepa_model.compute_losses(output=self._build_predictions(), step=0, epoch=10)
        assert isinstance(losses.gram_loss, torch.Tensor)
        assert losses.gram_loss_weight == pytest.approx(0.5)

    def test_compute_losses_starts_at_zero_weight(self, mjepa_model: MJEPA):
        losses = mjepa_model.compute_losses(output=self._build_predictions(), step=0, epoch=0)
        assert isinstance(losses.gram_loss, torch.Tensor)
        assert losses.gram_loss_weight == pytest.approx(0.0)


class TestMJEPAForward:
    def test_forward_includes_gram_target_at_all_epochs(self, mjepa_model: MJEPA, dummy_batch):
        mjepa_model.eval()

        with torch.no_grad():
            predictions_epoch_0 = mjepa_model(dummy_batch, jepa_scale=2, epoch=0)
            predictions_epoch_20 = mjepa_model(dummy_batch, jepa_scale=2, epoch=20)

        assert predictions_epoch_0.gram_target_output is not None
        assert predictions_epoch_20.gram_target_output is not None
        assert not predictions_epoch_0.gram_target_output.requires_grad

    def test_forward_training_mode(self, mjepa_model: MJEPA, dummy_batch):
        mjepa_model.train()
        predictions = mjepa_model(dummy_batch, jepa_scale=2, epoch=0)
        assert predictions.pred.requires_grad


class TestMJEPAEndToEnd:
    def test_full_training_step(self, mjepa_model: MJEPA, dummy_batch):
        mjepa_model.train()

        predictions = mjepa_model(dummy_batch, jepa_scale=2, epoch=0)
        losses = mjepa_model.compute_losses(output=predictions, step=0, epoch=0).reduce()
        losses.backward()

        for param in mjepa_model.student.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break

    def test_teacher_update_workflow(self, mjepa_model: MJEPA):
        initial_weight = mjepa_model.teacher.stem.patch.weight.data.clone()
        mjepa_model.student.stem.patch.weight.data.fill_(1.0)
        mjepa_model.update_teacher(step=0, total_steps=100)
        updated_weight = mjepa_model.teacher.stem.patch.weight.data
        assert not torch.allclose(updated_weight, initial_weight)
