import math
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import yaml
from vit import ViTConfig

from mjepa.jepa import (
    COSINE_JEPA_LOSS_KIND,
    MSE_JEPA_LOSS_KIND,
    PREDICTOR_PROJ_INIT_STD,
    CrossAttentionPredictor,
    JEPAConfig,
    autocast_context,
    compute_gram_loss,
    compute_jepa_prediction_loss,
    compute_sigreg_loss,
    config_constructor,
    generate_masks,
    get_momentum,
    is_gram_update_epoch,
    register_constructors,
    setup_teacher,
    update_teacher,
)


class TestUpdateTeacher:
    def test_update_teacher_with_momentum(self):
        """Test teacher update with non-zero momentum."""
        student = nn.Linear(10, 10)
        teacher = nn.Linear(10, 10)

        # Initialize with different weights
        student.weight.data.fill_(1.0)
        teacher.weight.data.fill_(0.0)

        momentum = 0.9
        update_teacher(student, teacher, momentum)

        # Weight should be updated with (1-momentum) * student_weight
        expected = 0.1  # (1-0.9) * 1.0
        assert torch.allclose(teacher.weight.data, torch.full_like(teacher.weight.data, expected))

    def test_update_teacher_momentum_one(self):
        """Test teacher update with momentum=1.0, should be no-op."""
        student = nn.Linear(10, 10)
        teacher = nn.Linear(10, 10)

        # Initialize with different weights
        student.weight.data.fill_(1.0)
        teacher.weight.data.fill_(0.0)

        momentum = 1.0
        update_teacher(student, teacher, momentum)

        # Teacher weights should remain unchanged
        assert torch.allclose(teacher.weight.data, torch.zeros_like(teacher.weight.data))

    def test_update_teacher_invalid_momentum(self):
        """Test that invalid momentum values raise an assertion error."""
        student = nn.Linear(10, 10)
        teacher = nn.Linear(10, 10)

        with pytest.raises(AssertionError):
            update_teacher(student, teacher, -0.1)

        with pytest.raises(AssertionError):
            update_teacher(student, teacher, 1.1)

    @pytest.mark.parametrize(
        "student_dtype,teacher_dtype",
        [
            (torch.float32, torch.bfloat16),
            (torch.bfloat16, torch.float32),
        ],
    )
    def test_update_teacher_supports_mixed_dtypes(self, student_dtype, teacher_dtype):
        """Test mixed-dtype EMA updates preserve the teacher dtype."""
        student = nn.Linear(4, 4, dtype=student_dtype)
        teacher = nn.Linear(4, 4, dtype=teacher_dtype)
        student.weight.data.fill_(2.0)
        teacher.weight.data.fill_(0.0)

        update_teacher(student, teacher, momentum=0.5)

        assert teacher.weight.dtype == teacher_dtype
        expected = torch.full_like(teacher.weight.data, 1.0)
        assert torch.allclose(teacher.weight.data, expected)


class TestGetMomentum:
    def test_get_momentum_start(self):
        """Test momentum at start of training."""
        assert get_momentum(0, 100, 0.9) == 0.9

    @pytest.mark.parametrize(
        "scheduled,expected",
        [
            (True, 0.95),
            (False, 0.9),
        ],
    )
    def test_get_momentum_middle(self, scheduled, expected):
        """Test momentum in middle of training."""
        assert get_momentum(50, 100, 0.9, scheduled) == expected

    @pytest.mark.parametrize(
        "scheduled,expected",
        [
            (True, 1.0),
            (False, 0.9),
        ],
    )
    def test_get_momentum_end(self, scheduled, expected):
        """Test momentum at end of training."""
        assert get_momentum(100, 100, 0.9, scheduled) == expected

    def test_get_momentum_interpolation(self):
        """Test momentum interpolation at arbitrary step."""
        acutal = get_momentum(25, 100, 0.8, scheduled=True)
        assert math.isclose(acutal, 0.85, rel_tol=1e-6)


class TestJEPAConfig:
    def test_default_config(self):
        """Test default configuration."""
        config = JEPAConfig()
        assert config.context_ratio == 0.5
        assert config.target_ratio == 0.25
        assert config.scale == 4
        assert config.momentum == 0.99
        assert config.predictor_depth == 4
        assert config.disable_predictor_regularizers is False
        assert config.teacher_dtype is None
        assert config.stem_jepa_loss_weight == 0.0
        assert config.jepa_loss_kind == MSE_JEPA_LOSS_KIND

    def test_custom_config(self):
        """Test custom configuration."""
        config = JEPAConfig(
            context_ratio=0.6,
            target_ratio=0.3,
            scale=8,
            momentum=0.95,
            predictor_depth=6,
            disable_predictor_regularizers=True,
            teacher_dtype=torch.bfloat16,
            stem_jepa_loss_weight=0.5,
            jepa_loss_kind=COSINE_JEPA_LOSS_KIND,
        )
        assert config.context_ratio == 0.6
        assert config.target_ratio == 0.3
        assert config.scale == 8
        assert config.momentum == 0.95
        assert config.predictor_depth == 6
        assert config.disable_predictor_regularizers is True
        assert config.teacher_dtype == torch.bfloat16
        assert config.stem_jepa_loss_weight == 0.5
        assert config.jepa_loss_kind == COSINE_JEPA_LOSS_KIND

    def test_invalid_context_ratio(self):
        """Test invalid context ratio."""
        with pytest.raises(ValueError):
            JEPAConfig(context_ratio=0)

        with pytest.raises(ValueError):
            JEPAConfig(context_ratio=1.1)

    def test_invalid_target_ratio(self):
        """Test invalid target ratio."""
        with pytest.raises(ValueError):
            JEPAConfig(target_ratio=0)

        with pytest.raises(ValueError):
            JEPAConfig(target_ratio=1.1)


class TestYAMLConfig:
    def test_config_constructor(self, mocker):
        """Test the config_constructor function."""
        # Create mock loader and node
        loader = mocker.Mock()
        node = mocker.Mock()

        # Configure the loader to return a dictionary
        config_dict = {
            "context_ratio": 0.7,
            "target_ratio": 0.4,
            "scale": 16,
            "momentum": 0.98,
            "predictor_depth": 8,
            "disable_predictor_regularizers": True,
            "teacher_dtype": "bfloat16",
            "stem_jepa_loss_weight": 0.25,
            "jepa_loss_kind": COSINE_JEPA_LOSS_KIND,
        }
        loader.construct_mapping.return_value = config_dict

        # Call the constructor
        config = config_constructor(loader, node)

        # Verify the returned config
        assert isinstance(config, JEPAConfig)
        assert config.context_ratio == 0.7
        assert config.target_ratio == 0.4
        assert config.scale == 16
        assert config.momentum == 0.98
        assert config.predictor_depth == 8
        assert config.disable_predictor_regularizers is True
        assert config.teacher_dtype == torch.bfloat16
        assert config.stem_jepa_loss_weight == 0.25
        assert config.jepa_loss_kind == COSINE_JEPA_LOSS_KIND

        # Verify the loader was called correctly
        loader.construct_mapping.assert_called_once_with(node, deep=True)

    def test_yaml_load(self):
        """Test loading a JEPAConfig from YAML."""
        # Register constructors
        register_constructors()

        # Create a YAML string
        yaml_str = """
        !!python/object:mjepa.JEPAConfig
        context_ratio: 0.8
        target_ratio: 0.3
        scale: 12
        momentum: 0.97
        predictor_depth: 5
        disable_predictor_regularizers: true
        teacher_dtype: bfloat16
        stem_jepa_loss_weight: 0.75
        jepa_loss_kind: cosine
        """

        # Load the YAML
        config = yaml.safe_load(yaml_str)

        # Verify the loaded config
        assert isinstance(config, JEPAConfig)
        assert config.context_ratio == 0.8
        assert config.target_ratio == 0.3
        assert config.scale == 12
        assert config.momentum == 0.97
        assert config.predictor_depth == 5
        assert config.disable_predictor_regularizers is True
        assert config.teacher_dtype == torch.bfloat16
        assert config.stem_jepa_loss_weight == 0.75
        assert config.jepa_loss_kind == COSINE_JEPA_LOSS_KIND

    def test_invalid_teacher_dtype_string(self):
        """Test invalid teacher dtype string."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            JEPAConfig(teacher_dtype="halfish")

    def test_invalid_jepa_loss_kind(self):
        """Test invalid JEPA reconstruction loss kind."""
        with pytest.raises(ValueError, match="jepa_loss_kind must be one of"):
            JEPAConfig(jepa_loss_kind=cast(Any, "l1"))

    def test_invalid_stem_jepa_loss_weight(self):
        """Test invalid stem JEPA loss weight."""
        with pytest.raises(ValueError, match="stem_jepa_loss_weight"):
            JEPAConfig(stem_jepa_loss_weight=-0.1)


class TestComputeSigREGLoss:
    @pytest.mark.parametrize(
        "x_shape",
        [
            (1, 32, 128),
            (2, 16, 128),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_compute_sigreg_loss(self, x_shape, dtype):
        """Test the compute_sigreg_loss function."""
        x = torch.randn(*x_shape, dtype=dtype)
        global_step = 0
        num_slices = 256
        loss = compute_sigreg_loss(x, global_step, num_slices)
        assert loss.shape == ()
        assert loss.item() > 0
        assert not torch.isnan(loss)

    @pytest.mark.parametrize(
        "x_shape",
        [
            (1, 32, 128),
            (2, 16, 128),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_compute_sigreg_loss_deterministic(self, x_shape, dtype):
        """Test the compute_sigreg_loss function."""
        x = torch.randn(*x_shape, dtype=dtype)
        global_step = 0
        num_slices = 256
        loss1 = compute_sigreg_loss(x, global_step, num_slices)
        loss2 = compute_sigreg_loss(x, global_step, num_slices)
        loss3 = compute_sigreg_loss(x, global_step + 1, num_slices)
        assert loss1 == loss2
        assert loss1 != loss3

    def test_compute_sigreg_loss_isotropic_gaussian(self):
        """Test that SigREG loss is lower for isotropic Gaussian embeddings."""
        B, L, D = 2, 32, 128
        global_step = 0
        num_slices = 256

        # Create isotropic Gaussian embeddings (should have lower loss)
        isotropic_x = torch.randn(B, L, D)

        # Create non-isotropic embeddings (concentrated in one direction)
        non_isotropic_x = torch.zeros(B, L, D)
        non_isotropic_x[:, :, 0] = torch.randn(B, L) * 10  # High variance in first dimension
        non_isotropic_x[:, :, 1:] = torch.randn(B, L, D - 1) * 0.1  # Low variance in other dimensions

        isotropic_loss = compute_sigreg_loss(isotropic_x, global_step, num_slices)
        non_isotropic_loss = compute_sigreg_loss(non_isotropic_x, global_step, num_slices)

        # Isotropic Gaussian should have lower SigREG loss
        assert isotropic_loss < non_isotropic_loss
        assert not torch.isnan(isotropic_loss)
        assert not torch.isnan(non_isotropic_loss)

    def test_compute_sigreg_loss_rejects_empty_token_axis(self):
        """Test empty token sequences raise a clear error instead of returning NaN."""
        empty_tokens = torch.randn(2, 0, 128)

        with pytest.raises(ValueError, match="at least one token"):
            compute_sigreg_loss(empty_tokens, global_step=0, num_slices=256)


class TestAutocastContext:
    @pytest.mark.parametrize(
        ("dtype", "enabled"),
        [
            (torch.float32, False),
            (torch.bfloat16, True),
            (torch.float16, True),
        ],
    )
    def test_autocast_context_uses_only_amp_safe_dtypes(self, mocker, dtype, enabled):
        sentinel = mocker.sentinel.autocast_context
        autocast = mocker.patch("mjepa.jepa.torch.autocast", return_value=sentinel)

        result = autocast_context("cuda", dtype)

        assert result is sentinel
        autocast.assert_called_once_with(device_type="cuda", dtype=dtype, enabled=enabled)


class TestCrossAttentionPredictor:
    """Test CrossAttentionPredictor initialization and device/dtype handling."""

    @staticmethod
    def _instantiate_backbone(
        dtype: torch.dtype = torch.float32,
        *,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ):
        vit_config = ViTConfig(
            in_channels=3,
            hidden_size=64,
            patch_size=[4, 4],
            img_size=[32, 32],
            depth=2,
            num_attention_heads=4,
            ffn_hidden_size=128,
            num_register_tokens=2,
            num_cls_tokens=2,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            drop_path_rate=drop_path_rate,
            pos_enc="rope",
            dtype=dtype,
        )
        return vit_config.instantiate()

    @staticmethod
    def _assert_predictor_device_dtype(
        predictor: CrossAttentionPredictor, device: torch.device, dtype: torch.dtype
    ) -> None:
        for name, param in predictor.named_parameters():
            if device.type == "cuda":
                assert param.device.type == device.type, f"Parameter {name} is on {param.device}, expected {device}"
            else:
                assert param.device == device, f"Parameter {name} is on {param.device}, expected {device}"
            assert param.dtype == dtype, f"Parameter {name} has dtype {param.dtype}, expected {dtype}"

    @staticmethod
    def _assert_predictor_regularizers(
        predictor: CrossAttentionPredictor, hidden_dropout: float, attention_dropout: float, drop_path_rate: float
    ) -> None:
        for block in predictor.blocks:
            block = cast(Any, block)
            assert block.drop_path_rate == pytest.approx(drop_path_rate)
            assert block.cross_attention.dropout.p == pytest.approx(hidden_dropout)
            assert block.cross_attention.attention_dropout.p == pytest.approx(attention_dropout)
            assert block.mlp.dropout.p == pytest.approx(hidden_dropout)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_predictor_initialization_cpu_dtype(self, dtype):
        """Test that CrossAttentionPredictor initializes all params with correct dtype on CPU."""
        backbone = self._instantiate_backbone(dtype)
        device = torch.device("cpu")
        predictor = CrossAttentionPredictor(backbone, depth=2, out_dim=None, device=device)
        self._assert_predictor_device_dtype(predictor, device, dtype)

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_predictor_initialization_cuda_dtype(self, dtype):
        """Test that CrossAttentionPredictor initializes all params with correct dtype on CUDA."""
        backbone = self._instantiate_backbone(dtype)
        device = torch.device("cuda")
        predictor = CrossAttentionPredictor(backbone, depth=2, out_dim=None, device=device)
        self._assert_predictor_device_dtype(predictor, device, dtype)

    def test_predictor_all_params_share_backbone_dtype(self):
        """Test that all predictor parameters share the same dtype as backbone.config.dtype."""
        backbone_dtype = torch.bfloat16
        backbone = self._instantiate_backbone(backbone_dtype)
        device = torch.device("cpu")
        predictor = CrossAttentionPredictor(backbone, depth=2, out_dim=128, device=device)

        # Verify backbone config dtype
        assert backbone.config.dtype == backbone_dtype

        # Check that all parameters match backbone.config.dtype
        param_count = 0
        for name, param in predictor.named_parameters():
            param_count += 1
            assert param.dtype == backbone.config.dtype, (
                f"Parameter {name} has dtype {param.dtype}, expected {backbone.config.dtype}"
            )

        # Ensure we actually checked some parameters
        assert param_count > 0, "No parameters found in predictor"

    def test_predictor_proj_uses_vit_style_initialization(self):
        """Test predictor output head uses the configured small-weight initialization."""
        backbone = self._instantiate_backbone()
        predictor = CrossAttentionPredictor(backbone, depth=2, out_dim=128, device=torch.device("cpu"))

        assert torch.count_nonzero(predictor.predictor_proj.bias) == 0
        assert predictor.predictor_proj.weight.std().item() == pytest.approx(PREDICTOR_PROJ_INIT_STD, rel=0.25)

    @pytest.mark.parametrize("predictor_depth", [1, 2, 4])
    def test_predictor_initialization_different_depths(self, predictor_depth):
        """Test that CrossAttentionPredictor with different depths initializes correctly."""
        dtype = torch.float32
        backbone = self._instantiate_backbone(dtype)
        device = torch.device("cpu")
        predictor = CrossAttentionPredictor(backbone, depth=predictor_depth, out_dim=None, device=device)
        self._assert_predictor_device_dtype(predictor, device, dtype)

        # Check that the number of blocks matches the depth
        assert len(predictor.blocks) == predictor_depth

    @pytest.mark.parametrize("out_dim", [None, 64, 128, 256])
    def test_predictor_initialization_different_out_dims(self, out_dim):
        """Test that CrossAttentionPredictor with different output dimensions initializes correctly."""
        dtype = torch.float32
        hidden_size = 64
        backbone = self._instantiate_backbone(dtype)
        device = torch.device("cpu")
        predictor = CrossAttentionPredictor(backbone, depth=2, out_dim=out_dim, device=device)
        self._assert_predictor_device_dtype(predictor, device, dtype)

        # Check the output projection dimension
        expected_out_dim = out_dim if out_dim is not None else hidden_size
        assert predictor.predictor_proj.out_features == expected_out_dim

    def test_predictor_regularizers_follow_backbone_by_default(self):
        """Test predictor regularizer values are inherited from the backbone by default."""
        hidden_dropout = 0.2
        attention_dropout = 0.3
        drop_path_rate = 0.4
        backbone = self._instantiate_backbone(
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            drop_path_rate=drop_path_rate,
        )
        predictor = CrossAttentionPredictor(backbone, depth=2, disable_predictor_regularizers=False)

        self._assert_predictor_regularizers(predictor, hidden_dropout, attention_dropout, drop_path_rate)

    def test_predictor_regularizers_can_be_disabled(self):
        """Test predictor regularizers are set to zero when override is enabled."""
        hidden_dropout = 0.2
        attention_dropout = 0.3
        drop_path_rate = 0.4
        backbone = self._instantiate_backbone(
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            drop_path_rate=drop_path_rate,
        )
        predictor = CrossAttentionPredictor(backbone, depth=2, disable_predictor_regularizers=True)

        self._assert_predictor_regularizers(predictor, 0.0, 0.0, 0.0)

    def test_predictor_positional_device_arg_is_backwards_compatible(self):
        """Test positional device argument does not toggle regularizer override."""
        hidden_dropout = 0.2
        attention_dropout = 0.3
        drop_path_rate = 0.4
        backbone = self._instantiate_backbone(
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            drop_path_rate=drop_path_rate,
        )
        device = torch.device("cpu")

        predictor = CrossAttentionPredictor(backbone, 2, None, device)

        self._assert_predictor_regularizers(predictor, hidden_dropout, attention_dropout, drop_path_rate)

    def test_predictor_forward_promotes_output_to_float32(self):
        """Test predictor outputs are promoted to float32."""
        vit_config = ViTConfig(
            in_channels=3,
            hidden_size=64,
            patch_size=[4, 4],
            img_size=[32, 32],
            depth=2,
            num_attention_heads=4,
            ffn_hidden_size=128,
            num_register_tokens=2,
            num_cls_tokens=2,
            pos_enc="rope",
            dtype=torch.bfloat16,
        )
        backbone = vit_config.instantiate()
        predictor = CrossAttentionPredictor(backbone, depth=2)
        context = torch.randn(2, 32, 64, dtype=torch.bfloat16)
        context_mask = torch.zeros(2, 64, dtype=torch.bool)
        context_mask[:, :32] = True
        target_mask = torch.zeros(2, 64, dtype=torch.bool)
        target_mask[:, 32:48] = True

        output = predictor((8, 8), context, context_mask, target_mask)

        assert output.dtype == torch.float32

    def test_predictor_forward_heads_disable_outer_autocast_for_fp32_projections(self):
        """Test projection heads do not inherit a lower-precision outer autocast context."""
        backbone = self._instantiate_backbone(torch.float32)
        predictor = CrossAttentionPredictor(backbone, depth=2)
        predictor.enable_shallow_head()

        context = torch.randn(2, 32, 64)
        context_mask = torch.zeros(2, 64, dtype=torch.bool)
        context_mask[:, :32] = True
        target_mask = torch.zeros(2, 64, dtype=torch.bool)
        target_mask[:, 32:48] = True

        autocast_enabled: list[bool] = []

        def capture_autocast(_module, _inputs):
            autocast_enabled.append(torch.is_autocast_enabled("cpu"))

        hook = predictor.predictor_proj.register_forward_pre_hook(capture_autocast)
        try:
            with autocast_context("cpu", torch.bfloat16):
                predictor.forward_heads((8, 8), context, context_mask, target_mask)
        finally:
            hook.remove()

        assert autocast_enabled == [False]

    def test_predictor_can_enable_and_emit_shallow_head(self):
        """Test predictor can emit both deep and shallow targets from the shared trunk."""
        backbone = self._instantiate_backbone(torch.float32)
        predictor = CrossAttentionPredictor(backbone, depth=2)
        predictor.enable_shallow_head()

        context = torch.randn(2, 32, 64)
        context_mask = torch.zeros(2, 64, dtype=torch.bool)
        context_mask[:, :32] = True
        target_mask = torch.zeros(2, 64, dtype=torch.bool)
        target_mask[:, 32:48] = True

        deep_output, shallow_output = predictor.forward_heads((8, 8), context, context_mask, target_mask)

        assert deep_output.dtype == torch.float32
        assert shallow_output is not None
        assert shallow_output.dtype == torch.float32
        assert shallow_output.shape == deep_output.shape


class TestGenerateMasks:
    """Test generate_masks function."""

    @pytest.fixture
    def backbone(self):
        """Create a backbone for testing."""
        vit_config = ViTConfig(
            in_channels=3,
            hidden_size=64,
            patch_size=[4, 4],
            img_size=[32, 32],
            depth=2,
            num_attention_heads=4,
            ffn_hidden_size=128,
        )
        return vit_config.instantiate()

    def test_masks_are_boolean(self, backbone):
        """Test that generated masks are boolean tensors."""
        x = torch.randn(2, 3, 32, 32)
        context_mask, target_mask = generate_masks(backbone, x, context_ratio=0.5, target_ratio=0.25, scale=2)

        assert context_mask.dtype == torch.bool
        assert target_mask.dtype == torch.bool

    def test_masks_have_correct_shape(self, backbone):
        """Test that masks have correct batch and token dimensions."""
        x = torch.randn(2, 3, 32, 32)
        context_mask, target_mask = generate_masks(backbone, x, context_ratio=0.5, target_ratio=0.25, scale=2)

        # Masks should have shape (batch, num_tokens)
        batch_size = x.shape[0]
        tokenized_size = backbone.stem.tokenized_size(x.shape[-2:])
        num_tokens = tokenized_size[0] * tokenized_size[1]

        assert context_mask.shape == (batch_size, num_tokens)
        assert target_mask.shape == (batch_size, num_tokens)

    def test_masks_non_overlapping(self, backbone):
        """Test that context and target masks do not overlap."""
        x = torch.randn(2, 3, 32, 32)
        context_mask, target_mask = generate_masks(backbone, x, context_ratio=0.5, target_ratio=0.25, scale=2)

        # No position should be True in both masks
        overlap = context_mask & target_mask
        assert not overlap.any(), "Context and target masks should not overlap"

    def test_target_ratio_one_targets_all_tokens(self, backbone):
        """Test that a full target ratio marks every visual token as a target."""
        x = torch.randn(2, 3, 32, 32)
        context_mask, target_mask = generate_masks(backbone, x, context_ratio=0.5, target_ratio=1.0, scale=2)

        assert target_mask.all()
        assert (context_mask & target_mask).any(), "Full target masks should include context tokens"

    def test_context_ratio_approximate(self, backbone):
        """Test that context mask has approximately the expected ratio of True values."""
        x = torch.randn(4, 3, 32, 32)
        context_ratio = 0.5
        context_mask, _ = generate_masks(backbone, x, context_ratio=context_ratio, target_ratio=0.25, scale=2)

        actual_ratio = context_mask.float().mean().item()
        # Allow some tolerance due to discrete masking
        assert abs(actual_ratio - context_ratio) < 0.2

    def test_different_scales(self, backbone):
        """Test generate_masks with different scale values."""
        x = torch.randn(2, 3, 32, 32)

        for scale in [1, 2, 4]:
            context_mask, target_mask = generate_masks(backbone, x, context_ratio=0.5, target_ratio=0.25, scale=scale)

            # Masks should still have correct shapes
            assert context_mask.shape == target_mask.shape


class TestComputeGramLoss:
    """Test compute_gram_loss function."""

    def test_basic_gram_loss(self):
        """Test basic Gram loss computation."""
        student = torch.randn(2, 16, 64)
        teacher = torch.randn(2, 16, 64)

        loss = compute_gram_loss(student, teacher)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_gram_loss_zero_for_identical_inputs(self):
        """Test Gram loss is zero when student equals teacher."""
        features = torch.randn(2, 16, 64)

        loss = compute_gram_loss(features, features.clone())

        assert loss.item() < 1e-5, "Loss should be near zero for identical inputs"

    def test_gram_loss_normalize_option(self):
        """Test Gram loss with and without normalization."""
        student = torch.randn(2, 16, 64)
        teacher = torch.randn(2, 16, 64)

        loss_normalized = compute_gram_loss(student, teacher, normalize=True)
        loss_unnormalized = compute_gram_loss(student, teacher, normalize=False)

        # Both should be valid losses
        assert not torch.isnan(loss_normalized)
        assert not torch.isnan(loss_unnormalized)

    def test_gram_loss_remove_neg_option(self):
        """Test Gram loss with and without negative removal."""
        student = torch.randn(2, 16, 64)
        teacher = torch.randn(2, 16, 64)

        loss_remove_neg = compute_gram_loss(student, teacher, remove_neg=True)
        loss_keep_neg = compute_gram_loss(student, teacher, remove_neg=False)

        # Both should be valid losses
        assert not torch.isnan(loss_remove_neg)
        assert not torch.isnan(loss_keep_neg)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_gram_loss_different_dtypes(self, dtype):
        """Test Gram loss with different data types."""
        student = torch.randn(2, 16, 64, dtype=dtype)
        teacher = torch.randn(2, 16, 64, dtype=dtype)

        loss = compute_gram_loss(student, teacher)

        assert not torch.isnan(loss)

    def test_gram_loss_different_batch_sizes(self):
        """Test Gram loss with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            student = torch.randn(batch_size, 16, 64)
            teacher = torch.randn(batch_size, 16, 64)

            loss = compute_gram_loss(student, teacher)

            assert loss.shape == ()
            assert not torch.isnan(loss)


class TestComputeJEPAPredictionLoss:
    """Test compute_jepa_prediction_loss function."""

    def test_mse_matches_torch_mse_loss(self):
        """Test MSE mode delegates to torch MSE loss."""
        student = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]])
        teacher = torch.tensor([[[2.0, 1.0], [0.0, 3.0]]])

        loss = compute_jepa_prediction_loss(student, teacher, kind=MSE_JEPA_LOSS_KIND)

        assert torch.isclose(loss, torch.nn.functional.mse_loss(student, teacher))

    def test_cosine_matches_manual_distance(self):
        """Test cosine mode computes mean one-minus-cosine distance."""
        student = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        teacher = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])

        loss = compute_jepa_prediction_loss(student, teacher, kind=COSINE_JEPA_LOSS_KIND)
        expected = (1.0 - torch.nn.functional.cosine_similarity(student, teacher, dim=-1)).mean()

        assert torch.isclose(loss, expected)

    def test_cosine_is_zero_for_identical_nonzero_inputs(self):
        """Test cosine mode is near zero for identical nonzero inputs."""
        features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

        loss = compute_jepa_prediction_loss(features, features.clone(), kind=COSINE_JEPA_LOSS_KIND)

        assert loss.item() < 1e-6


class TestSetupTeacher:
    """Test setup_teacher function."""

    def test_teacher_is_deepcopy(self):
        """Test that teacher is a deep copy of backbone."""
        backbone = nn.Linear(10, 10)
        backbone.weight.data.fill_(1.0)

        teacher = setup_teacher(backbone)

        # Modify backbone and verify teacher is unchanged
        backbone.weight.data.fill_(2.0)
        assert torch.allclose(teacher.weight.data, torch.ones_like(teacher.weight.data))

    def test_teacher_has_frozen_gradients(self):
        """Test that teacher parameters do not require gradients."""
        backbone = nn.Linear(10, 10)

        teacher = setup_teacher(backbone)

        for param in teacher.parameters():
            assert not param.requires_grad

    def test_teacher_is_in_eval_mode(self):
        """Test that teacher is in evaluation mode."""
        backbone = nn.Sequential(nn.Linear(10, 10), nn.BatchNorm1d(10))

        teacher = setup_teacher(backbone)

        assert not teacher.training

    def test_setup_teacher_preserves_weights(self):
        """Test that teacher has same weights as backbone."""
        backbone = nn.Linear(10, 10)
        backbone_weights = backbone.weight.data.clone()

        teacher = setup_teacher(backbone)

        assert torch.allclose(teacher.weight.data, backbone_weights)

    def test_setup_teacher_nested_module(self):
        """Test setup_teacher with nested modules."""
        backbone = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

        teacher = setup_teacher(backbone)

        assert not teacher.training
        for param in teacher.parameters():
            assert not param.requires_grad

    def test_setup_teacher_casts_dtype_when_requested(self):
        """Test setup_teacher casts teacher weights when requested."""
        backbone = nn.Linear(10, 10, dtype=torch.float32)

        teacher = setup_teacher(backbone, dtype=torch.bfloat16)

        for param in teacher.parameters():
            assert param.dtype == torch.bfloat16

    def test_setup_teacher_updates_vit_config_dtype(self):
        """Test setup_teacher keeps ViT config dtype in sync with cast weights."""
        vit_config = ViTConfig(
            in_channels=3,
            hidden_size=64,
            patch_size=[4, 4],
            img_size=[32, 32],
            depth=2,
            num_attention_heads=4,
            ffn_hidden_size=128,
            pos_enc="rope",
            dtype=torch.float32,
        )
        backbone = vit_config.instantiate()

        teacher = setup_teacher(backbone, dtype=torch.bfloat16)

        assert teacher.config.dtype == torch.bfloat16
        assert teacher.stem.patch.weight.dtype == torch.bfloat16


class TestIsGramUpdateEpoch:
    """Test is_gram_update_epoch function."""

    def test_returns_false_when_gram_start_is_none(self):
        """Test that function returns False when gram_start_epoch is None."""
        assert not is_gram_update_epoch(epoch=50, gram_start_epoch=None, gram_update_interval_epoch=10)

    def test_returns_false_before_gram_start(self):
        """Test that function returns False before gram_start_epoch."""
        assert not is_gram_update_epoch(epoch=5, gram_start_epoch=10, gram_update_interval_epoch=5)

    def test_returns_false_at_gram_start(self):
        """Test that function returns False at gram_start_epoch."""
        # epoch > gram_start_epoch is required, so at gram_start it's False
        assert not is_gram_update_epoch(epoch=10, gram_start_epoch=10, gram_update_interval_epoch=5)

    def test_returns_true_at_first_update_after_start(self):
        """Test that function returns True at first update epoch after gram_start."""
        # First update epoch is gram_start + gram_update_interval
        assert is_gram_update_epoch(epoch=15, gram_start_epoch=10, gram_update_interval_epoch=5)

    def test_returns_true_at_subsequent_update_epochs(self):
        """Test that function returns True at subsequent update epochs."""
        # Epochs 20, 25, 30, etc. should be update epochs
        assert is_gram_update_epoch(epoch=20, gram_start_epoch=10, gram_update_interval_epoch=5)
        assert is_gram_update_epoch(epoch=25, gram_start_epoch=10, gram_update_interval_epoch=5)
        assert is_gram_update_epoch(epoch=30, gram_start_epoch=10, gram_update_interval_epoch=5)

    def test_returns_false_between_update_epochs(self):
        """Test that function returns False between update epochs."""
        assert not is_gram_update_epoch(epoch=12, gram_start_epoch=10, gram_update_interval_epoch=5)
        assert not is_gram_update_epoch(epoch=17, gram_start_epoch=10, gram_update_interval_epoch=5)

    def test_returns_false_when_interval_is_zero(self):
        """Test that interval zero disables periodic updates."""
        assert not is_gram_update_epoch(epoch=11, gram_start_epoch=10, gram_update_interval_epoch=0)
        assert not is_gram_update_epoch(epoch=20, gram_start_epoch=10, gram_update_interval_epoch=0)

    @pytest.mark.parametrize("interval", [1, 2, 5, 10])
    def test_different_intervals(self, interval):
        """Test with different gram_update_interval_epoch values."""
        gram_start = 10

        # First update should be at gram_start + interval
        assert is_gram_update_epoch(
            epoch=gram_start + interval,
            gram_start_epoch=gram_start,
            gram_update_interval_epoch=interval,
        )


class TestJEPAConfigValidation:
    """Additional validation tests for JEPAConfig."""

    def test_invalid_gram_teacher_epoch(self):
        """Test invalid gram_teacher_epoch."""
        with pytest.raises(ValueError):
            JEPAConfig(gram_teacher_epoch=0)

        with pytest.raises(ValueError):
            JEPAConfig(gram_teacher_epoch=-1)

    def test_invalid_gram_start_epoch(self):
        """Test invalid gram_start_epoch."""
        with pytest.raises(ValueError):
            JEPAConfig(gram_start_epoch=0)

        with pytest.raises(ValueError):
            JEPAConfig(gram_start_epoch=-1)

    def test_gram_start_before_teacher_epoch(self):
        """Test that gram_start_epoch must be >= gram_teacher_epoch."""
        with pytest.raises(ValueError):
            JEPAConfig(gram_teacher_epoch=100, gram_start_epoch=50)

    def test_invalid_gram_update_interval(self):
        """Test invalid gram_update_interval_epoch."""
        with pytest.raises(ValueError):
            JEPAConfig(gram_update_interval_epoch=-1)

    def test_invalid_gram_resolution_scale(self):
        """Test invalid gram_resolution_scale."""
        with pytest.raises(ValueError):
            JEPAConfig(gram_resolution_scale=0)

        with pytest.raises(ValueError):
            JEPAConfig(gram_resolution_scale=-0.5)

    def test_invalid_gram_loss_weight(self):
        """Test invalid gram_loss_weight."""
        with pytest.raises(ValueError):
            JEPAConfig(gram_loss_weight=0)

        with pytest.raises(ValueError):
            JEPAConfig(gram_loss_weight=-0.1)

    def test_invalid_sigreg_loss_weight(self):
        """Test invalid sigreg_loss_weight."""
        with pytest.raises(ValueError):
            JEPAConfig(sigreg_loss_weight=-0.1)

    def test_valid_gram_config(self):
        """Test valid Gram configuration."""
        config = JEPAConfig(
            gram_teacher_epoch=50,
            gram_start_epoch=100,
            gram_update_interval_epoch=10,
            gram_resolution_scale=2.0,
            gram_remove_neg=True,
            gram_loss_weight=0.5,
            sigreg_loss_weight=1e-3,
        )
        assert config.gram_teacher_epoch == 50
        assert config.gram_start_epoch == 100
