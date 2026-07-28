import math
from copy import deepcopy
from dataclasses import dataclass, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from vit import ViT
from vit.pos_enc import LearnablePosition
from vit.tokens import apply_mask, generate_non_overlapping_mask
from vit.transformer import CrossAttentionTransformer


torch._dynamo.config.cache_size_limit = 1024 * 1024 * 1024 * 1024

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}
PREDICTOR_PROJ_INIT_STD = 0.02
_AUTOCAST_DTYPES = {torch.bfloat16, torch.float16}
MSE_JEPA_LOSS_KIND = "mse"
COSINE_JEPA_LOSS_KIND = "cosine"
JEPALossKind: TypeAlias = Literal["mse", "cosine"]
JEPA_LOSS_KINDS: tuple[JEPALossKind, ...] = (MSE_JEPA_LOSS_KIND, COSINE_JEPA_LOSS_KIND)
CROSS_ATTENTION_PREDICTOR_MODE = "cross_attention"
DECODER_PREDICTOR_MODE = "decoder"
ENCODER_PREDICTOR_MODE = "encoder"
PredictorAttentionMode: TypeAlias = Literal["cross_attention", "decoder", "encoder"]
PREDICTOR_ATTENTION_MODES: tuple[PredictorAttentionMode, ...] = (
    CROSS_ATTENTION_PREDICTOR_MODE,
    DECODER_PREDICTOR_MODE,
    ENCODER_PREDICTOR_MODE,
)
LEGACY_CLS_PREDICTION_MODE = "legacy_cross_attention"
ADALN_BLIND_CLS_PREDICTION_MODE = "adaln_blind"
ADALN_SHARED_CLS_PREDICTION_MODE = "adaln_shared"
PROJECTED_CLS_PREDICTION_MODE = "projected_cross_attention"
SLOT_BIAS_CLS_PREDICTION_MODE = "slot_bias_cross_attention"
RESIDUAL_PROJECTED_CLS_PREDICTION_MODE = "residual_projected_cross_attention"
RESIDUAL_MLP_CLS_PREDICTION_MODE = "residual_mlp_cross_attention"
PARTITIONED_SHARED_CLS_PREDICTION_MODE = "partitioned_shared_cross_attention"
PARTITIONED_INDEPENDENT_CLS_PREDICTION_MODE = "partitioned_independent_cross_attention"
JOINT_CONTEXT_CLS_PREDICTION_MODE = "joint_context"
SAMPLE_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE = "joint_context_sample_routed"
TOKEN_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE = "joint_context_token_routed"
DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE = "joint_context_dual_routed"
PACKED_DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE = "joint_context_packed_dual_routed"
SOURCE_BALANCED_TOKEN_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE = "joint_context_token_routed_source_balanced"
CLS_CONTEXT_TOKENS = 4
PROJECTED_CLS_CONTEXT_TOKENS = CLS_CONTEXT_TOKENS
JOINT_CONTEXT_ROUTE_PATTERN = (0, 0, 1, 2)
DUAL_ROUTE_PATTERN = (1, 2)
JOINT_CONTEXT_JOINT_ROUTE = 0
JOINT_CONTEXT_CLS_ONLY_ROUTE = 1
JOINT_CONTEXT_VISUAL_ONLY_ROUTE = 2
CLSPredictionMode: TypeAlias = Literal[
    "legacy_cross_attention",
    "adaln_blind",
    "adaln_shared",
    "projected_cross_attention",
    "slot_bias_cross_attention",
    "residual_projected_cross_attention",
    "residual_mlp_cross_attention",
    "partitioned_shared_cross_attention",
    "partitioned_independent_cross_attention",
    "joint_context",
    "joint_context_sample_routed",
    "joint_context_token_routed",
    "joint_context_dual_routed",
    "joint_context_packed_dual_routed",
    "joint_context_token_routed_source_balanced",
]
CLS_PREDICTION_MODES: tuple[CLSPredictionMode, ...] = (
    LEGACY_CLS_PREDICTION_MODE,
    ADALN_BLIND_CLS_PREDICTION_MODE,
    ADALN_SHARED_CLS_PREDICTION_MODE,
    PROJECTED_CLS_PREDICTION_MODE,
    SLOT_BIAS_CLS_PREDICTION_MODE,
    RESIDUAL_PROJECTED_CLS_PREDICTION_MODE,
    RESIDUAL_MLP_CLS_PREDICTION_MODE,
    PARTITIONED_SHARED_CLS_PREDICTION_MODE,
    PARTITIONED_INDEPENDENT_CLS_PREDICTION_MODE,
    JOINT_CONTEXT_CLS_PREDICTION_MODE,
    SAMPLE_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    TOKEN_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    PACKED_DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    SOURCE_BALANCED_TOKEN_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
)
ADALN_CLS_PREDICTION_MODES = (ADALN_BLIND_CLS_PREDICTION_MODE, ADALN_SHARED_CLS_PREDICTION_MODE)
PROJECTED_CLS_PREDICTION_MODES = (
    PROJECTED_CLS_PREDICTION_MODE,
    RESIDUAL_PROJECTED_CLS_PREDICTION_MODE,
)
PARTITIONED_CLS_PREDICTION_MODES = (
    PARTITIONED_SHARED_CLS_PREDICTION_MODE,
    PARTITIONED_INDEPENDENT_CLS_PREDICTION_MODE,
)
EXPANDED_CLS_PREDICTION_MODES = (
    *PROJECTED_CLS_PREDICTION_MODES,
    *PARTITIONED_CLS_PREDICTION_MODES,
    SLOT_BIAS_CLS_PREDICTION_MODE,
    RESIDUAL_MLP_CLS_PREDICTION_MODE,
)
JOINT_CONTEXT_CLS_PREDICTION_MODES = (
    JOINT_CONTEXT_CLS_PREDICTION_MODE,
    SAMPLE_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    TOKEN_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    PACKED_DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    SOURCE_BALANCED_TOKEN_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
)
ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODES = (
    SAMPLE_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    TOKEN_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    PACKED_DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    SOURCE_BALANCED_TOKEN_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
)
DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODES = (
    DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
    PACKED_DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE,
)
SINGLE_CLS_PREDICTION_MODES = (
    *ADALN_CLS_PREDICTION_MODES,
    *EXPANDED_CLS_PREDICTION_MODES,
    *JOINT_CONTEXT_CLS_PREDICTION_MODES,
)


def _validate_cls_context_tokens(cls_prediction_mode: CLSPredictionMode, cls_context_tokens: int) -> None:
    if isinstance(cls_context_tokens, bool) or not isinstance(cls_context_tokens, int) or cls_context_tokens <= 0:
        raise ValueError("cls_context_tokens must be a positive integer")
    if cls_prediction_mode not in PARTITIONED_CLS_PREDICTION_MODES and cls_context_tokens != CLS_CONTEXT_TOKENS:
        raise ValueError("cls_context_tokens is only configurable for partitioned CLS prediction modes")


def _balanced_routes(route_count: int, pattern_values: tuple[int, ...], device: torch.device) -> Tensor:
    pattern = torch.tensor(pattern_values, device=device)
    repeats = math.ceil(route_count / len(pattern_values))
    routes = pattern.repeat(repeats)[:route_count]
    return routes[torch.randperm(route_count, device=device)]


def joint_context_query_multiplier(cls_prediction_mode: CLSPredictionMode) -> int:
    """Return the number of predictor queries executed per spatial target query."""
    return 2 if cls_prediction_mode == PACKED_DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE else 1


def create_joint_context_attention_mask(
    cls_prediction_mode: CLSPredictionMode,
    batch_size: int,
    target_tokens: int,
    visual_context_tokens: int,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> Tensor | None:
    """Create balanced key-visibility masks for one visual-context plus CLS predictor pass."""
    if cls_prediction_mode == JOINT_CONTEXT_CLS_PREDICTION_MODE:
        return None
    if cls_prediction_mode not in ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODES:
        raise ValueError("joint context attention masks require a routed joint-context CLS prediction mode")
    if batch_size <= 0 or target_tokens <= 0 or visual_context_tokens <= 0:
        raise ValueError("joint context attention mask dimensions must be positive")

    if cls_prediction_mode == SAMPLE_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE:
        routes = _balanced_routes(batch_size, JOINT_CONTEXT_ROUTE_PATTERN, device)
        routes = routes[:, None].expand(-1, target_tokens)
    elif cls_prediction_mode == PACKED_DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE:
        visual_routes = torch.full(
            (batch_size, target_tokens),
            JOINT_CONTEXT_VISUAL_ONLY_ROUTE,
            device=device,
        )
        cls_routes = torch.full(
            (batch_size, target_tokens),
            JOINT_CONTEXT_CLS_ONLY_ROUTE,
            device=device,
        )
        routes = torch.cat([visual_routes, cls_routes], dim=1)
    else:
        route_pattern = (
            DUAL_ROUTE_PATTERN
            if cls_prediction_mode == DUAL_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE
            else JOINT_CONTEXT_ROUTE_PATTERN
        )
        base_routes = _balanced_routes(target_tokens, route_pattern, device)
        permutations = torch.rand(batch_size, target_tokens, device=device).argsort(dim=1)
        routes = base_routes[None].expand(batch_size, -1).gather(1, permutations)

    if cls_prediction_mode == SOURCE_BALANCED_TOKEN_ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODE:
        attention_bias = torch.zeros(
            batch_size,
            routes.shape[1],
            visual_context_tokens + 1,
            dtype=dtype or torch.float32,
            device=device,
        )
        cls_only = routes == JOINT_CONTEXT_CLS_ONLY_ROUTE
        visual_only = routes == JOINT_CONTEXT_VISUAL_ONLY_ROUTE
        joint = routes == JOINT_CONTEXT_JOINT_ROUTE
        attention_bias[..., :-1].masked_fill_(cls_only[..., None], -torch.inf)
        attention_bias[..., -1].masked_fill_(visual_only, -torch.inf)
        attention_bias[..., -1].masked_fill_(joint, math.log(visual_context_tokens))
        return attention_bias[:, None]

    attention_mask = torch.ones(
        batch_size,
        routes.shape[1],
        visual_context_tokens + 1,
        dtype=torch.bool,
        device=device,
    )
    cls_only = routes == JOINT_CONTEXT_CLS_ONLY_ROUTE
    visual_only = routes == JOINT_CONTEXT_VISUAL_ONLY_ROUTE
    attention_mask[..., :-1].masked_fill_(cls_only[..., None], False)
    attention_mask[..., -1].masked_fill_(visual_only, False)
    return attention_mask[:, None]


def _normalize_dtype(dtype: torch.dtype | str | None, field_name: str) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        normalized = _DTYPE_MAP.get(dtype.removeprefix("torch."))
        if normalized is not None:
            return normalized
    raise ValueError(f"Unsupported dtype for {field_name}: {dtype!r}")


def _sync_module_config_dtype(model: nn.Module, dtype: torch.dtype) -> None:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "dtype"):
        return

    if is_dataclass(config):
        updated_config = replace(cast(Any, config), dtype=dtype)
        config_attr = "_config" if hasattr(model, "_config") else "config"
        setattr(model, config_attr, updated_config)
        return

    setattr(config, "dtype", dtype)


def autocast_context(device_type: str, dtype: torch.dtype):
    enabled = dtype in _AUTOCAST_DTYPES
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)


class PartitionedCLSProjection(nn.Module):
    """Lift disjoint channel groups from one CLS embedding into predictor context tokens."""

    def __init__(
        self,
        hidden_size: int,
        *,
        shared: bool,
        num_context_tokens: int = CLS_CONTEXT_TOKENS,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if isinstance(num_context_tokens, bool) or not isinstance(num_context_tokens, int) or num_context_tokens <= 0:
            raise ValueError("num_context_tokens must be a positive integer")
        if hidden_size % num_context_tokens != 0:
            raise ValueError("hidden_size must be divisible by cls_context_tokens")
        self.hidden_size = hidden_size
        self.num_context_tokens = num_context_tokens
        self.chunk_size = hidden_size // num_context_tokens
        self.shared = shared
        projection_count = 1 if shared else num_context_tokens
        self.weight = nn.Parameter(
            torch.empty(projection_count, hidden_size, self.chunk_size, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.empty(num_context_tokens, hidden_size, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, std=PREDICTOR_PROJ_INIT_STD)
        nn.init.zeros_(self.bias)

    def forward(self, cls_embedding: Tensor) -> Tensor:
        if cls_embedding.ndim != 2:
            raise ValueError("cls_embedding must have shape [batch, hidden_size]")
        expected_shape = (cls_embedding.shape[0], self.hidden_size)
        if tuple(cls_embedding.shape) != expected_shape:
            raise ValueError(f"cls_embedding must have shape {expected_shape}")
        chunks = cls_embedding.reshape(cls_embedding.shape[0], self.num_context_tokens, self.chunk_size)
        if self.shared:
            output = F.linear(chunks, self.weight[0])
        else:
            output = torch.einsum("bsc,sdc->bsd", chunks, self.weight)
        return output + self.bias


class CrossAttentionPredictor(nn.Module):
    r"""
    Predicts targets from context embeddings.

    The predictor consists of the following components:
        - Separate learnable position encodings for both context and target embeddings
        - A configurable series of predictor blocks operating in cross-attention, decoder, or encoder mode.
        - A linear projection head to create the final output.

    Args:
        backbone: Backbone model with a :meth:`create_cross_attention_layer` method.
        depth: Depth of the predictor network.
        out_dim: Output dimension of the predictor.
            If ``None``, the output dimension will be the same as the input dimension.
        device: Device to place the predictor on.
        disable_predictor_regularizers: Whether to force predictor stochastic depth, hidden dropout,
            and attention dropout to ``0.0``.
        attention_mode: Predictor attention mode. ``"cross_attention"`` keeps the current behavior,
            ``"decoder"`` adds self-attention over target queries before cross-attention, and
            ``"encoder"`` runs self-attention over the concatenated query/context sequence.
        cls_prediction_mode: CLS auxiliary-prediction path. AdaLN modes condition the shared
            predictor MLPs and expose a visually blinded MLP-only prediction path.
        cls_context_tokens: Number of predictor context tokens created by partitioned CLS modes.
    """

    def __init__(
        self,
        backbone: ViT,
        depth: int,
        out_dim: int | None = None,
        device: torch.device | None = None,
        disable_predictor_regularizers: bool = False,
        attention_mode: PredictorAttentionMode = CROSS_ATTENTION_PREDICTOR_MODE,
        cls_prediction_mode: CLSPredictionMode = LEGACY_CLS_PREDICTION_MODE,
        cls_context_tokens: int = CLS_CONTEXT_TOKENS,
    ):
        super().__init__()
        if attention_mode not in PREDICTOR_ATTENTION_MODES:
            raise ValueError(f"attention_mode must be one of {list(PREDICTOR_ATTENTION_MODES)}")
        if cls_prediction_mode not in CLS_PREDICTION_MODES:
            raise ValueError(f"cls_prediction_mode must be one of {list(CLS_PREDICTION_MODES)}")
        _validate_cls_context_tokens(cls_prediction_mode, cls_context_tokens)
        if cls_prediction_mode in ADALN_CLS_PREDICTION_MODES and attention_mode != CROSS_ATTENTION_PREDICTOR_MODE:
            raise ValueError("AdaLN CLS prediction requires attention_mode='cross_attention'")
        if cls_prediction_mode in EXPANDED_CLS_PREDICTION_MODES and attention_mode != CROSS_ATTENTION_PREDICTOR_MODE:
            raise ValueError("Expanded CLS prediction requires attention_mode='cross_attention'")
        if (
            cls_prediction_mode in JOINT_CONTEXT_CLS_PREDICTION_MODES
            and attention_mode != CROSS_ATTENTION_PREDICTOR_MODE
        ):
            raise ValueError("Joint-context CLS prediction requires attention_mode='cross_attention'")
        factory_kwargs = {"device": device, "dtype": backbone.config.dtype}
        self.hidden_size = backbone.config.hidden_size
        self.out_dim = out_dim or backbone.config.hidden_size
        self.attention_mode = attention_mode
        self.cls_prediction_mode: CLSPredictionMode = cls_prediction_mode
        self.cls_context_tokens = cls_context_tokens
        spatial_size = backbone.stem.tokenized_size(backbone.config.img_size)
        self.pos_enc_target = LearnablePosition(backbone.config.hidden_size, spatial_size, **factory_kwargs)
        self.rope = backbone.rope

        # Predictor blocks and output projection
        self.blocks = nn.ModuleList(
            [self._create_block(backbone, device, attention_mode, cls_prediction_mode) for _ in range(depth)]
        )
        if disable_predictor_regularizers:
            self._disable_predictor_regularizers()

        self.predictor_proj = self._create_projection(device=device, dtype=backbone.config.dtype)
        self.predictor_proj_shallow: nn.Linear | None = None
        self.cls_context_projection = (
            self._create_cls_context_projection(device=device, dtype=backbone.config.dtype)
            if cls_prediction_mode in PROJECTED_CLS_PREDICTION_MODES
            else None
        )
        self.cls_context_mlp = (
            self._create_cls_context_mlp(device=device, dtype=backbone.config.dtype)
            if cls_prediction_mode == RESIDUAL_MLP_CLS_PREDICTION_MODE
            else None
        )
        self.cls_context_slot_bias = (
            self._create_cls_context_slot_bias(device=device, dtype=backbone.config.dtype)
            if cls_prediction_mode == SLOT_BIAS_CLS_PREDICTION_MODE
            else None
        )
        self.cls_context_partition_projection = (
            self._create_cls_context_partition_projection(
                shared=cls_prediction_mode == PARTITIONED_SHARED_CLS_PREDICTION_MODE,
                device=device,
                dtype=backbone.config.dtype,
            )
            if cls_prediction_mode in PARTITIONED_CLS_PREDICTION_MODES
            else None
        )

    @property
    def predictor_dtype(self) -> torch.dtype:
        return self.predictor_proj.weight.dtype

    @staticmethod
    def _create_block(
        backbone: ViT,
        device: torch.device | None,
        attention_mode: PredictorAttentionMode,
        cls_prediction_mode: CLSPredictionMode,
    ) -> nn.Module:
        if attention_mode == CROSS_ATTENTION_PREDICTOR_MODE:
            if cls_prediction_mode in ADALN_CLS_PREDICTION_MODES:
                return CrossAttentionTransformer(
                    hidden_size=backbone.config.hidden_size,
                    ffn_hidden_size=backbone.config.ffn_hidden_size,
                    num_attention_heads=backbone.config.num_attention_heads,
                    hidden_dropout=backbone.config.hidden_dropout,
                    attention_dropout=backbone.config.attention_dropout,
                    attention_bias=backbone.config.attention_bias,
                    mlp_bias=backbone.config.mlp_bias,
                    activation=backbone.config.activation,
                    norm_type=backbone.config.norm_type,
                    qk_normalization=backbone.config.qk_normalization,
                    drop_path_rate=backbone.config.drop_path_rate,
                    layer_scale=backbone.config.layer_scale,
                    glu_limit=backbone.config.glu_limit,
                    glu_extra_bias=backbone.config.glu_extra_bias,
                    device=device,
                    dtype=backbone.config.dtype,
                    conditioning_size=backbone.config.hidden_size,
                    adaln_gate_init=0.0,
                )
            return backbone.create_cross_attention_layer(device=device, dtype=backbone.config.dtype)
        if attention_mode == DECODER_PREDICTOR_MODE:
            return backbone.create_decoder_layer(device=device, dtype=backbone.config.dtype)
        if attention_mode == ENCODER_PREDICTOR_MODE:
            return backbone.create_encoder_layer(device=device, dtype=backbone.config.dtype)
        raise ValueError(f"Unsupported predictor attention mode: {attention_mode!r}")

    @staticmethod
    def _reset_projection_parameters(projection: nn.Linear) -> None:
        # Teacher targets are RMS-normalized and the predictor query starts at small positional scales,
        # so keep the output head small and zero-centered at initialization.
        nn.init.trunc_normal_(projection.weight, std=PREDICTOR_PROJ_INIT_STD)
        if projection.bias is not None:
            nn.init.zeros_(projection.bias)

    def _create_projection(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> nn.Linear:
        projection = nn.Linear(
            self.hidden_size,
            self.out_dim,
            device=device,
            dtype=dtype,
        )
        self._reset_projection_parameters(projection)
        return projection

    def _create_cls_context_mlp(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Sequential:
        input_projection = nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        output_projection = nn.Linear(
            self.hidden_size,
            CLS_CONTEXT_TOKENS * self.hidden_size,
            device=device,
            dtype=dtype,
        )
        self._reset_projection_parameters(input_projection)
        self._reset_projection_parameters(output_projection)
        return nn.Sequential(input_projection, nn.GELU(), output_projection)

    def _create_cls_context_projection(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Linear:
        projection = nn.Linear(
            self.hidden_size,
            CLS_CONTEXT_TOKENS * self.hidden_size,
            device=device,
            dtype=dtype,
        )
        self._reset_projection_parameters(projection)
        return projection

    def _create_cls_context_slot_bias(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Parameter:
        slot_bias = nn.Parameter(torch.empty(1, CLS_CONTEXT_TOKENS, self.hidden_size, device=device, dtype=dtype))
        nn.init.trunc_normal_(slot_bias, std=PREDICTOR_PROJ_INIT_STD)
        return slot_bias

    def _create_cls_context_partition_projection(
        self,
        *,
        shared: bool,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> PartitionedCLSProjection:
        return PartitionedCLSProjection(
            self.hidden_size,
            shared=shared,
            num_context_tokens=self.cls_context_tokens,
            device=device,
            dtype=dtype,
        )

    def enable_shallow_head(self) -> None:
        if self.predictor_proj_shallow is not None:
            return

        self.predictor_proj_shallow = self._create_projection(
            device=self.predictor_proj.weight.device,
            dtype=self.predictor_proj.weight.dtype,
        )

    @staticmethod
    def _project(query: Tensor, projection: nn.Linear | None) -> Tensor | None:
        return projection(query).float() if projection is not None else None

    def _disable_predictor_regularizers(self) -> None:
        # Force predictor regularizers to zero without changing the backbone configuration.
        no_regularization = 0.0
        for block in self.blocks:
            typed_block = cast(Any, block)
            typed_block.drop_path_rate = no_regularization
            for attention in self._attention_modules(block):
                attention.dropout.p = no_regularization
                attention.attention_dropout.p = no_regularization
            typed_block.mlp.dropout.p = no_regularization

    @staticmethod
    def _attention_modules(block: nn.Module) -> tuple[Any, ...]:
        typed_block = cast(Any, block)
        return tuple(
            attention
            for attention_name in ("self_attention", "cross_attention")
            if (attention := getattr(typed_block, attention_name, None)) is not None
        )

    @staticmethod
    def _append_identity_rope(rope: Tensor, suffix_length: int) -> Tensor:
        if suffix_length == 0:
            return rope

        sin, cos = rope
        suffix_shape = (*sin.shape[:-2], suffix_length, sin.shape[-1])
        sin_suffix = sin.new_zeros(suffix_shape)
        cos_suffix = cos.new_ones(suffix_shape)
        return torch.stack([torch.cat([sin, sin_suffix], dim=-2), torch.cat([cos, cos_suffix], dim=-2)], dim=0)

    @staticmethod
    def _prepare_encoder_inputs(
        query: Tensor,
        context: Tensor,
        rope_q: Tensor | None,
        rope_k: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, int]:
        query_length = query.shape[1]
        combined = torch.cat([query, context], dim=1)
        if rope_q is None:
            return combined, None, query_length
        if rope_k is None:
            return combined, CrossAttentionPredictor._append_identity_rope(rope_q, context.shape[1]), query_length
        return combined, torch.cat([rope_q, rope_k], dim=3), query_length

    def _forward_encoder_blocks(
        self,
        query: Tensor,
        context: Tensor,
        rope_q: Tensor | None,
        rope_k: Tensor | None,
    ) -> Tensor:
        combined, combined_rope, query_length = self._prepare_encoder_inputs(query, context, rope_q, rope_k)
        for block in self.blocks:
            combined = cast(Any, block)(combined, rope=combined_rope)
        return combined[:, :query_length]

    def _forward_cross_or_decoder_blocks(
        self,
        query: Tensor,
        context: Tensor,
        rope_q: Tensor | None,
        rope_k: Tensor | None,
        conditioning: Tensor | None = None,
        context_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if context_attention_mask is not None and self.attention_mode != CROSS_ATTENTION_PREDICTOR_MODE:
            raise ValueError("context attention masks require attention_mode='cross_attention'")
        for block in self.blocks:
            if self.cls_prediction_mode in ADALN_CLS_PREDICTION_MODES:
                query = cast(Any, block)(
                    query,
                    context,
                    rope_q=rope_q,
                    rope_k=rope_k,
                    conditioning=conditioning,
                    attn_mask=context_attention_mask,
                )
            elif self.attention_mode == CROSS_ATTENTION_PREDICTOR_MODE:
                query = cast(Any, block)(
                    query,
                    context,
                    rope_q=rope_q,
                    rope_k=rope_k,
                    attn_mask=context_attention_mask,
                )
            else:
                query = cast(Any, block)(query, context, rope_q=rope_q, rope_k=rope_k)
        return query

    def _target_queries(self, tokenized_size: tuple[int, int], target_mask: Tensor) -> Tensor:
        batch_size, _ = target_mask.shape
        positions = self.pos_enc_target(tokenized_size).expand(batch_size, -1, -1)
        return apply_mask(target_mask, positions).to(dtype=self.predictor_dtype)

    def _prepare_conditioning(self, conditioning: Tensor | None, batch_size: int) -> Tensor | None:
        if self.cls_prediction_mode not in ADALN_CLS_PREDICTION_MODES:
            if conditioning is not None:
                raise ValueError("conditioning is only supported by AdaLN CLS prediction modes")
            return None
        if conditioning is None:
            raise ValueError("conditioning is required for AdaLN CLS prediction")
        expected_shape = (batch_size, self.hidden_size)
        if conditioning.ndim != 2 or tuple(conditioning.shape) != expected_shape:
            raise ValueError(f"conditioning must have shape {expected_shape}, got {tuple(conditioning.shape)}")
        return conditioning.to(dtype=self.predictor_dtype)

    def forward_features(
        self,
        tokenized_size: tuple[int, int],
        context: Tensor,
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
        conditioning: Tensor | None = None,
        context_attention_mask: Tensor | None = None,
    ) -> Tensor:
        with autocast_context(context.device.type, self.predictor_dtype):
            # Prepare inputs
            query = self._target_queries(tokenized_size, target_mask)
            base_target_tokens = query.shape[1]
            context = context.to(dtype=self.predictor_dtype)
            conditioning = self._prepare_conditioning(conditioning, target_mask.shape[0])
            if context_attention_mask is None and self.cls_prediction_mode in ROUTED_JOINT_CONTEXT_CLS_PREDICTION_MODES:
                context_attention_mask = create_joint_context_attention_mask(
                    self.cls_prediction_mode,
                    query.shape[0],
                    base_target_tokens,
                    context.shape[1] - 1,
                    device=context.device,
                    dtype=query.dtype,
                )
            rope_q, rope_k = self.prepare_rope(
                tokenized_size,
                context_mask,
                target_mask,
                rope_seed=rope_seed,
                context_length=context.shape[1],
            )
            if joint_context_query_multiplier(self.cls_prediction_mode) == 2:
                query = torch.cat([query, query], dim=1)
                if rope_q is not None:
                    rope_q = torch.cat([rope_q, rope_q], dim=-2)

            # Run query and context through predictor
            if self.attention_mode == ENCODER_PREDICTOR_MODE:
                return self._forward_encoder_blocks(query, context, rope_q, rope_k)
            return self._forward_cross_or_decoder_blocks(
                query,
                context,
                rope_q,
                rope_k,
                conditioning,
                context_attention_mask,
            )

    def forward_heads(
        self,
        tokenized_size: tuple[int, int],
        context: Tensor,
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
        conditioning: Tensor | None = None,
        context_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        query = self.forward_features(
            tokenized_size,
            context,
            context_mask,
            target_mask,
            rope_seed=rope_seed,
            conditioning=conditioning,
            context_attention_mask=context_attention_mask,
        )
        with autocast_context(query.device.type, self.predictor_dtype):
            deep_output = self.predictor_proj(query).float()
            shallow_output = self._project(query, self.predictor_proj_shallow)
        return deep_output, shallow_output

    def forward(
        self,
        tokenized_size: tuple[int, int],
        context: Tensor,
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
        conditioning: Tensor | None = None,
        context_attention_mask: Tensor | None = None,
    ) -> Tensor:
        deep_output, _ = self.forward_heads(
            tokenized_size,
            context,
            context_mask,
            target_mask,
            rope_seed=rope_seed,
            conditioning=conditioning,
            context_attention_mask=context_attention_mask,
        )
        return deep_output

    def forward_blind(
        self,
        tokenized_size: tuple[int, int],
        conditioning: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        """Predict masked targets from one conditioning embedding without attention or visual tokens."""
        if self.cls_prediction_mode not in ADALN_CLS_PREDICTION_MODES:
            raise ValueError("forward_blind requires an AdaLN CLS prediction mode")
        with autocast_context(conditioning.device.type, self.predictor_dtype):
            query = self._target_queries(tokenized_size, target_mask)
            prepared_conditioning = self._prepare_conditioning(conditioning, target_mask.shape[0])
            assert prepared_conditioning is not None
            for block in self.blocks:
                typed_block = cast(Any, block)
                query = query + typed_block.layer_scale_mlp(typed_block.mlp(query, conditioning=prepared_conditioning))
            return self.predictor_proj(query).float()

    def _validate_single_cls_context(self, cls_context: Tensor) -> None:
        if cls_context.ndim != 3:
            raise ValueError("cls_context must have shape [batch, 1, hidden_size]")
        expected_shape = (cls_context.shape[0], 1, self.hidden_size)
        if tuple(cls_context.shape) != expected_shape:
            raise ValueError(f"cls_context must contain exactly one CLS token with shape {expected_shape}")

    def project_cls_context(self, cls_context: Tensor) -> Tensor:
        """Expand one CLS embedding into four affine views for the legacy predictor path."""
        projection = self.cls_context_projection
        if self.cls_prediction_mode not in PROJECTED_CLS_PREDICTION_MODES or projection is None:
            raise ValueError("project_cls_context requires an affine projected CLS prediction mode")
        self._validate_single_cls_context(cls_context)
        with autocast_context(cls_context.device.type, self.predictor_dtype):
            context = cls_context.to(dtype=self.predictor_dtype)
            projected = projection(context[:, 0]).reshape(
                cls_context.shape[0],
                CLS_CONTEXT_TOKENS,
                self.hidden_size,
            )
            if self.cls_prediction_mode == RESIDUAL_PROJECTED_CLS_PREDICTION_MODE:
                projected = projected + context.expand(-1, CLS_CONTEXT_TOKENS, -1)
            return projected

    def expand_cls_context(self, cls_context: Tensor) -> Tensor:
        """Expand one CLS embedding into contexts for a configured legacy predictor replay."""
        if self.cls_prediction_mode in PROJECTED_CLS_PREDICTION_MODES:
            return self.project_cls_context(cls_context)
        if self.cls_prediction_mode in PARTITIONED_CLS_PREDICTION_MODES:
            self._validate_single_cls_context(cls_context)
            assert self.cls_context_partition_projection is not None
            with autocast_context(cls_context.device.type, self.predictor_dtype):
                context = cls_context.to(dtype=self.predictor_dtype)
                return self.cls_context_partition_projection(context[:, 0])
        if self.cls_prediction_mode == RESIDUAL_MLP_CLS_PREDICTION_MODE:
            self._validate_single_cls_context(cls_context)
            assert self.cls_context_mlp is not None
            with autocast_context(cls_context.device.type, self.predictor_dtype):
                context = cls_context.to(dtype=self.predictor_dtype)
                residual = self.cls_context_mlp(context[:, 0]).reshape(
                    cls_context.shape[0],
                    CLS_CONTEXT_TOKENS,
                    self.hidden_size,
                )
                return context.expand(-1, CLS_CONTEXT_TOKENS, -1) + residual
        if self.cls_prediction_mode != SLOT_BIAS_CLS_PREDICTION_MODE or self.cls_context_slot_bias is None:
            raise ValueError("expand_cls_context requires an expanded CLS prediction mode")
        self._validate_single_cls_context(cls_context)
        with autocast_context(cls_context.device.type, self.predictor_dtype):
            context = cls_context.to(dtype=self.predictor_dtype)
            return context.expand(-1, CLS_CONTEXT_TOKENS, -1) + self.cls_context_slot_bias

    def forward_shallow(
        self,
        tokenized_size: tuple[int, int],
        context: Tensor,
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
    ) -> Tensor:
        _, shallow_output = self.forward_heads(tokenized_size, context, context_mask, target_mask, rope_seed=rope_seed)
        if shallow_output is None:
            raise ValueError("Shallow predictor head is not enabled")
        return shallow_output

    if TYPE_CHECKING:

        def __call__(
            self,
            tokenized_size: tuple[int, int],
            context: Tensor,
            context_mask: Tensor | None,
            target_mask: Tensor,
            rope_seed: int | None = None,
            conditioning: Tensor | None = None,
            context_attention_mask: Tensor | None = None,
        ) -> Tensor:
            return self.forward(
                tokenized_size,
                context,
                context_mask,
                target_mask,
                rope_seed,
                conditioning,
                context_attention_mask,
            )

    def prepare_rope(
        self,
        tokenized_size: tuple[int, int],
        context_mask: Tensor | None,
        target_mask: Tensor,
        rope_seed: int | None = None,
        context_length: int | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.rope is None:
            return None, None

        H, W = tokenized_size
        rope = self.rope(H=H, W=W, rope_seed=rope_seed)
        sin, cos = rope
        B = target_mask.shape[0]

        sin_q = apply_mask(target_mask, sin[None].expand(B, -1, -1))
        cos_q = apply_mask(target_mask, cos[None].expand(B, -1, -1))
        rope_q = torch.stack([sin_q[:, None, ...], cos_q[:, None, ...]], dim=0)

        if context_mask is not None:
            sin_k = apply_mask(context_mask, sin[None].expand(B, -1, -1))
            cos_k = apply_mask(context_mask, cos[None].expand(B, -1, -1))
            rope_k = torch.stack([sin_k[:, None, ...], cos_k[:, None, ...]], dim=0)
            if context_length is not None:
                identity_suffix_length = context_length - sin_k.shape[-2]
                if identity_suffix_length < 0:
                    raise ValueError("context length cannot be smaller than the positioned context")
                rope_k = self._append_identity_rope(rope_k, identity_suffix_length)
        else:
            rope_k = None
        return rope_q, rope_k


@torch.no_grad()
def generate_masks(
    backbone: ViT,
    x: Tensor,
    context_ratio: float,
    target_ratio: float,
    scale: int,
    roll: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""Generate context masks and the corresponding prediction target masks.

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
    context_mask = backbone.create_mask(x, context_ratio, scale, roll=roll)

    if target_ratio == 1.0:
        # Full target coverage intentionally includes context tokens.
        target_mask = torch.ones_like(context_mask)
    else:
        # For partial targets, keep the existing non-overlapping behavior.
        target_mask = generate_non_overlapping_mask(context_mask, context_ratio, target_ratio)
    return context_mask, target_mask


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float = 0.0) -> None:
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
    teacher_params = [cast(Tensor, param) for param in teacher.parameters()]
    student_params = [cast(Tensor, param) for param in student.parameters()]
    matching_dtypes = all(
        teacher_param.dtype == student_param.dtype and teacher_param.device == student_param.device
        for teacher_param, student_param in zip(teacher_params, student_params)
    )
    if matching_dtypes:
        # PyTorch documents foreach operations as beta, but its type stubs do not export this symbol.
        torch._foreach_lerp_(teacher_params, student_params, weight)  # pyright: ignore[reportPrivateImportUsage]
        return

    for teacher_param, student_param in zip(teacher_params, student_params):
        teacher_param.lerp_(student_param.to(device=teacher_param.device, dtype=teacher_param.dtype), weight)


def get_momentum(step: int, total_steps: int, momentum: float, scheduled: bool = False) -> float:
    r"""Linearly anneal momentum from the given value to 1.0 over the course of training.

    Args:
        step: Current step in the training loop.
        total_steps: Total number of steps in the training loop.
        momentum: The base momentum for the EMA update.
        scheduled: Whether to schedule the momentum.

    Returns:
        The current momentum value.
    """
    return min(momentum + (1 - momentum) * (step / total_steps), 1.0) if scheduled else momentum


M = TypeVar("M", bound=nn.Module)


def setup_teacher(backbone: M, dtype: torch.dtype | str | None = None) -> M:
    r"""Create a teacher model from a backbone model.

    The teacher will have parameters frozen and will be set to evaluation mode.

    Args:
        backbone: Backbone model to create a teacher from.
        dtype: Optional dtype override for the copied teacher weights.

    Returns:
        Teacher model.
    """
    teacher = deepcopy(backbone)
    teacher_dtype = _normalize_dtype(dtype, "teacher dtype")
    if teacher_dtype is not None:
        teacher = cast(M, teacher.to(dtype=teacher_dtype))
        _sync_module_config_dtype(teacher, teacher_dtype)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.compile(fullgraph=True)
def compute_gram_loss(student: Tensor, teacher: Tensor, normalize: bool = True, remove_neg: bool = True) -> Tensor:
    r"""Compute the Gram loss between student features and the Gram teacher's features.

    Args:
        student: Student features.
        teacher: Gram teacher features.
        normalize: Whether to normalize the features.
        remove_neg: Whether to remove negative values from the Gram matrix.

    Shapes:
        student: :math:`(*, L, D)`
        teacher: :math:`(*, L, D)`
        Output: Scalar

    Returns:
        The Gram loss.
    """
    student = F.normalize(student, dim=-1) if normalize else student
    teacher = F.normalize(teacher, dim=-1) if normalize else teacher

    student_sim = student.bmm(student.mT)
    teacher_sim = teacher.bmm(teacher.mT)
    if remove_neg:
        teacher_sim = teacher_sim.clamp(min=0.0)
        student_sim = student_sim.clamp(min=0.0)

    return F.mse_loss(student_sim, teacher_sim)


def compute_jepa_prediction_loss(
    student: Tensor,
    teacher: Tensor,
    kind: JEPALossKind = MSE_JEPA_LOSS_KIND,
) -> Tensor:
    r"""Compute the JEPA reconstruction loss between student predictions and teacher targets.

    Args:
        student: Student predictions.
        teacher: Teacher targets.
        kind: Reconstruction loss kind to apply.

    Shapes:
        student: :math:`(*, L, D)`
        teacher: :math:`(*, L, D)`
        Output: Scalar

    Returns:
        The reconstruction loss.
    """
    student = student.float()
    teacher = teacher.float()

    if kind == MSE_JEPA_LOSS_KIND:
        return F.mse_loss(student, teacher)
    if kind == COSINE_JEPA_LOSS_KIND:
        return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()
    raise ValueError(f"Unsupported JEPA loss kind: {kind!r}")


def is_gram_update_epoch(epoch: int, gram_start_epoch: int | None, gram_update_interval_epoch: int) -> bool:
    r"""Check if the current epoch is a Gram update epoch.

    Args:
        epoch: Current epoch.
        gram_start_epoch: The epoch at which to store a checkpoint and begin computing the Gram loss.
        gram_update_interval_epoch: The interval at which to update the Gram teacher after the initial setup.
    """
    if gram_start_epoch is None or gram_update_interval_epoch <= 0:
        return False
    return epoch > gram_start_epoch and (epoch - gram_start_epoch) % gram_update_interval_epoch == 0


def forward_gram_teacher(
    gram_teacher: ViT,
    img: Tensor,
    rope_seed: int | None = None,
    resolution_scale: float = 1.0,
) -> Tensor:
    r"""Forward pass through the Gram teacher.

    Applies image upsampling and downsampling to the input and output, respectively.

    Args:
        gram_teacher: Gram teacher model.
        img: Input image.
        rope_seed: Rope seed.
        resolution_scale: Resolution scale.

    Shapes:
        img: :math:`(*, C, H, W)`
        Output: :math:`(*, L, D)`

    Returns:
        Output features.
    """
    # Resize input according to resolution scale, tracking the tokenized size of the input and output
    target_tokenized_size = gram_teacher.stem.tokenized_size(img.shape[-2:])
    img = F.interpolate(img, scale_factor=resolution_scale, mode="bilinear", align_corners=False)
    output_tokenized_size = gram_teacher.stem.tokenized_size(img.shape[-2:])

    assert not gram_teacher.training, "Gram teacher must be in evaluation mode"

    # Forward pass and resize output features to original size
    gram_teacher_output = gram_teacher(img, rope_seed=rope_seed).visual_tokens
    B, _, D = gram_teacher_output.shape
    gram_teacher_output = gram_teacher_output.movedim(1, -1).reshape(B, D, *output_tokenized_size)
    gram_teacher_output = F.interpolate(
        gram_teacher_output, scale_factor=1 / resolution_scale, mode="bilinear", align_corners=False
    )
    gram_teacher_output = gram_teacher_output.flatten(2).movedim(-1, 1)

    assert gram_teacher_output.shape == (B, math.prod(target_tokenized_size), D)
    return gram_teacher_output


@torch.compile
def _compute_sigreg_loss_impl(x: Tensor, global_step: int, num_slices: int = 256) -> Tensor:
    r"""Compute the LeJEPA SigREG loss.

    This loss encourages features to follow an isotropic Gaussian distribution.

    Args:
        x: Input tensor.
        global_step: Global step.
        num_slices: Number of slices to use for the projection.

    Shapes:
        x: :math:`(*, L, D)`
        Output: Scalar

    Returns:
        The SigREG loss.
    """
    B, L, D = x.shape

    proj_shape = (1, D, num_slices)
    # Only pass CUDA devices to fork_rng to avoid CUDA initialization on CPU
    fork_devices = [x.device] if x.device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.random.manual_seed(global_step)
        A = torch.randn(proj_shape, device=x.device)
    A = F.normalize(A, dim=-2)

    t = torch.linspace(-5, 5, 17, device=x.device)
    exp_f = torch.exp(-0.5 * t**2)

    x_t = torch.bmm(x.type_as(A), A.expand(B, -1, -1)).unsqueeze(-1) * t
    ecf = (1j * x_t).exp().mean(-3)

    if dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(ecf, op=dist.ReduceOp.AVG)
        assert isinstance(ecf, Tensor)
    else:
        world_size = 1

    err = (ecf - exp_f).abs().square().mul(exp_f)
    N = L * world_size
    T = torch.trapz(err, t, dim=-1) * N
    assert T.shape == (B, num_slices)
    return T.mean()


def compute_sigreg_loss(x: Tensor, global_step: int, num_slices: int = 256) -> Tensor:
    r"""Compute the LeJEPA SigREG loss.

    Args:
        x: Input tensor with a non-empty token dimension.
        global_step: Global step.
        num_slices: Number of slices to use for the projection.

    Shapes:
        x: :math:`(*, L, D)` with :math:`L > 0`
        Output: Scalar

    Returns:
        The SigREG loss.
    """
    if x.shape[-2] <= 0:
        raise ValueError("x must contain at least one token")
    return _compute_sigreg_loss_impl(x, global_step, num_slices)


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
            from this value to 1.0 over the course of training if scheduled is ``True``.
        scheduled: Whether to schedule the momentum.
        predictor_depth: Depth of the predictor network.
        predictor_attention_mode: Predictor block attention mode.
        cls_prediction_mode: CLS auxiliary prediction path.
        cls_context_tokens: Number of predictor contexts produced by partitioned CLS modes.
        disable_predictor_regularizers: Whether to force predictor stochastic depth, hidden dropout,
            and attention dropout to ``0.0``.
        teacher_dtype: Optional dtype override for the teacher and Gram teacher weights.
        use_gram_anchoring: Whether to enable Gram anchoring at all.
        gram_teacher_epoch: Epoch when the Gram teacher is first synchronized from the teacher.
        gram_start_epoch: The epoch at which to begin computing the Gram loss.
            If ``None``, the Gram loss cannot be computed.
        gram_update_interval_epoch: The interval at which to update the Gram teacher after the initial setup.
        gram_resolution_scale: The scale at which to feed inputs through the Gram teacher.
        gram_remove_neg: Whether to remove negative values from the Gram matrix.
        gram_loss_weight: The coefficient of the Gram loss.
        sigreg_loss_weight: The coefficient of the SigREG loss.
        jepa_loss_weight: The coefficient of the primary JEPA prediction loss.
        stem_jepa_loss_weight: The coefficient of the stem-target JEPA loss.
        jepa_loss_kind: Reconstruction loss kind applied to both JEPA prediction losses.
    """

    context_ratio: float = 0.5
    target_ratio: float = 0.25
    scale: int = 4
    momentum: float = 0.99
    scheduled: bool = False
    predictor_depth: int = 4
    predictor_attention_mode: PredictorAttentionMode = CROSS_ATTENTION_PREDICTOR_MODE
    cls_prediction_mode: CLSPredictionMode = LEGACY_CLS_PREDICTION_MODE
    cls_context_tokens: int = CLS_CONTEXT_TOKENS
    disable_predictor_regularizers: bool = False
    teacher_dtype: torch.dtype | str | None = None
    use_gram_anchoring: bool = False
    gram_teacher_epoch: int = 100
    gram_start_epoch: int | None = None
    gram_update_interval_epoch: int = 10
    gram_resolution_scale: float = 2.0
    gram_remove_neg: bool = False
    gram_loss_weight: float = 1.0
    sigreg_loss_weight: float = 1e-4
    jepa_loss_weight: float = 1.0
    stem_jepa_loss_weight: float = 0.0
    jepa_loss_kind: JEPALossKind = MSE_JEPA_LOSS_KIND

    def __post_init__(self) -> None:
        self.teacher_dtype = _normalize_dtype(self.teacher_dtype, "teacher_dtype")
        if not 0 < self.context_ratio <= 1:
            raise ValueError("context_ratio must be in the range (0, 1]")
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in the range (0, 1]")
        if self.predictor_attention_mode not in PREDICTOR_ATTENTION_MODES:
            raise ValueError(f"predictor_attention_mode must be one of {list(PREDICTOR_ATTENTION_MODES)}")
        if self.cls_prediction_mode not in CLS_PREDICTION_MODES:
            raise ValueError(f"cls_prediction_mode must be one of {list(CLS_PREDICTION_MODES)}")
        _validate_cls_context_tokens(self.cls_prediction_mode, self.cls_context_tokens)
        if (
            self.cls_prediction_mode in ADALN_CLS_PREDICTION_MODES
            and self.predictor_attention_mode != CROSS_ATTENTION_PREDICTOR_MODE
        ):
            raise ValueError("AdaLN CLS prediction requires predictor_attention_mode='cross_attention'")
        if (
            self.cls_prediction_mode in EXPANDED_CLS_PREDICTION_MODES
            and self.predictor_attention_mode != CROSS_ATTENTION_PREDICTOR_MODE
        ):
            raise ValueError("Expanded CLS prediction requires predictor_attention_mode='cross_attention'")
        if (
            self.cls_prediction_mode in JOINT_CONTEXT_CLS_PREDICTION_MODES
            and self.predictor_attention_mode != CROSS_ATTENTION_PREDICTOR_MODE
        ):
            raise ValueError("Joint-context CLS prediction requires predictor_attention_mode='cross_attention'")
        if self.use_gram_anchoring and self.gram_start_epoch is None:
            raise ValueError("gram_start_epoch must be set when use_gram_anchoring is enabled")
        if self.gram_teacher_epoch <= 0:
            raise ValueError("gram_teacher_epoch must be a positive integer")
        if self.gram_start_epoch is not None and self.gram_start_epoch <= 0:
            raise ValueError("gram_start_epoch must be a positive integer or None")
        if self.gram_update_interval_epoch < 0:
            raise ValueError("gram_update_interval_epoch must be a non-negative integer")
        if self.gram_start_epoch is not None and self.gram_start_epoch < self.gram_teacher_epoch:
            raise ValueError("gram_start_epoch must be greater than or equal to gram_teacher_epoch")
        if self.gram_resolution_scale <= 0:
            raise ValueError("gram_resolution_scale must be a positive float")
        if self.gram_loss_weight <= 0:
            raise ValueError("gram_loss_weight must be a positive float")
        if self.sigreg_loss_weight < 0:
            raise ValueError("sigreg_loss_weight must be a non-negative float")
        if self.jepa_loss_weight < 0:
            raise ValueError("jepa_loss_weight must be a non-negative float")
        if self.stem_jepa_loss_weight < 0:
            raise ValueError("stem_jepa_loss_weight must be a non-negative float")
        if self.jepa_loss_kind not in JEPA_LOSS_KINDS:
            raise ValueError(f"jepa_loss_kind must be one of {list(JEPA_LOSS_KINDS)}")


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
