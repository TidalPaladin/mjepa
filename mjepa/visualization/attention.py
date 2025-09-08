import math
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Self, Sequence, Tuple, Type

import matplotlib.pyplot as plt
import safetensors.torch as st
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from torch import Tensor
from vit import ViT, ViTConfig
from vit.attention import AttentivePool
from vit.head import Head, MLPHead

from mjepa.jepa import register_constructors

from .pca import ExpandImagePathsAction, existing_file_type, output_path_type, torch_dtype_type


register_constructors()


def _apply_colormap(x: Tensor) -> Tensor:
    x = torch.from_numpy(plt.cm.inferno(x.squeeze().cpu())[..., :3]).float()
    return x


def _process_weights(
    weights: Tensor,
    original_img_shape: Tuple[int, int],
    grid_size: Tuple[int, int],
    ax: plt.Axes,
    batch_idx: int,
    head_idx: int,
    caption: str | None = None,
) -> Tensor:
    weights = F.interpolate(weights.view(1, 1, *grid_size), size=original_img_shape, mode="nearest").view(
        1, *original_img_shape
    )

    # Normalize weights
    w_min, w_max = weights.aminmax()
    weights = (weights - w_min) / (w_max - w_min + 1e-8)

    # Apply colormap
    weights_colored = _apply_colormap(weights)

    # Overlay attention weights on image using colormap alpha
    ax.imshow(weights_colored.numpy(), origin="upper")

    # Set title with layer/head/token info
    if caption is not None:
        ax.set_title(caption, fontsize=32)
    elif batch_idx == 0 and head_idx >= 0:
        ax.set_title(f"Head {head_idx + 1}", fontsize=32)
    ax.axis("off")


def _create_subplot_grid(B: int, H: int) -> Tuple[plt.Figure, plt.Axes, int, int]:
    """Create subplot grid based on batch size and number of heads."""
    if H == 1:
        # Special layout for single attention head: 4 image-weight pairs per row
        pairs_per_row = 4
        nrows = math.ceil(B / pairs_per_row)
        ncols = pairs_per_row * 2
    else:
        # Original layout for multiple attention heads
        head_offset = 3
        nrows, ncols = B, H + head_offset

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    # Ensure axes is always 2D
    if nrows == 1 and ncols == 1:
        axes = axes.reshape(1, 1)
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    return fig, axes, nrows, ncols


def _visualize_single_head_layout(
    axes: plt.Axes,
    image: Tensor,
    weights: Tensor,
    original_img_shape: Tuple[int, int],
    grid: Tuple[int, int],
    B: int,
    nrows: int,
    ncols: int,
) -> None:
    """Handle visualization for single attention head layout."""
    pairs_per_row = 4

    for batch_idx in range(B):
        row = batch_idx // pairs_per_row
        col_start = (batch_idx % pairs_per_row) * 2

        # Show image
        axes[row, col_start].imshow(image[batch_idx].numpy(), origin="upper")
        axes[row, col_start].axis("off")

        # Show attention weight
        _process_weights(
            weights[batch_idx, ..., 0],
            original_img_shape,
            grid,
            axes[row, col_start + 1],
            batch_idx,
            -1,
            caption=None,
        )

    # Hide unused subplots
    for batch_idx in range(B, nrows * pairs_per_row):
        row = batch_idx // pairs_per_row
        col_start = (batch_idx % pairs_per_row) * 2
        for col in [col_start, col_start + 1]:
            if row < nrows and col < ncols:
                axes[row, col].axis("off")


def _visualize_multi_head_layout(
    axes: plt.Axes,
    image: Tensor,
    weights: Tensor,
    original_img_shape: Tuple[int, int],
    grid: Tuple[int, int],
    B: int,
    H: int,
) -> None:
    """Handle visualization for multiple attention heads layout."""
    head_offset = 3

    for batch_idx in range(B):
        # Show original image
        axes[batch_idx, 0].imshow(image[batch_idx].numpy(), origin="upper")
        axes[batch_idx, 0].axis("off")
        if batch_idx == 0:
            axes[batch_idx, 0].set_title("Image", fontsize=32)

        # Show average and max attention
        for col, (agg_func, caption) in enumerate(
            [(lambda x: x.mean(dim=-1), "Average"), (lambda x: x.amax(dim=-1), "Max")], start=1
        ):
            _process_weights(
                agg_func(weights[batch_idx]),
                original_img_shape,
                grid,
                axes[batch_idx, col],
                batch_idx,
                -1,
                caption=caption if batch_idx == 0 else None,
            )

        # Show individual attention heads
        for head_idx in range(H):
            _process_weights(
                weights[batch_idx, ..., head_idx],
                original_img_shape,
                grid,
                axes[batch_idx, head_idx + head_offset],
                batch_idx,
                head_idx,
            )


def visualize_attention_weights(
    image: Tensor,
    pred: Tensor,
    weights: Tensor,
) -> plt.Figure:
    """Visualize attention weights for a specific layer and token across all heads.

    Args:
        image: Input image tensor of shape (B, C, H, W)
        pred: Model predictions
        weights: Attention weights tensor

    Returns:
        matplotlib Figure with attention weight visualizations overlaid on the input image
    """
    # Preprocess image
    original_img_shape = image.shape[2:]  # (H, W)
    image = image.cpu()
    if image.shape[1] == 1:
        image = image.expand(-1, 3, -1, -1)
    image = rearrange(image, "... c h w -> ... h w c").cpu()

    B, *grid, H = weights.shape

    # Create subplot grid
    fig, axes, nrows, ncols = _create_subplot_grid(B, H)

    # Choose layout based on number of heads
    if H == 1:
        _visualize_single_head_layout(axes, image, weights, original_img_shape, grid, B, nrows, ncols)
        top_adjust = 0.88
    else:
        _visualize_multi_head_layout(axes, image, weights, original_img_shape, grid, B, H)
        top_adjust = 0.92

    # Finalize figure
    fig.suptitle("Attention Weight Visualization", fontsize=48, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=top_adjust)

    return fig


@dataclass
class AttentionVisualizer:
    model: ViT
    head_key: str
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    size: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if self.size is None:
            self.size = self.model.config.img_size
        self.model = self.model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        if not isinstance(self.head, (Head, MLPHead)):
            raise ValueError(f"Head {self.head_key} is not an attention head")
        if not isinstance(self.head.pool, AttentivePool):
            raise ValueError(f"Head {self.head_key} does not have an attention pool")

    @property
    def head(self) -> Head | MLPHead:
        return self.model.heads[self.head_key]

    @property
    def pooling(self) -> AttentivePool:
        return self.head.pool

    @property
    def num_attention_heads(self) -> int:
        return self.model.config.heads[self.head_key].num_attention_heads or self.model.config.num_attention_heads

    @torch.inference_mode()
    def _forward_attention_weights(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        assert not self.model.training, "Model must be in evaluation mode"
        assert img.device == self.device, "Image must be on the same device as the model"
        tokenized_size = self.model.stem.tokenized_size(img.shape[2:])
        B = img.shape[0]
        H = self.num_attention_heads
        with torch.autocast(self.device.type, dtype=self.dtype):
            features = self.model(img)
            weights = self.pooling.forward_weights(features).view(B, H, *tokenized_size).movedim(1, -1)
            pred = self.head(features)
        return pred.float(), weights.float().contiguous()

    @torch.inference_mode()
    def __call__(self, img: Tensor) -> plt.Figure:
        assert not self.model.training, "Model must be in evaluation mode"
        img = img.to(self.device)
        img = F.interpolate(img, size=self.size, mode="bilinear", align_corners=False)

        pred, weights = self._forward_attention_weights(img)
        fig = visualize_attention_weights(img, pred, weights)
        return fig

    @torch.inference_mode()
    def save(self, fig: plt.Figure, output: Path) -> None:
        fig.savefig(output)

    @classmethod
    def create_parser(cls: Type[Self], custom_loader: bool = False) -> ArgumentParser:
        """Create argument parser with built-in validation."""
        parser = ArgumentParser(prog="attention-visualize", description="Visualize attention weights of a ViT model")
        parser.add_argument("config", type=existing_file_type, help="Path to model YAML configuration file")
        parser.add_argument("checkpoint", type=existing_file_type, help="Path to safetensors checkpoint")
        parser.add_argument(
            "input",
            nargs="+",
            action=ExpandImagePathsAction if not custom_loader else None,
            help="Path to input image(s) or directory containing .tiff files",
        )
        parser.add_argument("output", type=output_path_type, help="Path to output file")
        parser.add_argument(
            "-s",
            "--size",
            type=int,
            nargs=2,
            default=None,
            help="Size of the input image. By default size is inferred from the model configuration.",
        )
        parser.add_argument("-d", "--device", default="cpu", type=torch.device, help="Device to run the model on")
        parser.add_argument(
            "-dt", "--dtype", default="fp32", type=torch_dtype_type, help="Data type to run the model on"
        )
        parser.add_argument("--head", default="cls", help="Head to visualize")
        return parser

    @classmethod
    def from_args(cls: Type[Self], args: Namespace) -> Self:
        # Create model
        config = yaml.full_load(args.config.read_text())["backbone"]
        assert isinstance(config, ViTConfig)
        model = config.instantiate()

        # Load checkpoint
        state_dict = st.load_file(args.checkpoint)
        model.load_state_dict(state_dict)

        return cls(
            model,
            args.head,
            args.device,
            args.dtype,
            args.size,
        )
