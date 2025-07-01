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


def visualize_attention_weights(
    image: Tensor,
    pred: Tensor,
    weights: Tensor,
    overlay_alpha: float = 0.6,
) -> plt.Figure:
    """Visualize attention weights for a specific layer and token across all heads.

    Args:
        weights: Dictionary with keys like 'layer_i' and attention weight tensors
        layer: Specific layer index to visualize
        token: Specific token index to visualize
        image: Input image tensor of shape (B, C, H, W), None to skip image display
        overlay_alpha: Alpha value for blending attention weights over image (0-1)

    Returns:
        matplotlib Figure with attention weight visualizations overlaid on the input image
    """
    original_img_shape = image.shape[2:]  # (H, W)
    image = image.cpu()
    if image.shape[1] == 1:
        image = image.expand(-1, 3, -1, -1)
    image = rearrange(image, "... c h w -> ... h w c").cpu()

    B, *grid, H = weights.shape
    head_offset = 3
    nrows, ncols = B, H + head_offset

    # Create figure and subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    # Generate visualizations for each head
    for batch_idx in range(B):
        ax = axes[batch_idx, 0]
        ax.imshow(image[batch_idx].numpy(), origin="upper")
        ax.axis("off")
        if batch_idx == 0:
            ax.set_title("Image", fontsize=32)

        _process_weights(
            weights[batch_idx].mean(dim=-1),
            original_img_shape,
            grid,
            axes[batch_idx, 1],
            batch_idx,
            -1,
            caption="Average" if batch_idx == 0 else None,
        )
        _process_weights(
            weights[batch_idx].amax(dim=-1),
            original_img_shape,
            grid,
            axes[batch_idx, 2],
            batch_idx,
            -1,
            caption="Max" if batch_idx == 0 else None,
        )

        for head_idx in range(H):
            _process_weights(
                weights[batch_idx, ..., head_idx],
                original_img_shape,
                grid,
                axes[batch_idx, head_idx + head_offset],
                batch_idx,
                head_idx,
            )

    # Add overall title
    fig.suptitle(f"Attention Weight Visualization", fontsize=48, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    return fig


@dataclass
class AttentionVisualizer:
    model: ViT
    head_key: str
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    size: Sequence[int] | None = None
    overlay_alpha: float = 0.6

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
            weights = self.pooling.forward_weights(features).view(B, *tokenized_size, H)
            pred = self.head(features)
        return pred.float(), weights.float().contiguous()

    @torch.inference_mode()
    def __call__(self, img: Tensor) -> plt.Figure:
        assert not self.model.training, "Model must be in evaluation mode"
        img = img.to(self.device)
        img = F.interpolate(img, size=self.size, mode="bilinear", align_corners=False)

        pred, weights = self._forward_attention_weights(img)
        fig = visualize_attention_weights(
            img,
            pred,
            weights,
            overlay_alpha=self.overlay_alpha,
        )
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
        parser.add_argument("--overlay-alpha", type=float, default=0.6, help="Alpha value for overlay blending (0-1)")
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
            args.overlay_alpha,
        )
