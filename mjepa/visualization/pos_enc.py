from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Self, Sequence, Type

import matplotlib.pyplot as plt
import safetensors.torch as st
import torch
import yaml
from torch import Tensor
from vit import ViT, ViTConfig

from mjepa.jepa import register_constructors

from .pca import existing_file_type, output_path_type, torch_dtype_type


register_constructors()


def create_pos_enc_maps(positions: Tensor) -> plt.Figure:
    # Get shape and flatten leading dimensions
    *spatial, dim = positions.shape
    flat = positions.reshape(-1, dim)

    # Get corner and center indices
    h, w = spatial
    corners = [0, w - 1, (h - 1) * w, h * w - 1]  # top-left, top-right, bottom-left, bottom-right
    center_idx = (h // 2) * w + (w // 2)

    # Combine corners and center
    indices = corners + [center_idx]
    labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"]

    # Compute similarities for each position
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Position Embedding Similarities", fontsize=16)

    for idx, (pos_idx, label) in enumerate(zip(indices, labels)):
        # Get similarities between position token and all others
        pos_token = flat[pos_idx : pos_idx + 1]
        sims = torch.mm(pos_token, flat.T)

        # Reshape to spatial dimensions and convert to numpy
        sims = sims.reshape(1, *spatial).squeeze().cpu().numpy()

        # Plot heatmap
        im = axes[idx].imshow(sims, cmap="RdBu")
        axes[idx].axis("off")
        axes[idx].set_title(label, fontsize=12)

        # Add colorbar
        plt.colorbar(im, ax=axes[idx])

        # Add X marker at position
        pos_y, pos_x = pos_idx // spatial[1], pos_idx % spatial[1]
        axes[idx].plot(pos_x, pos_y, "x", markersize=10, markeredgewidth=2, color="red")

    plt.tight_layout()
    return fig


@dataclass
class PositionVisualizer:
    model: ViT
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    size: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if self.size is None:
            self.size = self.model.config.img_size
        self.model = self.model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.inference_mode()
    def _get_positions(self) -> Tensor:
        size = self.model.stem.pos_enc.spatial_size
        with torch.autocast(self.device.type, dtype=self.dtype):
            positions = self.model.stem.pos_enc(size)
        return positions.view(*size, -1).float()

    @torch.inference_mode()
    def __call__(self) -> plt.Figure:
        assert not self.model.training, "Model must be in evaluation mode"
        positions = self._get_positions()
        return create_pos_enc_maps(positions)

    @torch.inference_mode()
    def save(self, fig: plt.Figure, output: Path) -> None:
        fig.savefig(output, dpi=300, bbox_inches="tight")

    @classmethod
    def create_parser(cls: Type[Self], custom_loader: bool = False) -> ArgumentParser:
        """Create argument parser with built-in validation."""
        parser = ArgumentParser(prog="pca-visualize", description="Visualize PCA ViT output features")
        parser.add_argument("config", type=existing_file_type, help="Path to model YAML configuration file")
        parser.add_argument("checkpoint", type=existing_file_type, help="Path to safetensors checkpoint")
        parser.add_argument("output", type=output_path_type, help="Path to output PNG file")
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
            args.device,
            args.dtype,
            args.size,
        )
