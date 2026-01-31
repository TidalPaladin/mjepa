from argparse import Action, ArgumentParser, Namespace
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, cast

import matplotlib.pyplot as plt  # type: ignore[import]
import safetensors.torch as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dicom_preprocessing import load_tiff_f32
from einops import rearrange
from torch import Tensor
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from vit import ViT, ViTConfig

from mjepa.jepa import register_constructors


register_constructors()


def cosine_similarity_heatmap(
    features: Tensor,
    target_coords: list[tuple[int, int]],
    normalize: Sequence[str] = ["spatial"],
    combine_coords: bool = False,
) -> Tensor:
    """Compute cosine similarity heatmaps between target coordinates and all other tokens.

    Args:
        features: Input features with shape (n, h, w, d)
        target_coords: List of (row, col) coordinate pairs
        normalize: Which axes to normalize the output
        combine_coords: If True, combine multiple coordinates into single heatmap by averaging

    Returns:
        Cosine similarity heatmaps with shape (n, h, w, len(target_coords)) if combine_coords=False,
        or (n, h, w, 1) if combine_coords=True
    """
    n, h, w, d = features.shape
    features_flat = rearrange(features, "n h w d -> n (h w) d")

    # Normalize features for cosine similarity
    features_norm = F.normalize(features_flat, dim=-1)

    similarity_maps = []

    for row, col in target_coords:
        # Ensure coordinates are within bounds
        row = max(0, min(row, h - 1))
        col = max(0, min(col, w - 1))

        # Get target token index
        target_idx = row * w + col
        target_tokens = features_norm[:, target_idx, :]  # Shape: (n, d)

        # Compute cosine similarities
        similarities = torch.einsum("nd,nmd->nm", target_tokens, features_norm)  # Shape: (n, h*w)

        # Reshape back to spatial dimensions
        similarities = rearrange(similarities, "n (h w) -> n h w", h=h, w=w)
        similarity_maps.append(similarities)

    if combine_coords:
        # Average all similarity maps to combine multiple coordinates
        combined_heatmap = torch.stack(similarity_maps, dim=-1).mean(dim=-1, keepdim=True)  # Shape: (n, h, w, 1)
        heatmaps = combined_heatmap
    else:
        # Stack along last dimension (original behavior)
        heatmaps = torch.stack(similarity_maps, dim=-1)  # Shape: (n, h, w, num_targets)

    # Normalize to [0,1] range
    norm_axes = []
    if "batch" in normalize:
        norm_axes.append(0)
    if "channel" in normalize:
        norm_axes.append(3)
    if "spatial" in normalize:
        norm_axes += [1, 2]

    heatmaps = heatmaps - heatmaps.amin(dim=norm_axes, keepdim=True)
    heatmaps = heatmaps / (heatmaps.amax(dim=norm_axes, keepdim=True) + 1e-8)

    return heatmaps.float()


def load_image(path: Path, size: Sequence[int]) -> Tensor:
    """Load and preprocess an image file."""
    if path.suffix.lower() in (".tiff", ".tif"):
        img = load_tiff_f32(path)
        img = rearrange(img, "n h w c -> n c h w")
        img = torch.from_numpy(img)
        img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
        return img
    else:
        raise NotImplementedError(f"Unsupported image type: {path.suffix}")


def existing_file_type(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    return path


def output_path_type(value: str) -> Path:
    path = Path(value)
    if not path.parent.is_dir():
        raise NotADirectoryError(f"Output directory does not exist: {path.parent}")
    return path


def torch_dtype_type(value: str) -> torch.dtype:
    match value.lower():
        case "fp32":
            return torch.float32
        case "fp16":
            return torch.float16
        case "bf16":
            return torch.bfloat16
        case _:
            raise ValueError(f"Invalid data type: {value}")


def coordinate_type(value: str) -> tuple[int, int]:
    """Parse coordinate string in format 'row,col'."""
    try:
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError()
        row, col = int(parts[0]), int(parts[1])
        return (row, col)
    except (ValueError, IndexError):
        raise ValueError(f"Invalid coordinate format: {value}. Expected 'row,col'")


class ExpandImagePathsAction(Action):
    """Custom action that expands directory paths to .tiff files and validates all paths exist."""

    def __call__(
        self, parser: ArgumentParser, namespace: Namespace, values: Any, option_string: str | None = None
    ) -> None:
        if not isinstance(values, list):
            values = [values]

        expanded_paths: list[Path] = []
        for value in values:
            path = Path(value)
            if path.is_dir():
                tiff_files = list(path.glob("*.tiff"))
                if not tiff_files:
                    parser.error(f"No .tiff files found in directory: {path}")
                expanded_paths.extend(tiff_files)
            elif path.is_file():
                expanded_paths.append(path)
            else:
                parser.error(f"Input path does not exist: {path}")

        setattr(namespace, self.dest, expanded_paths)


@dataclass
class CosineVisualizer:
    """Visualizer for cosine similarity heatmaps between target tokens and all other tokens."""

    model: ViT
    target_coords: list[tuple[int, int]]
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    invert: bool = False
    zero: bool = False
    size: Sequence[int] | None = None
    normalize: Sequence[str] = field(default_factory=lambda: ["spatial"])
    no_output_norm: bool = False
    combine_coords: bool = False

    def __post_init__(self) -> None:
        if self.size is None:
            self.size = self.model.config.img_size
        self.model = self.model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        if self.no_output_norm:
            # Replace output norm with identity
            self.model.output_norm = nn.Identity()  # type: ignore
        torch.set_float32_matmul_precision("high")

    @torch.inference_mode()
    def _forward_features(self, img: Tensor) -> Tensor:
        """Extract features from the model."""
        H, W = img.shape[-2:]
        assert not self.model.training, "Model must be in evaluation mode"
        assert img.device == self.device, "Image must be on the same device as the model"
        with torch.autocast(self.device.type, dtype=self.dtype):
            features = cast(Tensor, self.model(img))
        features = features.to(torch.float32)
        tokenized_dims = self.model.stem.tokenized_size((H, W))  # Add batch dimension
        Ht, Wt = tokenized_dims[:2]  # Take first two dimensions
        features = rearrange(features, "n (ht wt) d -> n ht wt d", ht=Ht, wt=Wt)
        return features

    def _apply_colormap(self, x: Tensor) -> Tensor:
        """Apply colormap to heatmap for visualization."""
        x = torch.from_numpy(getattr(plt.cm, "viridis")(x.squeeze().cpu())[..., :3]).float()
        x = rearrange(x, "... h w c -> ... c h w").to(self.device)
        return x

    def _draw_red_box(self, img: Tensor, coords: tuple[int, int], token_size: tuple[float, float]) -> Tensor:
        """Draw a red box covering the entire token at the specified coordinate.

        Args:
            img: Image tensor with shape (n, c, h, w)
            coords: (row, col) token coordinate
            token_size: (token_h, token_w) size of each token in pixels (float values)

        Returns:
            Image with red box overlay
        """
        n, c, h, w = img.shape
        row, col = coords
        token_h, token_w = token_size

        # Calculate pixel boundaries for the entire token
        start_y = int(row * token_h)
        end_y = int((row + 1) * token_h)
        start_x = int(col * token_w)
        end_x = int((col + 1) * token_w)

        # Ensure boundaries are within image bounds
        start_y = max(0, start_y)
        end_y = min(h, end_y)
        start_x = max(0, start_x)
        end_x = min(w, end_x)

        # Create a copy to avoid modifying the original
        img_with_box = img.clone()

        # Fill the entire token area with red
        img_with_box[:, 0, start_y:end_y, start_x:end_x] = 1.0  # Red channel
        img_with_box[:, 1, start_y:end_y, start_x:end_x] = 0.0  # Green channel
        img_with_box[:, 2, start_y:end_y, start_x:end_x] = 0.0  # Blue channel

        return img_with_box

    def _compute_cosine_heatmaps(self, features: Tensor, output_size: Sequence[int]) -> Tensor:
        """Compute cosine similarity heatmaps for all target coordinates."""
        heatmaps = cosine_similarity_heatmap(features, self.target_coords, self.normalize, self.combine_coords)
        visualizations: list[Tensor] = []

        for i in range(heatmaps.shape[-1]):
            heatmap_i = heatmaps[..., i].unsqueeze_(1)
            heatmap_i = F.interpolate(heatmap_i, size=output_size, mode="nearest")
            heatmap_i = self._apply_colormap(heatmap_i)
            visualizations.append(heatmap_i)

        # Concatenate multiple heatmaps along the width axis
        return torch.cat(visualizations, dim=-1)

    def _create_grid(self, img: Tensor, heatmaps: Tensor, features_shape: tuple[int, ...], **kwargs) -> Tensor:
        """Create visualization grid with original image and heatmaps."""
        # Expand grayscale to RGB if needed
        img = img.expand(-1, 3, -1, -1)

        # Calculate token size in pixels
        h, w = img.shape[-2:]
        feature_h, feature_w = features_shape[:2]
        token_h = h / feature_h
        token_w = w / feature_w

        # Original image has no markers
        img_no_markers = img.clone()

        # Add red box markers to heatmaps - each heatmap gets only its corresponding coordinate marker
        heatmaps_with_markers = heatmaps.clone()

        # Split heatmaps along width to process each one individually
        # heatmaps has shape (n, c, h, total_w) where total_w = w * num_coords
        num_coords = len(self.target_coords)
        coord_width = heatmaps.shape[-1] // num_coords

        for i, coords in enumerate(self.target_coords):
            # Extract the specific heatmap for this coordinate
            start_idx = i * coord_width
            end_idx = (i + 1) * coord_width
            heatmap_slice = heatmaps_with_markers[:, :, :, start_idx:end_idx]

            # Add red box marker only to this specific heatmap
            heatmap_with_marker = self._draw_red_box(heatmap_slice, coords, (token_h, token_w))

            # Put it back
            heatmaps_with_markers[:, :, :, start_idx:end_idx] = heatmap_with_marker

        elements = torch.cat([img_no_markers, heatmaps_with_markers], dim=-1)
        kwargs.setdefault("nrow", 1)
        grid = make_grid(elements, **kwargs)
        return grid

    def _append_inverted_img(self, img: Tensor) -> Tensor:
        """Append inverted version of the image."""
        return torch.stack([img, 1 - img], dim=1).flatten(0, 1)

    def _append_zero_img(self, img: Tensor) -> Tensor:
        """Append zero image."""
        return torch.cat([img, torch.zeros_like(img[0, None])], dim=0)

    @torch.inference_mode()
    def __call__(self, img: Tensor) -> Tensor:
        """Generate cosine similarity visualization grid."""
        assert not self.model.training, "Model must be in evaluation mode"
        if self.invert:
            img = self._append_inverted_img(img)
        if self.zero:
            img = self._append_zero_img(img)
        img = img.to(self.device)
        img = F.interpolate(img, size=self.size, mode="bilinear", align_corners=False)

        features = self._forward_features(img)
        heatmaps = self._compute_cosine_heatmaps(features, img.shape[-2:])
        grid = self._create_grid(img, heatmaps, features.shape[1:3])
        return grid

    @torch.inference_mode()
    def save(self, grid: Tensor, output: Path) -> None:
        """Save the visualization grid to file."""
        save_image(grid, output)

    @classmethod
    def create_parser(cls: type[Self], custom_loader: bool = False) -> ArgumentParser:
        """Create argument parser with built-in validation."""
        parser = ArgumentParser(
            prog="cosine-visualize", description="Visualize cosine similarity heatmaps for ViT token features"
        )
        parser.add_argument("config", type=existing_file_type, help="Path to model YAML configuration file")
        parser.add_argument("checkpoint", type=existing_file_type, help="Path to safetensors checkpoint")
        parser.add_argument(
            "input",
            nargs="+",
            action=ExpandImagePathsAction if not custom_loader else "store",
            help="Path to input image(s) or directory containing .tiff files",
        )
        parser.add_argument("output", type=output_path_type, help="Path to output PNG file")
        parser.add_argument(
            "-c",
            "--coordinates",
            nargs="+",
            type=coordinate_type,
            help="Target coordinates in format 'row,col' (e.g., '5,10')",
        )
        parser.add_argument(
            "-s",
            "--size",
            type=int,
            nargs=2,
            default=None,
            help="Size of the input image. By default size is inferred from the model configuration.",
        )
        parser.add_argument("-d", "--device", default="cpu", type=torch.device, help="Device to run the model on")
        parser.add_argument("-i", "--invert", action="store_true", help="Also process an inverted image")
        parser.add_argument("-z", "--zero", action="store_true", help="Also process an all-zero image")
        parser.add_argument(
            "--normalize",
            choices=["batch", "channel", "spatial"],
            default=["spatial"],
            nargs="+",
            help="Which axes to normalize",
        )
        parser.add_argument(
            "-dt", "--dtype", default="fp32", type=torch_dtype_type, help="Data type to run the model on"
        )
        parser.add_argument(
            "--no-output-norm", action="store_true", help="Disable output normalization from the backbone"
        )
        parser.add_argument(
            "--combine-coords",
            action="store_true",
            help="Combine multiple coordinate pairs into single heatmap by averaging",
        )
        return parser

    @classmethod
    def from_args(cls: type[Self], args: Namespace) -> Self:
        """Create visualizer from command line arguments."""
        # Create model
        config = yaml.full_load(args.config.read_text())["backbone"]
        assert isinstance(config, ViTConfig)
        model = config.instantiate()

        # Load checkpoint
        state_dict = st.load_file(args.checkpoint)
        model.load_state_dict(state_dict)

        return cls(
            model,
            args.coordinates,
            args.device,
            args.dtype,
            args.invert,
            args.zero,
            args.size,
            args.normalize,
            args.no_output_norm,
            args.combine_coords,
        )


def main() -> None:
    """Main CLI entry point."""
    parser = CosineVisualizer.create_parser()
    args = parser.parse_args()

    visualizer = CosineVisualizer.from_args(args)

    # Process all input images
    all_grids: list[Tensor] = []
    for img_path in tqdm(args.input, desc="Processing images"):
        img = load_image(img_path, visualizer.size or (224, 224))
        grid = visualizer(img)
        all_grids.append(grid)

    # Combine all grids into a single visualization
    if len(all_grids) > 1:
        # Stack grids vertically for multiple images
        final_grid = torch.cat(all_grids, dim=-2)
    else:
        final_grid = all_grids[0]

    visualizer.save(final_grid, args.output)
    print(f"Cosine similarity visualization saved to {args.output}")


if __name__ == "__main__":
    main()
