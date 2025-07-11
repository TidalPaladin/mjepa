from argparse import Action, ArgumentParser, Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Self, Sequence, Type, cast

import matplotlib.pyplot as plt
import safetensors.torch as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dicom_preprocessing import load_tiff_f32
from einops import rearrange, reduce
from torch import Tensor
from torchvision.transforms.v2.functional import to_pil_image
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from vit import ViT, ViTConfig

from mjepa.jepa import register_constructors


register_constructors()


def pca_topk(features: Tensor, offset: int = 0, k: int = 3, normalize: Sequence[str] = ["spatial"]) -> Tensor:
    r"""Applies PCA to the features and returns the top k principal components.

    Args:
        features: Input features
        offset: Offset of the principal components to visualize
        k: Number of principal components to return

    Shapes:
        - features: :math:`(n, h, w, d)`
        - Output: :math:`(n, h, w, k)`

    Returns:
        Top k principal components
    """
    # Center the features
    features_mean = reduce(features, "n ht wt d -> d", "mean")
    features = features - features_mean

    # Standardize features by dividing by standard deviation
    features_std = reduce(features, "n ht wt d -> d", torch.std)
    features = features / (features_std + 1e-8)  # Add small epsilon to avoid division by zero

    # Reshape to 2D matrix for PCA
    n, h, w, d = features.shape
    features_2d = features.reshape(-1, d)

    # Compute covariance matrix and eigendecomposition
    cov = features_2d.T @ features_2d / (features_2d.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Get top k principal components
    topk_indices: Tensor = torch.argsort(eigenvalues, descending=True)[offset : offset + k]
    topk_components: Tensor = eigenvectors[:, topk_indices]

    # Project features onto top k components
    projected = features_2d @ topk_components
    projected = projected.reshape(n, h, w, k)

    # Normalize to [0,1] range for each component
    norm_axes = []
    if "batch" in normalize:
        norm_axes.append(0)
    if "channel" in normalize:
        norm_axes.append(3)
    if "spatial" in normalize:
        norm_axes += [1, 2]
    projected = projected - projected.amin(dim=norm_axes, keepdim=True)
    projected = projected / projected.amax(dim=norm_axes, keepdim=True)

    return projected.float()


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


class ExpandImagePathsAction(Action):
    """Custom action that expands directory paths to .tiff files and validates all paths exist."""

    def __call__(
        self, parser: ArgumentParser, namespace: Namespace, values: Any, option_string: Optional[str] = None
    ) -> None:
        if not isinstance(values, list):
            values = [values]

        expanded_paths: List[Path] = []
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
class PCAVisualizer:
    model: ViT
    device: torch.device = torch.device("cpu")
    num_components: int = 1
    offset: int = 0
    dtype: torch.dtype = torch.float32
    invert: bool = False
    zero: bool = False
    size: Sequence[int] | None = None
    animate: bool = False
    normalize: Sequence[str] = field(default_factory=lambda: ["spatial"])
    no_output_norm: bool = False

    def __post_init__(self) -> None:
        if self.size is None:
            self.size = self.model.config.img_size
        self.model = self.model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        if self.no_output_norm:
            self.model.output_norm = nn.Identity()
        torch.set_float32_matmul_precision("high")

    @torch.inference_mode()
    def _forward_features(self, img: Tensor) -> Tensor:
        H, W = img.shape[-2:]
        assert not self.model.training, "Model must be in evaluation mode"
        assert img.device == self.device, "Image must be on the same device as the model"
        with torch.autocast(self.device.type, dtype=self.dtype):
            features = cast(Tensor, self.model(img))
        features = features.to(torch.float32)
        Ht, Wt = self.model.stem.tokenized_size((H, W))
        features = rearrange(features, "n (ht wt) d -> n ht wt d", ht=Ht, wt=Wt)
        return features

    def _apply_colormap(self, x: Tensor) -> Tensor:
        x = torch.from_numpy(plt.cm.inferno(x.squeeze().cpu())[..., :3]).float()
        x = rearrange(x, "... h w c -> ... c h w").to(self.device)
        return x

    def _compute_pca(self, features: Tensor, output_size: Sequence[int]) -> Tensor:
        pca = pca_topk(features, self.offset, k=self.num_components, normalize=self.normalize)
        visualizations: List[Tensor] = []
        for i in range(pca.shape[-1]):
            pca_i = pca[..., i].unsqueeze_(1)
            pca_i = F.interpolate(pca_i, size=output_size, mode="nearest")
            pca_i = self._apply_colormap(pca_i)
            visualizations.append(pca_i)

        # Concatenate multiple components along the width axis
        return torch.cat(visualizations, dim=-1)

    def _create_grid(self, img: Tensor, pca: Tensor, **kwargs) -> Tensor:
        img = img.expand(-1, 3, -1, -1)
        elements = torch.cat([img, pca], dim=-1)
        kwargs.setdefault("nrow", 1)
        grid = make_grid(elements, **kwargs)
        return grid

    def _append_inverted_img(self, img: Tensor) -> Tensor:
        return torch.stack([img, 1 - img], dim=1).flatten(0, 1)

    def _append_zero_img(self, img: Tensor) -> Tensor:
        return torch.cat([img, torch.zeros_like(img[0, None])], dim=0)

    @torch.inference_mode()
    def __call__(self, img: Tensor) -> Tensor:
        assert not self.model.training, "Model must be in evaluation mode"
        if self.invert:
            img = self._append_inverted_img(img)
        if self.zero:
            img = self._append_zero_img(img)
        img = img.to(self.device)
        img = F.interpolate(img, size=self.size, mode="bilinear", align_corners=False)

        if self.animate:
            feature_list: List[Tensor] = []
            for _img in tqdm(img, desc="Animating"):
                # Compute zoom window boundaries that shrink towards center
                features = self._forward_features(_img[None])
                feature_list.append(features)

            pca = self._compute_pca(torch.cat(feature_list, dim=0), self.size).chunk(img.shape[0], dim=0)
            grids: List[Tensor] = [self._create_grid(i[None], p) for i, p in zip(img, pca)]
            return torch.stack(grids, dim=0)
        else:
            features = self._forward_features(img)
            pca = self._compute_pca(features, img.shape[-2:])
            grid = self._create_grid(img, pca)
            return grid

    @torch.inference_mode()
    def save(self, grid: Tensor, output: Path) -> None:
        if self.animate:
            # Convert to uint8 and convert to PIL images
            grid = grid.mul(255).clip_(0, 255).to(torch.uint8).cpu()
            frames = [to_pil_image(frame) for frame in grid]

            # Save as animated GIF using PIL
            output = output.with_suffix(".gif")
            frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
        else:
            save_image(grid, output)

    @classmethod
    def create_parser(cls: Type[Self], custom_loader: bool = False) -> ArgumentParser:
        """Create argument parser with built-in validation."""
        parser = ArgumentParser(prog="pca-visualize", description="Visualize PCA ViT output features")
        parser.add_argument("config", type=existing_file_type, help="Path to model YAML configuration file")
        parser.add_argument("checkpoint", type=existing_file_type, help="Path to safetensors checkpoint")
        parser.add_argument(
            "input",
            nargs="+",
            action=ExpandImagePathsAction if not custom_loader else None,
            help="Path to input image(s) or directory containing .tiff files",
        )
        parser.add_argument("output", type=output_path_type, help="Path to output PNG file")
        parser.add_argument(
            "-s",
            "--size",
            type=int,
            nargs=2,
            default=None,
            help="Size of the input image. By default size is inferred from the model configuration.",
        )
        parser.add_argument(
            "-a", "--animate", action="store_true", help="Treat each image as a separate animation frame"
        )
        parser.add_argument("-d", "--device", default="cpu", type=torch.device, help="Device to run the model on")
        parser.add_argument(
            "-o", "--offset", type=int, default=0, help="Offset of the principal components to visualize"
        )
        parser.add_argument("-i", "--invert", action="store_true", help="Also process an inverted image")
        parser.add_argument("-z", "--zero", action="store_true", help="Also process an all-zero image")
        parser.add_argument(
            "--normalize", choices=["batch", "channel", "spatial"], default=["spatial"], help="Which axes to normalize"
        )
        parser.add_argument(
            "-m", "--mode", choices=["rgb", "single"], default="rgb", help="Mode to visualize the principal components"
        )
        parser.add_argument(
            "-n",
            "--num-components",
            type=int,
            default=1,
            help="Number of principal component groups to visualize",
        )
        parser.add_argument(
            "-dt", "--dtype", default="fp32", type=torch_dtype_type, help="Data type to run the model on"
        )
        parser.add_argument(
            "--no-output-norm", action="store_true", help="Disable output normalization from the backbone"
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
            args.num_components,
            args.offset,
            args.dtype,
            args.invert,
            args.zero,
            args.size,
            args.animate,
            args.normalize,
            args.no_output_norm,
        )
