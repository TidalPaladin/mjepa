from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Self, Sequence, Type, cast

import matplotlib.pyplot as plt
import safetensors.torch as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from vit import ViT, ViTConfig

from mjepa.jepa import register_constructors

from .pca import ExpandImagePathsAction, existing_file_type, output_path_type, torch_dtype_type


register_constructors()


def create_norm_histogram(features: Tensor, register_tokens: Tensor) -> plt.Figure:
    # Compute L2 norms along the feature dimension
    feature_norms = torch.norm(features, p=2, dim=-1)  # Shape: (n, num_tokens)
    register_norms = torch.norm(register_tokens, p=2, dim=-1)  # Shape: (n, num_register_tokens)

    # Flatten to get all norm values
    all_feature_norms = feature_norms.flatten().cpu().numpy()
    all_register_norms = register_norms.flatten().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(
        [all_feature_norms, all_register_norms],
        bins=50,
        alpha=0.6,
        label=["Features", "Register Tokens"],
        color=["blue", "red"],
    )
    plt.xlabel("L2 Norm")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Distribution of L2 Norms")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save the plot
    plt.tight_layout()
    return plt.gcf()


@dataclass
class NormVisualizer:
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
        self.model.output_norm = nn.Identity()
        torch.set_float32_matmul_precision("high")

    @torch.inference_mode()
    def _forward_features(self, img: Tensor) -> Tensor:
        H, W = img.shape[-2:]
        assert not self.model.training, "Model must be in evaluation mode"
        assert img.device == self.device, "Image must be on the same device as the model"
        with torch.autocast(self.device.type, dtype=self.dtype):
            features = cast(Tensor, self.model(img, return_register_tokens=True))
        features = features.to(torch.float32)
        return features

    @torch.inference_mode()
    def __call__(self, img: Tensor) -> plt.Figure:
        assert not self.model.training, "Model must be in evaluation mode"
        img = img.to(self.device)
        img = F.interpolate(img, size=self.size, mode="bilinear", align_corners=False)
        features = self._forward_features(img)
        register_tokens = features[..., : self.model.config.num_register_tokens, :]
        features = features[..., self.model.config.num_register_tokens :, :]
        return create_norm_histogram(features, register_tokens)

    @torch.inference_mode()
    def save(self, fig: plt.Figure, output: Path) -> None:
        fig.savefig(output, dpi=300, bbox_inches="tight")

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
