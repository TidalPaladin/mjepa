from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import safetensors.torch as st
import torch
import torch.nn as nn
import yaml
from einops import rearrange
from torch import Tensor
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import Compose, ToDtype, ToImage
from vit import ViTConfig

from mjepa.jepa import register_constructors


register_constructors()


def load_image(dataset: CIFAR10, index: int) -> Tensor:
    img, _ = dataset[index]
    return rearrange(img, "c h w -> () c h w")


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="cifar10-norms", description="Visualize ViT output feature L2 norms for CIFAR-10")
    parser.add_argument("config", type=Path, help="Path to model YAML configuration file")
    parser.add_argument("checkpoint", type=Path, help="Path to safetensors checkpoint")
    parser.add_argument("path", type=Path, help="Path to CIFAR10 dataset")
    parser.add_argument("output", type=Path, help="Path to output plot")
    parser.add_argument("-d", "--device", default="cpu", help="Device to run the model on")
    parser.add_argument("-s", "--size", type=int, nargs=2, default=(512, 384), help="Image size")
    parser.add_argument("-p", "--pre-norm", action="store_true", help="Use pre-norm outputs (for ViT with output norm)")
    return parser.parse_args()


def main(args: Namespace) -> None:
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.path.is_dir():
        raise NotADirectoryError(args.path)
    if not args.output.parent.is_dir():
        raise NotADirectoryError(args.output.parent)
    device = torch.device(args.device)

    # Load images
    imgs = []
    transform = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    dataset = CIFAR10(args.path, download=True, train=False, transform=transform)
    for i in range(4):
        img = load_image(dataset, i)
        img = img.to(device)
        imgs.append(img)
    img = torch.cat(imgs, dim=0)

    # Load model and move to device
    config = yaml.full_load(args.config.read_text())["backbone"]
    assert isinstance(config, ViTConfig)
    model = config.instantiate()
    model = model.to(device)
    model.eval()

    # Load checkpoint
    state_dict = st.load_file(args.checkpoint)
    model.load_state_dict(state_dict)

    # Replace output norm with identity if needed
    if args.pre_norm:
        model.output_norm = nn.Identity()

    # Run forward pass
    H, W = img.shape[-2:]
    with torch.autocast(device.type, dtype=torch.bfloat16), torch.inference_mode():
        features = cast(Tensor, model(img))
    features = features.to(torch.float32)

    # Compute L2 norms along the feature dimension
    norms = torch.norm(features, p=2, dim=-1)  # Shape: (n, num_tokens)

    # Flatten to get all norm values
    all_norms = norms.flatten().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(all_norms, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("L2 Norm")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Distribution of Feature L2 Norms")
    plt.grid(True, alpha=0.3)

    # Add statistics to the plot
    mean_norm = all_norms.mean()
    std_norm = all_norms.std()
    plt.axvline(mean_norm, color="red", linestyle="--", label=f"Mean: {mean_norm:.3f}")
    plt.axvline(
        mean_norm + std_norm, color="orange", linestyle="--", alpha=0.7, label=f"Mean + Std: {mean_norm + std_norm:.3f}"
    )
    plt.axvline(
        mean_norm - std_norm, color="orange", linestyle="--", alpha=0.7, label=f"Mean - Std: {mean_norm - std_norm:.3f}"
    )
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Norm statistics:")
    print(f"  Mean: {mean_norm:.6f}")
    print(f"  Std:  {std_norm:.6f}")
    print(f"  Min:  {all_norms.min():.6f}")
    print(f"  Max:  {all_norms.max():.6f}")
    print(f"Plot saved to: {args.output}")


def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
