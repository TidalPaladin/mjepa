from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple, cast

import matplotlib.pyplot as plt
import safetensors.torch as st
import torch
import torch.nn.functional as F
import yaml
from dicom_preprocessing import load_tiff_f32
from einops import rearrange
from torch import Tensor
from torchvision.transforms.v2.functional import center_crop
from vit import ViTConfig

from mjepa.jepa import register_constructors


register_constructors()


def load_image(path: Path, size: Tuple[int, int], crop: bool = False) -> Tensor:
    if path.suffix.lower() in (".tiff", ".tif"):
        img = load_tiff_f32(path)
        img = rearrange(img, "n h w c -> n c h w")
        img = torch.from_numpy(img)
        if crop:
            img = center_crop(img, size)
        else:
            img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
        return img
    else:
        raise NotImplementedError(f"Unsupported image type: {path.suffix}")


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="pca-visualize", description="Visualize PCA ViT output features")
    parser.add_argument("config", type=Path, help="Path to model YAML configuration file")
    parser.add_argument("checkpoint", type=Path, help="Path to safetensors checkpoint")
    parser.add_argument("image", type=Path, nargs="+", help="Path to input image")
    parser.add_argument("output", type=Path, help="Path to output plot")
    parser.add_argument("-d", "--device", default="cpu", help="Device to run the model on")
    parser.add_argument("-s", "--size", type=int, nargs=2, default=(512, 384), help="Image size")
    return parser.parse_args()


def main(args: Namespace) -> None:
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if len(args.image) == 1 and args.image[0].is_dir():
        args.image = list(args.image[0].glob("*.tiff"))
    if not all(image.is_file() for image in args.image):
        raise FileNotFoundError(args.image)
    if not args.output.parent.is_dir():
        raise NotADirectoryError(args.output.parent)
    device = torch.device(args.device)

    # Load images
    imgs = []
    for image in args.image:
        img = load_image(image, args.size)
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

    # Run forward pass
    H, W = img.shape[-2:]
    with torch.autocast(device.type, dtype=torch.bfloat16), torch.no_grad():
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
