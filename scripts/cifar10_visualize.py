from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, cast

import safetensors.torch as st
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange, reduce
from torch import Tensor
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import Compose, ToDtype, ToImage
from torchvision.utils import make_grid, save_image
from vit import ViTConfig


def load_image(dataset: CIFAR10, index: int) -> Tensor:
    img, _ = dataset[index]
    return rearrange(img, "c h w -> () c h w")


def pca_top3(features: Tensor, offset: int = 0) -> Tensor:
    r"""Find the top 3 principal components of the features and return the corresponding RGB image.

    Args:
        features: Input features

    Shapes:
        - features: :math:`(n, h, w, d)`
        - Output: :math:`(n, 3, h, w)`

    Returns:
        RGB image of the top 3 principal components
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

    # Get top 3 principal components
    top3_indices: Tensor = torch.argsort(eigenvalues, descending=True)[offset : offset + 3]
    top3_components: Tensor = eigenvectors[:, top3_indices]

    # Project features onto top 3 components
    projected = features_2d @ top3_components
    projected = projected.reshape(n, h, w, 3)

    # Normalize to [0,1] range for each component
    projected = projected - projected.amin(dim=(1, 2, 3), keepdim=True)[0]
    projected = projected / projected.amax(dim=(1, 2, 3), keepdim=True)[0]

    # Rearrange to NCHW format
    projected = rearrange(projected, "n h w c -> n c h w")

    return projected


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="pca-visualize", description="Visualize PCA ViT output features")
    parser.add_argument("config", type=Path, help="Path to model YAML configuration file")
    parser.add_argument("checkpoint", type=Path, help="Path to safetensors checkpoint")
    parser.add_argument("path", type=Path, help="Path to CIFAR10 dataset")
    parser.add_argument("output", type=Path, help="Path to output PNG file")
    parser.add_argument("-d", "--device", default="cpu", help="Device to run the model on")
    parser.add_argument("-s", "--size", type=int, nargs=2, default=(512, 384), help="Image size")
    parser.add_argument("-c", "--crop", action="store_true", help="Crop image to model input size instead of resizing")
    parser.add_argument("-o", "--offset", type=int, default=0, help="Offset of the principal components to visualize")
    parser.add_argument("-r", "--raw", action="store_true", help="Do not apply PCA, visualize raw features")
    parser.add_argument(
        "-n",
        "--num-components",
        type=int,
        default=1,
        help="Number of 3-channel principal component groups to visualize",
    )
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

    # Run forward pass
    H, W = img.shape[-2:]
    with torch.autocast(device.type, dtype=torch.bfloat16):
        features = cast(Tensor, model(img))
    features = features.to(torch.float32)
    Ht, Wt = model.stem.tokenized_size((H, W))
    features = rearrange(features, "n (ht wt) d -> n ht wt d", ht=Ht, wt=Wt)
    features = F.layer_norm(features, features.shape[-1:])

    # Compute PCA and scale to [0,255] range
    pca_tensors: List[Tensor] = []
    for i in range(args.num_components):
        if args.raw:
            pca: Tensor = rearrange(
                features[..., args.offset + i * 3 : args.offset + (i + 1) * 3], "n h w c -> n c h w"
            )
        else:
            pca: Tensor = pca_top3(features, args.offset + i * 3)
        pca = F.interpolate(pca, size=(H, W), mode="nearest")
        pca_tensors.append(pca)
    pca = torch.cat(pca_tensors, dim=-1)

    # Save original image and PCA image side by side
    img = img.expand(-1, 3, -1, -1)
    elements = torch.cat([img, pca], dim=-1)
    grid = make_grid(elements, nrow=1)
    save_image(grid, args.output)


def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
