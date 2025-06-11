from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import time
from typing import List, Tuple, cast

import matplotlib.pyplot as plt
import safetensors.torch as st
import torch
import torch.nn.functional as F
import yaml
from dicom_preprocessing import load_tiff_f32
from einops import rearrange, reduce
from torch import Tensor
from torchvision.transforms.v2.functional import center_crop
from torchvision.utils import make_grid, save_image
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


def pca_topk(features: Tensor, offset: int = 0, k: int = 3) -> Tensor:
    r"""Find the top 3 principal components of the features and return the corresponding RGB image.

    Args:
        features: Input features
        offset: Offset of the principal components to visualize
        k: Number of principal components to visualize

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

    # Get top k principal components
    topk_indices: Tensor = torch.argsort(eigenvalues, descending=True)[offset : offset + k]
    topk_components: Tensor = eigenvectors[:, topk_indices]

    # Project features onto top k components
    projected = features_2d @ topk_components
    projected = projected.reshape(n, h, w, k)

    # Normalize to [0,1] range for each component
    projected = projected - projected.amin()
    projected = projected / projected.amax()

    # Rearrange to NCHW format
    projected = rearrange(projected, "n h w c -> n c h w")

    return projected


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="pca-visualize", description="Visualize PCA ViT output features")
    parser.add_argument("config", type=Path, help="Path to model YAML configuration file")
    parser.add_argument("checkpoint", type=Path, help="Path to safetensors checkpoint")
    parser.add_argument("image", type=Path, nargs="+", help="Path to input image")
    parser.add_argument("output", type=Path, help="Path to output PNG file")
    parser.add_argument("-d", "--device", default="cpu", help="Device to run the model on")
    parser.add_argument("-s", "--size", type=int, nargs=2, default=(512, 384), help="Image size")
    parser.add_argument("-c", "--crop", action="store_true", help="Crop image to model input size instead of resizing")
    parser.add_argument("-o", "--offset", type=int, default=0, help="Offset of the principal components to visualize")
    parser.add_argument("-r", "--raw", action="store_true", help="Do not apply PCA, visualize raw features")
    parser.add_argument("-i", "--invert", action="store_true", help="Also process an inverted image")
    parser.add_argument("-z", "--zero", action="store_true", help="Also process an all-zero image")
    parser.add_argument(
        "-m", "--mode", choices=["rgb", "single"], default="rgb", help="Mode to visualize the principal components"
    )
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
        img = load_image(image, args.size, args.crop)
        img = img.to(device)
        imgs.append(img)
        if args.invert:
            imgs.append(1 - img)
    if args.zero:
        imgs.append(torch.zeros_like(imgs[0]))
    img = torch.cat(imgs, dim=0)

    # Load model and move to device
    config = yaml.full_load(args.config.read_text())["backbone"]
    assert isinstance(config, ViTConfig)
    model = config.instantiate()
    model = model.to(device)
    model.requires_grad_(False)
    model.eval()

    # Load checkpoint
    state_dict = st.load_file(args.checkpoint)
    model.load_state_dict(state_dict)

    # Run forward pass
    H, W = img.shape[-2:]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"), torch.inference_mode():
        start = time()
        features = cast(Tensor, model(img))
        end = time()
        torch.cuda.synchronize()
        print(f"Forward pass took {end - start:.2f} seconds")
    features = features.to(torch.float32)
    Ht, Wt = model.stem.tokenized_size((H, W))
    features = rearrange(features, "n (ht wt) d -> n ht wt d", ht=Ht, wt=Wt)

    # Compute PCA and scale to [0,255] range
    pca_tensors: List[Tensor] = []
    k = 3 if args.mode == "rgb" else 1
    for i in range(args.num_components):
        if args.raw:
            pca: Tensor = rearrange(
                features[..., args.offset + i * k : args.offset + (i + 1) * k], "n h w c -> n c h w"
            )
        else:
            pca: Tensor = pca_topk(features, args.offset + i * k, k=k)
        pca = F.interpolate(pca, size=(H, W), mode="nearest")

        # Apply colormap when k=1
        if k == 1:
            pca = torch.from_numpy(plt.cm.inferno(pca.squeeze().cpu())[..., :3])
            pca = rearrange(pca, "n h w c -> n c h w").to(device)

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
