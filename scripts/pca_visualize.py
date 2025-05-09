import torch
import yaml
import safetensors.torch as st
import torch.nn.functional as F
from dataclasses import replace
from typing import Tuple, cast
from argparse import ArgumentParser, Namespace
from vit import ViTConfig, ViT
from pathlib import Path
from torch import Tensor
from dicom_preprocessing import load_tiff_f32
from einops import rearrange, reduce
from PIL import Image
from torchvision.utils import make_grid, save_image
from torchvision.transforms.v2.functional import center_crop

from mjepa import *


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
    top3_indices: Tensor = torch.argsort(eigenvalues, descending=True)[offset:offset+3]
    top3_components: Tensor = eigenvectors[:, top3_indices]

    # Project features onto top 3 components
    projected = features_2d @ top3_components
    projected = projected.reshape(n, h, w, 3)

    # Normalize to [0,1] range for each component
    projected = projected - projected.amin(dim=(1,2, 3), keepdim=True)[0]
    projected = projected / projected.amax(dim=(1,2, 3), keepdim=True)[0]

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
    img = torch.cat(imgs, dim=0)

    # Load model and move to device
    config = yaml.full_load(args.config.read_text())["backbone"]
    assert isinstance(config, ViTConfig)
    if config.backend == "te" and device.type == "cpu":
        print("Warning: TE backend requires GPU. Switching to torch backend.")
        config = replace(config, backend="torch")
    model = config.instantiate()
    model = model.to(device)
    model.eval()

    # Load checkpoint
    state_dict = st.load_file(args.checkpoint)
    if config.backend == "pytorch":
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith("_extra_state")}
    model.load_state_dict(state_dict)

    # Run forward pass
    H, W = img.shape[-2:]
    features, _, registers = cast(Tuple[Tensor, Tensor | None, Tensor | None], model(img))
    Ht, Wt = model.stem.tokenized_size((H, W))
    features = rearrange(features, "n (ht wt) d -> n ht wt d", ht=Ht, wt=Wt)

    # Compute PCA and scale to [0,255] range
    if args.raw:
        pca: Tensor = rearrange(features[..., args.offset:args.offset+3], "n h w c -> n c h w")
    else:
        pca: Tensor = pca_top3(features, args.offset)
    pca = F.interpolate(pca, size=(H, W), mode="nearest")

    # Save original image and PCA image side by side
    img = img.expand_as(pca)
    elements = torch.cat([img, pca], dim=-1)
    grid = make_grid(elements, nrow=2)
    save_image(grid, args.output)

    
def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
    
    
    
