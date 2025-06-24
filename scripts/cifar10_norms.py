from argparse import Namespace
from typing import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import Compose, ToDtype, ToImage

from mjepa.visualization.norm import NormVisualizer


def load_image(dataset: CIFAR10, index: int, size: Sequence[int]) -> Tensor:
    img, _ = dataset[index]
    img = rearrange(img, "c h w -> () c h w")
    img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
    return img


def parse_args() -> Namespace:
    parser = NormVisualizer.create_parser(custom_loader=True)
    return parser.parse_args()


def main(args: Namespace) -> None:
    visualizer = NormVisualizer.from_args(args)

    # Load images
    imgs = []
    transform = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    dataset = CIFAR10(args.input[0], download=True, train=False, transform=transform)
    for i in range(8):
        img = load_image(dataset, i, visualizer.size)
        imgs.append(img)
    img = torch.cat(imgs, dim=0)

    result = visualizer(img)
    visualizer.save(result, args.output)


def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
