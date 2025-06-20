from argparse import Namespace

import torch
from einops import rearrange
from torch import Tensor
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import Compose, ToDtype, ToImage

from mjepa.visualization.pca import PCAVisualizer


def load_image(dataset: CIFAR10, index: int) -> Tensor:
    img, _ = dataset[index]
    return rearrange(img, "c h w -> () c h w")


def parse_args() -> Namespace:
    parser = PCAVisualizer.create_parser(custom_loader=True)
    return parser.parse_args()


def main(args: Namespace) -> None:
    visualizer = PCAVisualizer.from_args(args)

    # Load images
    imgs = []
    transform = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    dataset = CIFAR10(args.input[0], download=True, train=False, transform=transform)
    for i in range(8):
        img = load_image(dataset, i)
        imgs.append(img)
    img = torch.cat(imgs, dim=0)

    result = visualizer(img)
    visualizer.save(result, args.output)


def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
