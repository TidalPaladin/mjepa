from argparse import Namespace

import torch

from mjepa.visualization.cosine import CosineVisualizer, load_image


def parse_args() -> Namespace:
    parser = CosineVisualizer.create_parser()
    return parser.parse_args()


def main(args: Namespace) -> None:
    visualizer = CosineVisualizer.from_args(args)

    # Load images
    imgs = []
    for image in args.input:
        img = load_image(image, visualizer.size)
        imgs.append(img)
    img = torch.cat(imgs, dim=0)

    result = visualizer(img)
    print("Saving to", args.output)
    visualizer.save(result, args.output)


def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
