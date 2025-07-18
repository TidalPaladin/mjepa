from argparse import Namespace

from mjepa.visualization.pos_enc import PositionVisualizer


def parse_args() -> Namespace:
    parser = PositionVisualizer.create_parser()
    return parser.parse_args()


def main(args: Namespace) -> None:
    visualizer = PositionVisualizer.from_args(args)
    result = visualizer()
    print("Saving to", args.output)
    visualizer.save(result, args.output)


def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
