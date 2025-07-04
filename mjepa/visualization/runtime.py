from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, Final, List, Self, Sequence, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from vit import ViT, ViTConfig

from mjepa.jepa import register_constructors

from .pca import existing_file_type, output_path_type, torch_dtype_type


NUM_WARMUP_RUNS: Final = 3

register_constructors()


def plot_times(times: Dict[Sequence[int], List[float]], device: torch.device, batch_size: int) -> plt.Figure:
    """Create a box plot comparing inference times for different input sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Check if all times are under 1 second
    all_times = [t for time_list in times.values() for t in time_list]
    use_ms = all(t < 1.0 for t in all_times)

    # Convert to ms if needed
    data = []
    for size in times.keys():
        if use_ms:
            data.append([t * 1000 for t in times[size]])
        else:
            data.append(times[size])

    # Convert sizes to labels and plot boxes
    labels = [f"{h}x{w}" for h, w in times.keys()]
    medians = [np.median(d) for d in data]
    latency_line = ax.plot(range(1, len(data) + 1), medians, "o-", label="Latency")[0]
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(labels)

    # Customize plot
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    ax.set_title(f"Model Inference Time by Input Size ({device_name}, Batch Size = {batch_size})")
    ax.set_xlabel("Input Resolution")
    ax.set_ylabel("Time (milliseconds)" if use_ms else "Time (seconds)", color="b")
    ax.grid(True, alpha=0.3)

    # Add second y-axis for throughput
    ax2 = ax.twinx()
    throughput = [batch_size / (np.median(d) / 1000 if use_ms else np.median(d)) for d in data]
    throughput_line = ax2.plot(range(1, len(data) + 1), throughput, "o--", color="r", label="Throughput")[0]
    ax2.set_ylabel("Throughput (Images/s)", color="r")
    ax2.tick_params(axis="y")

    # Add legend
    lines = [latency_line, throughput_line]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="center left")

    plt.tight_layout()

    return fig


@dataclass
class RuntimeVisualizer:
    model: ViT
    sizes: Sequence[Sequence[int]]
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    num_runs: int = 10
    batch_size: int = 1

    def __post_init__(self) -> None:
        self.model = self.model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        torch.set_float32_matmul_precision("high")

    @torch.inference_mode()
    def _get_runtime(self, size: Sequence[int]) -> List[float]:
        B, C = self.batch_size, self.model.config.in_channels
        x = torch.randn(B, C, *size, device=self.device)
        times: List[float] = []
        with torch.autocast(self.device.type, dtype=self.dtype):
            # First pass for warmup/compile
            for _ in range(NUM_WARMUP_RUNS):
                self.model(x)
            torch.cuda.synchronize()

            for _ in range(self.num_runs):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                start = time()
                self.model(x)
                torch.cuda.synchronize()
                times.append(time() - start)
        return times

    @torch.inference_mode()
    def __call__(self) -> plt.Figure:
        assert not self.model.training, "Model must be in evaluation mode"
        times = {size: self._get_runtime(size) for size in self.sizes}
        return plot_times(times, self.device, self.batch_size)

    @torch.inference_mode()
    def save(self, fig: plt.Figure, output: Path) -> None:
        fig.savefig(output, dpi=300, bbox_inches="tight")

    @classmethod
    def create_parser(cls: Type[Self], custom_loader: bool = False) -> ArgumentParser:
        """Create argument parser with built-in validation."""
        parser = ArgumentParser(prog="pca-visualize", description="Visualize PCA ViT output features")
        parser.add_argument("config", type=existing_file_type, help="Path to model YAML configuration file")
        parser.add_argument("output", type=output_path_type, help="Path to output PNG file")
        parser.add_argument(
            "-s",
            "--size",
            type=int,
            nargs=2,
            default=None,
            help="Size of the input image. By default size is inferred from the model configuration.",
        )
        parser.add_argument(
            "--scales",
            type=float,
            nargs="+",
            default=[0.25, 0.5, 1.0, 2.0, 4.0],
            help="Scales to benchmark. By default scales are [0.25, 0.5, 1.0, 2.0, 4.0].",
        )
        parser.add_argument(
            "-n",
            "--num-runs",
            type=int,
            default=10,
            help="Number of runs to average the runtime over. By default num_runs is 10.",
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=1,
            help="Batch size to run the model on. By default batch size is 1.",
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

        # Get sizes
        sizes = [(int(config.img_size[0] * scale), int(config.img_size[1] * scale)) for scale in args.scales]

        return cls(
            model,
            sizes,
            args.device,
            args.dtype,
            args.num_runs,
            args.batch_size,
        )
