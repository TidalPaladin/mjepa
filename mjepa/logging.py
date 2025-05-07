import os
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torchvision.utils import make_grid, save_image

from .trainer import rank_zero_info, rank_zero_only


class CSVLogger:
    def __init__(self, path: os.PathLike, interval: int = 1, accumulate_grad_batches: int = 1):
        path = Path(path)
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Parent directory {path.parent} does not exist")
        self.path = path
        self.interval = interval
        self.accumulate_grad_batches = accumulate_grad_batches

        # Create the file with headers if it doesn't exist
        if not self.path.exists():
            self.path.touch()

    def log(self, epoch: int, step: int, microbatch: int, **kwargs):
        if (step + 1) % self.interval != 0 or (microbatch + 1) % self.accumulate_grad_batches != 0:
            return

        # Create a single-row DataFrame for this log entry
        kwargs["step"] = step
        kwargs["epoch"] = epoch
        df = pd.DataFrame([kwargs])

        # Append to CSV file, with header only if file is empty
        file_empty = os.path.getsize(self.path) == 0
        df.to_csv(self.path, mode="a", header=file_empty, index=False)

    def get_df(self):
        """Read the entire log file into a DataFrame"""
        if not self.path.exists() or os.path.getsize(self.path) == 0:
            return pd.DataFrame()
        return pd.read_csv(self.path)


class SaveImage:
    def __init__(self, path: os.PathLike, max_save_images: int = 8):
        path = Path(path)
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Parent directory {path.parent} does not exist")
        self.path = path
        self.max_save_images = max_save_images

    @rank_zero_only
    def __call__(self, x: Tensor) -> None:
        rank_zero_info(f"Saving batch to {self.path}")
        with torch.no_grad():
            grid = make_grid(x[: self.max_save_images], nrow=4)
            save_image(grid, self.path)
