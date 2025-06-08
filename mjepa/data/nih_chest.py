from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Image as TVImage


FINDINGS = {
    "No Finding": 0,
    "Nodule": 1,
    "Pneumothorax": 2,
    "Mass": 3,
    "Consolidation": 4,
    "Hernia": 5,
    "Pleural_Thickening": 6,
    "Edema": 7,
    "Emphysema": 8,
    "Effusion": 9,
    "Fibrosis": 10,
    "Atelectasis": 11,
    "Pneumonia": 12,
    "Infiltration": 13,
    "Cardiomegaly": 14,
}


class NIHChestDataset(Dataset):
    r"""Dataset for NIH Chest X-ray images.

    Args:
        root: Path to the root directory of the dataset.
        train: Whether to use the training set or the test set
        transform: Optional transform to apply to the images.

    Examples are 2-tuples of:
        * Image (torchvision.tv_tensors.Image), zero-one normalized with shape :math:`(1, H, W)`
        * Dictionary of labels with keys giving the finding name and values giving a binary label
          for the presence of the finding. There is a 'No Finding' key with a value of 0.0 if the
          example does not contain any finding.
    """

    def __init__(
        self,
        root: Path,
        train: bool = False,
        transform: Transform | None = None,
    ):
        super().__init__()
        self.transform = transform
        if not root.is_dir():
            raise NotADirectoryError(root)
        self.root = root

        # Images are in a nested directory structure with no apparent linkage in the metadata CSV.
        # Create a flat set of symlinks that can be referenced by the metadata CSV.
        if not (symlink_dir := root / "images").is_dir():
            print(f"Creating symlinks in {symlink_dir}")
            symlink_dir.mkdir(parents=True, exist_ok=True)
            image_dirs = [p for p in root.glob("images_*") if p.is_dir()]
            assert image_dirs, "No image directories found"
            for image_dir in image_dirs:
                for image_file in image_dir.rglob("*.png"):
                    target = symlink_dir / image_file.name
                    target.symlink_to(image_file)

        if train:
            files = open(root / "train_val_list.txt", "r").read().splitlines()
        else:
            files = open(root / "test_list.txt", "r").read().splitlines()

        self.metadata = pd.read_csv(root / "Data_Entry_2017.csv", index_col="Image Index")
        self.metadata = self.metadata[self.metadata.index.isin(files)]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[TVImage, Dict[str, Tensor]]:
        metadata = self.metadata.iloc[index]

        # Load image (0-1 normalized float32)
        image_path = metadata.name
        image = Image.open(self.root / "images" / image_path).convert("F")
        image = torch.from_numpy(np.array(image)).div_(255).unsqueeze_(0)
        image = TVImage(image)

        # Load labels (binary labels for each finding, float32)
        label = {}
        findings = set(metadata["Finding Labels"].split("|"))
        for name, _ in FINDINGS.items():
            label[name] = torch.tensor(int(name in findings), dtype=torch.float32)

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--train", action="store_true")
    return parser.parse_args()


def main(args: Namespace) -> None:
    dataset = NIHChestDataset(args.root, args.train)
    print(len(dataset))
    print(dataset[0])


if __name__ == "__main__":
    args = parse_args()
    main(args)
