from os import PathLike
from pathlib import Path
from typing import Iterator, List, Sequence

import pandas as pd
import torch
from dicom_preprocessing import load_tiff_f32
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Image as TVImage
from torchvision.tv_tensors import Video as TVVideo


def load_and_wrap_pixels(path: PathLike, frames: Sequence[int] | None = None) -> TVVideo | TVImage:
    r"""Load a TIFF file and wrap it in a TVVideo or TVImage.

    Args:
        path: Path to the TIFF file.
        frames: Sequence of frames to load. If None, all frames are loaded.

    Returns:
        A TVVideo or TVImage tensor.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    # Load pixels in N H W C format
    image = load_tiff_f32(str(path), frames)
    assert image.ndim == 4, f"Expected 4D tensor, got {image.shape}"

    # Permute and wrap
    if image.shape[0] > 1:
        image = rearrange(torch.from_numpy(image), "n h w c -> n c h w")
        return TVVideo(image)
    else:
        image = rearrange(torch.from_numpy(image), "() h w c -> c h w")
        return TVImage(image)


class PreprocessedTIFFDataset(Dataset):
    r"""Dataset for TIFF files created by `dicom-preprocessing`.

    Handling of 3D volumes is as follows:
        - If `keep_volume` is True the entire volume is used. The output will be a :class:`TVVideo` tensor.
        - If `keep_volume` is False and `training` is True, a random slice is selected. The output will be a :class:`TVImage` tensor.
        - If `keep_volume` is False and `training` is False, the middle slice is used. The output will be a :class:`TVImage` tensor.

    Args:
        root: Path to the root directory of the dataset.
        training: Whether the dataset is used for training.
        transform: Transform to apply to the image.
        keep_volume: Whether to keep the volume of the image.
    """

    def __init__(
        self,
        root: Path,
        training: bool = False,
        transform: Transform | None = None,
        keep_volume: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.training = training
        self.keep_volume = keep_volume
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

        # Open manifest
        manifest_path = self.root / "manifest.parquet"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file {manifest_path} does not exist")
        self.manifest = pd.read_parquet(manifest_path)
        self.manifest = self.manifest.sort_values(by="inode")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root={self.root}, training={self.training}, keep_volume={self.keep_volume})"

    def __len__(self) -> int:
        return len(self.manifest)

    @property
    def sop_uids(self) -> Iterator[str]:
        yield from self.manifest["sop_instance_uid"].unique()

    @property
    def study_uids(self) -> Iterator[str]:
        yield from self.manifest["study_instance_uid"].unique()

    def _select_frames(self, num_frames: int) -> List[int] | None:
        match (num_frames, self.keep_volume, self.training):
            # 2D
            case (1, _, _):
                frames = None
            # 3D with keep volume
            case (_, True, _):
                frames = None
            # 3D without keep volume at train time
            case (_, False, True):
                frames = torch.randint(0, num_frames, (1,)).tolist()
            # 3D without keep volume at test time
            case (_, False, False):
                frames = [num_frames // 2]
            case _:
                raise ValueError(
                    f"Invalid combination of num_frames={num_frames}, keep_volume={self.keep_volume}, training={self.training}"
                )
        return frames

    def __getitem__(self, index: int) -> Tensor:
        data = self.manifest.iloc[index].to_dict()
        path = self.root / data["path"]
        if not path.is_file():
            raise FileNotFoundError(f"File {path} does not exist")

        # Decide what frames to load
        num_frames = data.get("num_frames", 1)
        frames = self._select_frames(num_frames)

        # Load image
        image = load_and_wrap_pixels(path, frames)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        assert isinstance(image, (TVImage, TVVideo))
        return image
