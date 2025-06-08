import os

import torch


try:
    if torch.cuda.is_available():
        arch_list = []
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            arch_list.append(f"{cap[0]}.{cap[1]}")
        if arch_list:
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted(set(arch_list), reverse=True))
except BaseException:
    pass


from .mixup import bce_mixup, cross_entropy_mixup, is_mixed
from .pointwise import AugmentationConfig, apply_invert, apply_mixup, apply_noise, apply_posterize


__all__ = [
    "AugmentationConfig",
    "apply_noise",
    "apply_mixup",
    "apply_invert",
    "apply_posterize",
    "bce_mixup",
    "is_mixed",
    "cross_entropy_mixup",
]
