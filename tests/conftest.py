import os
from pathlib import Path

import torch
import pytest
import torch.nn as nn

from vit import ViT, ViTConfig


def cuda_available():
    r"""Checks if CUDA is available and device is ready"""
    if not torch.cuda.is_available():
        return False

    capability = torch.cuda.get_device_capability()
    arch_list = torch.cuda.get_arch_list()
    if isinstance(capability, tuple):
        capability = f"sm_{''.join(str(x) for x in capability)}"

    if capability not in arch_list:
        return False

    return True


def handle_cuda_mark(item):  # pragma: no cover
    has_cuda_mark = any(item.iter_markers(name="cuda"))
    if has_cuda_mark and not cuda_available():
        import pytest

        pytest.skip("Test requires CUDA and device is not ready")


def pytest_runtest_setup(item):
    handle_cuda_mark(item)


@pytest.fixture
def vit_config():
    """Create a mock ViT configuration."""
    return ViTConfig(
        in_channels=3,
        hidden_size=128,
        patch_size=[16, 16],
        depth=2,
        num_attention_heads=4,
        ffn_hidden_size=256,
        activation="gelu",
        normalization="LayerNorm",
        backend="pytorch",
        checkpoint=False,
    )


@pytest.fixture
def vit(vit_config):
    """Create a mock ViT backbone."""
    return vit_config.instantiate()


@pytest.fixture
def dummy_image_batch():
    """Create a dummy batch of images."""
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def dummy_tensor_1d():
    """Create a dummy 1D tensor."""
    return torch.randn(128)


@pytest.fixture
def dummy_tensor_2d():
    """Create a dummy 2D tensor."""
    return torch.randn(16, 128)


@pytest.fixture
def dummy_tensor_3d():
    """Create a dummy 3D tensor."""
    return torch.randn(2, 32, 128)