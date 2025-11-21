import importlib.metadata

from .augmentation import AugmentationConfig
from .jepa import CrossAttentionPredictor, JEPAConfig
from .optimizer import OptimizerConfig
from .trainer import ResolutionConfig, TrainerConfig


__version__ = importlib.metadata.version("mjepa")
__all__ = [
    "TrainerConfig",
    "JEPAConfig",
    "AugmentationConfig",
    "OptimizerConfig",
    "CrossAttentionPredictor",
    "ResolutionConfig",
]
