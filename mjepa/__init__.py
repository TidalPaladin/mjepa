import importlib.metadata

from .augmentation import AugmentationConfig
from .jepa import CrossAttentionPredictor, JEPAConfig
from .mae import MAEConfig
from .optimizer import OptimizerConfig
from .trainer import TrainerConfig


__version__ = importlib.metadata.version("mjepa")
__all__ = [
    "TrainerConfig",
    "JEPAConfig",
    "MAEConfig",
    "AugmentationConfig",
    "OptimizerConfig",
    "CrossAttentionPredictor",
]
