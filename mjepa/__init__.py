import importlib.metadata

from .jepa import CrossAttentionPredictor, JEPAConfig
from .metrics import CLSPatchAlignmentMetric, SimilarityDistanceCouplingMetric
from .optimizer import OptimizerConfig
from .trainer import ResolutionConfig, TrainerConfig


__version__ = importlib.metadata.version("mjepa")
__all__ = [
    "TrainerConfig",
    "JEPAConfig",
    "OptimizerConfig",
    "CrossAttentionPredictor",
    "ResolutionConfig",
    "CLSPatchAlignmentMetric",
    "SimilarityDistanceCouplingMetric",
]
