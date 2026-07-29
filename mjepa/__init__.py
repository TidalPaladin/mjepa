import importlib.metadata

from .jepa import CLSPredictionMode, CrossAttentionPredictor, JEPAConfig, PredictorAttentionMode
from .metrics import CLSPatchAlignmentMetric, SimilarityDistanceCouplingMetric
from .optimizer import OptimizerConfig
from .trainer import CheckpointMetadata, ResolutionConfig, TrainerConfig, load_checkpoint_metadata


__version__ = importlib.metadata.version("mjepa")
__all__ = [
    "TrainerConfig",
    "JEPAConfig",
    "PredictorAttentionMode",
    "CLSPredictionMode",
    "OptimizerConfig",
    "CrossAttentionPredictor",
    "ResolutionConfig",
    "CLSPatchAlignmentMetric",
    "SimilarityDistanceCouplingMetric",
    "CheckpointMetadata",
    "load_checkpoint_metadata",
]
