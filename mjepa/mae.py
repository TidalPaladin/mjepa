from dataclasses import dataclass

import yaml


@dataclass
class MAEConfig:
    """
    Configuration for MAE related hyperparameters.

    Args:
        context_ratio: Ratio of the input to sample as context.
        target_ratio: Ratio of the input to sample as a prediction target.
        scale: Integer scale at which to sample contiguous blocks of context tokens.
            Increasing this ensures more adjacent tokens appear together in the context.
        predictor_depth: Depth of the predictor network.
        context_pos_emb: Whether to introduce positional encoding to the context
            as part of the predictor network.
        shared_pos_emb: Whether to use the backbone's positional encoding in the predictor.
    """

    context_ratio: float = 0.5
    target_ratio: float = 0.25
    scale: int = 4
    predictor_depth: int = 4
    context_pos_emb: bool = False
    shared_pos_emb: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.context_ratio <= 1:
            raise ValueError("context_ratio must be in the range (0, 1]")
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in the range (0, 1]")


def config_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    return MAEConfig(**values)


def register_constructors():
    tags = [
        "tag:yaml.org,2002:python/object:mjepa.MAEConfig",
    ]
    loaders = [yaml.SafeLoader, yaml.FullLoader, yaml.UnsafeLoader]
    for tag in tags:
        for loader in loaders:
            loader.add_constructor(tag, config_constructor)


register_constructors()
