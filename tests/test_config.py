from pathlib import Path

import yaml
from vit import AttentivePoolHeadConfig, HeadConfig, ViTConfig

from mjepa import JEPAConfig, OptimizerConfig, TrainerConfig


DEFAULT_CONFIG_PATH = Path(__file__).parents[1] / "config.yaml"
EXPECTED_HEAD_NAMES = {"cls", "attentive"}
EXPECTED_HEAD_OUTPUT_FEATURES = 10


def test_default_config_loads_and_instantiates_backbone():
    with DEFAULT_CONFIG_PATH.open() as config_file:
        config = yaml.safe_load(config_file)

    assert isinstance(config["trainer"], TrainerConfig)
    assert isinstance(config["jepa"], JEPAConfig)
    assert isinstance(config["optimizer"], OptimizerConfig)

    backbone_config = config["backbone"]
    assert isinstance(backbone_config, ViTConfig)
    assert set(backbone_config.heads) == EXPECTED_HEAD_NAMES

    cls_head = backbone_config.heads["cls"]
    attentive_head = backbone_config.heads["attentive"]
    assert isinstance(cls_head, HeadConfig)
    assert isinstance(attentive_head, AttentivePoolHeadConfig)
    assert cls_head.out_features == EXPECTED_HEAD_OUTPUT_FEATURES
    assert attentive_head.out_features == EXPECTED_HEAD_OUTPUT_FEATURES

    backbone = backbone_config.instantiate()
    assert set(backbone.heads) == EXPECTED_HEAD_NAMES
