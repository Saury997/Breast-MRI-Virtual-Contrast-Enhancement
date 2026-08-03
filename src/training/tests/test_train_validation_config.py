from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training.config import load_yaml_config
from training.train import resolve_validation_enabled


def test_loss_ablation_config_explicitly_enables_validation() -> None:
    config_path = SRC_DIR / "training" / "configs" / "loss_ablation.yaml"
    config = load_yaml_config(config_path)

    assert config["trainer"]["enable_validation"] is True
    assert resolve_validation_enabled(config, val_record_count=1) is True


def test_loss_ablation_enables_aux_segmentation_visualization() -> None:
    config_path = SRC_DIR / "training" / "configs" / "loss_ablation.yaml"
    config = load_yaml_config(config_path)

    assert config["logging"]["visualize_val_samples"] is True
    assert config["logging"]["num_visualization_cases"] > 0


def test_loss_ablation_uses_combined_segmentation_modes() -> None:
    config_path = SRC_DIR / "training" / "configs" / "loss_ablation.yaml"
    config = load_yaml_config(config_path)
    seg_config = config["loss"]["seg"]

    assert seg_config["enabled"] is True
    assert seg_config["modes"] == ["frozen_segmenter", "auxiliary_branch"]
    assert "checkpoint" in seg_config
    assert seg_config["model"]["type"] == "tiny_unet"


def test_explicit_validation_requires_validation_records() -> None:
    config = {"trainer": {"enable_validation": True}}

    with pytest.raises(ValueError, match="enable_validation=true"):
        resolve_validation_enabled(config, val_record_count=0)


def test_validation_can_be_explicitly_disabled_even_with_val_records() -> None:
    config = {"trainer": {"enable_validation": False}}

    assert resolve_validation_enabled(config, val_record_count=3) is False


def test_validation_defaults_to_available_val_records() -> None:
    assert resolve_validation_enabled({}, val_record_count=2) is True
    assert resolve_validation_enabled({}, val_record_count=0) is False
