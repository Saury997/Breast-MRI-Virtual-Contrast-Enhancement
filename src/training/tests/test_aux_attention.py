from pathlib import Path
import sys

import pytest
import torch

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training.models.resunet import ResUNet
from training.models.smp_model import SMPModel
from training.modules import ResidualTranslationModule


def _aux_attention_config(warmup_epochs: int = 5) -> dict:
    return {
        "enabled": True,
        "out_channels": 1,
        "attention": {
            "enabled": True,
            "mode": "decoder_final_gate",
            "strength": 1.0,
            "warmup_epochs": warmup_epochs,
        },
    }


def test_resunet_aux_attention_shapes_and_warmup_factor() -> None:
    model = ResUNet(
        in_channels=1,
        out_channels=1,
        base_channels=4,
        aux_segmentation=_aux_attention_config(warmup_epochs=5),
    )
    x = torch.randn(2, 1, 64, 64)

    assert model.aux_attention_factor(0) == 0.0
    assert model.aux_attention_factor(5) == 1.0
    assert model.aux_attention_factor(None) == 1.0
    assert any(key.startswith("aux_attention.adjust") for key in model.state_dict())

    residual = model(x)
    aux_residual, seg_logits = model.forward_with_aux(x, current_epoch=0)

    assert residual.shape == x.shape
    assert aux_residual.shape == x.shape
    assert seg_logits is not None
    assert seg_logits.shape == x.shape


def test_resunet_aux_attention_config_validation() -> None:
    with pytest.raises(ValueError, match="requires model.aux_segmentation.enabled=true"):
        ResUNet(
            aux_segmentation={
                "enabled": False,
                "attention": {"enabled": True},
            }
        )

    with pytest.raises(ValueError, match="out_channels=1"):
        ResUNet(
            aux_segmentation={
                "enabled": True,
                "out_channels": 2,
                "attention": {"enabled": True},
            }
        )


def test_smp_unet_aux_attention_shapes_and_warmup_factor() -> None:
    pytest.importorskip("segmentation_models_pytorch")
    model = SMPModel(
        architecture="unet",
        in_channels=1,
        out_channels=1,
        encoder_name="resnet18",
        encoder_weights=None,
        encoder_depth=3,
        decoder_channels=[32, 16, 8],
        aux_segmentation=_aux_attention_config(warmup_epochs=4),
    )
    x = torch.randn(1, 1, 64, 64)

    assert model.aux_attention_factor(0) == 0.0
    assert model.aux_attention_factor(4) == 1.0
    assert model.aux_attention_factor(None) == 1.0
    assert any(key.startswith("aux_attention.adjust") for key in model.state_dict())

    residual = model(x)
    aux_residual, seg_logits = model.forward_with_aux(x, current_epoch=2)

    assert residual.shape == x.shape
    assert aux_residual.shape == x.shape
    assert seg_logits is not None
    assert seg_logits.shape == x.shape


def test_smp_unetpp_aux_attention_shapes_and_warmup_factor() -> None:
    pytest.importorskip("segmentation_models_pytorch")
    model = SMPModel(
        architecture="unetpp",
        in_channels=1,
        out_channels=1,
        encoder_name="resnet18",
        encoder_weights=None,
        encoder_depth=3,
        decoder_channels=[32, 16, 8],
        aux_segmentation=_aux_attention_config(warmup_epochs=4),
    )
    x = torch.randn(1, 1, 64, 64)

    assert model.aux_attention_factor(0) == 0.0
    assert model.aux_attention_factor(4) == 1.0
    assert any(key.startswith("aux_attention.adjust") for key in model.state_dict())

    residual = model(x)
    aux_residual, seg_logits = model.forward_with_aux(x, current_epoch=2)

    assert residual.shape == x.shape
    assert aux_residual.shape == x.shape
    assert seg_logits is not None
    assert seg_logits.shape == x.shape


def test_module_forwards_epoch_to_aux_attention() -> None:
    model = ResUNet(
        in_channels=1,
        out_channels=1,
        base_channels=4,
        aux_segmentation=_aux_attention_config(warmup_epochs=10),
    )
    module = ResidualTranslationModule(model=model)
    x = torch.randn(1, 1, 64, 64)

    prediction, seg_logits, aux_parts = module._forward_prediction_and_aux(
        x,
        current_epoch=5,
    )

    assert prediction.shape == x.shape
    assert seg_logits is not None
    assert seg_logits.shape == x.shape
    assert float(aux_parts["aux_attention_factor"]) == 0.5
