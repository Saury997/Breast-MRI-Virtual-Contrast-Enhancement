"""Factory functions for synthesis models."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from training.models.resunet import ResUNet
from training.models.smp_model import SMPModel, is_smp_model_type


def create_model(
    model_type: str | None = None,
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int | None = None,
    aux_segmentation: dict[str, Any] | None = None,
    **kwargs: Any,
) -> nn.Module:
    config_type = kwargs.pop("type", None)
    if model_type is None:
        model_type = str(config_type or "resunet")
    elif config_type is not None and str(config_type).lower() != model_type.lower():
        raise ValueError(
            f"Conflicting model type values: {model_type!r} and {config_type!r}."
        )

    normalized = model_type.lower()
    if normalized == "resunet":
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported ResUNet config field(s): {unknown}")
        return ResUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=32 if base_channels is None else base_channels,
            aux_segmentation=aux_segmentation,
        )
    if is_smp_model_type(normalized):
        if base_channels is not None:
            raise ValueError("base_channels is only valid for model.type='resunet'.")
        return SMPModel(
            architecture=normalized,
            in_channels=in_channels,
            out_channels=out_channels,
            aux_segmentation=aux_segmentation,
            **kwargs,
        )
    raise ValueError(
        f"Unsupported model_type={model_type!r}. "
        "Available: resunet, unet, unetplusplus, fpn, pan, manet, linknet, "
        "pspnet, deeplabv3, deeplabv3plus, upernet. "
        "GAN/diffusion/flow are intentionally not wired yet."
    )
