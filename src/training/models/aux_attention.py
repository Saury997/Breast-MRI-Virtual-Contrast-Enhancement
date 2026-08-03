"""Auxiliary segmentation attention blocks for synthesis decoders."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _normalize_aux_attention_config(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("model.aux_segmentation.attention must be a mapping")
    allowed = {"enabled", "mode", "strength", "warmup_epochs"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            "Unsupported model.aux_segmentation.attention field(s): "
            + ", ".join(unknown)
        )

    mode = str(value.get("mode", "decoder_final_gate")).lower()
    if mode != "decoder_final_gate":
        raise ValueError(
            "model.aux_segmentation.attention.mode must be 'decoder_final_gate', "
            f"got {mode!r}."
        )
    warmup_epochs = int(value.get("warmup_epochs", 0))
    if warmup_epochs < 0:
        raise ValueError("model.aux_segmentation.attention.warmup_epochs must be >= 0")

    return {
        "enabled": bool(value.get("enabled", False)),
        "mode": mode,
        "strength": float(value.get("strength", 1.0)),
        "warmup_epochs": warmup_epochs,
    }


def normalize_aux_segmentation_config(
    aux_segmentation: dict[str, Any] | None,
) -> dict[str, Any]:
    if aux_segmentation is None:
        aux_segmentation = {}
    if not isinstance(aux_segmentation, dict):
        raise ValueError("model.aux_segmentation must be a mapping when provided")
    allowed = {"enabled", "out_channels", "attention"}
    unknown = sorted(set(aux_segmentation) - allowed)
    if unknown:
        raise ValueError(
            "Unsupported model.aux_segmentation field(s): "
            + ", ".join(unknown)
        )

    config = dict(aux_segmentation)
    config["enabled"] = bool(config.get("enabled", False))
    config["out_channels"] = int(config.get("out_channels", 1))
    config["attention"] = _normalize_aux_attention_config(config.get("attention"))
    if config["attention"]["enabled"] and not config["enabled"]:
        raise ValueError(
            "model.aux_segmentation.attention.enabled=true requires "
            "model.aux_segmentation.enabled=true."
        )
    if config["attention"]["enabled"] and config["out_channels"] != 1:
        raise ValueError(
            "model.aux_segmentation.attention requires "
            "model.aux_segmentation.out_channels=1."
        )
    return config


class AuxDecoderAttention(nn.Module):
    """Gate final decoder features with auxiliary segmentation logits."""

    def __init__(
        self,
        channels: int,
        strength: float = 1.0,
        warmup_epochs: int = 0,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.warmup_epochs = int(warmup_epochs)
        if norm == "batch":
            norm_layer: nn.Module = nn.BatchNorm2d(channels)
        elif norm == "group":
            norm_layer = nn.GroupNorm(_num_groups(channels), channels)
        else:
            raise ValueError(f"Unsupported aux attention norm={norm!r}")
        self.adjust = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            norm_layer,
            nn.SiLU(inplace=True),
        )

    def factor(self, current_epoch: int | None = None) -> float:
        if current_epoch is None or self.warmup_epochs <= 0:
            return 1.0
        return max(0.0, min(1.0, float(current_epoch) / float(self.warmup_epochs)))

    def forward(
        self,
        decoded: torch.Tensor,
        seg_logits: torch.Tensor,
        current_epoch: int | None = None,
    ) -> torch.Tensor:
        if seg_logits.shape[1] != 1:
            raise RuntimeError(
                "Aux decoder attention expects one-channel segmentation logits, "
                f"got {tuple(seg_logits.shape)}."
            )
        if seg_logits.shape[-2:] != decoded.shape[-2:]:
            raise RuntimeError(
                "Aux decoder attention shape mismatch: "
                f"decoded={tuple(decoded.shape)}, seg_logits={tuple(seg_logits.shape)}."
            )
        factor = self.factor(current_epoch)
        if factor <= 0.0:
            return decoded
        mask = torch.sigmoid(seg_logits)
        attended = decoded * (1.0 + self.strength * mask)
        adjusted = self.adjust(attended)
        return decoded + factor * (adjusted - decoded)
