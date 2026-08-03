"""A compact 2-D residual U-Net for pre-to-post contrast synthesis."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.models.aux_attention import (
    AuxDecoderAttention,
    normalize_aux_segmentation_config,
)


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ResidualBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResUNet(nn.Module):
    """U-Net that predicts an enhancement residual delta."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        aux_segmentation: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        aux_config = normalize_aux_segmentation_config(aux_segmentation)
        self.aux_segmentation_enabled = bool(aux_config.get("enabled", False))
        aux_out_channels = int(aux_config.get("out_channels", 1))
        attention_config = aux_config["attention"]
        self.aux_attention_enabled = bool(attention_config["enabled"])
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.enc1 = ResidualBlock(in_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)
        self.bottleneck = DownBlock(c4, c5)
        self.dec4 = UpBlock(c5, c4, c4)
        self.dec3 = UpBlock(c4, c3, c3)
        self.dec2 = UpBlock(c3, c2, c2)
        self.dec1 = UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)
        if self.aux_segmentation_enabled:
            self.seg_dec4 = UpBlock(c5, c4, c4)
            self.seg_dec3 = UpBlock(c4, c3, c3)
            self.seg_dec2 = UpBlock(c3, c2, c2)
            self.seg_dec1 = UpBlock(c2, c1, c1)
            self.seg_head = nn.Conv2d(c1, aux_out_channels, kernel_size=1)
            aux_prefixes = [
                "seg_dec4.",
                "seg_dec3.",
                "seg_dec2.",
                "seg_dec1.",
                "seg_head.",
            ]
            if self.aux_attention_enabled:
                self.aux_attention = AuxDecoderAttention(
                    c1,
                    strength=float(attention_config["strength"]),
                    warmup_epochs=int(attention_config["warmup_epochs"]),
                    norm="group",
                )
                aux_prefixes.append("aux_attention.")
            self.aux_state_key_prefixes = tuple(aux_prefixes)

    def _encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        bottleneck = self.bottleneck(s4)
        return s1, s2, s3, s4, bottleneck

    def _decode_generator(
        self,
        skips_and_bottleneck: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1, s2, s3, s4, x = skips_and_bottleneck
        g4 = self.dec4(x, s4)
        g3 = self.dec3(g4, s3)
        g2 = self.dec2(g3, s2)
        g1 = self.dec1(g2, s1)
        return g1, [g4, g3, g2, g1]

    def aux_attention_factor(self, current_epoch: int | None = None) -> float | None:
        if not self.aux_attention_enabled:
            return None
        return self.aux_attention.factor(current_epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.aux_attention_enabled:
            residual, _ = self.forward_with_aux(x, current_epoch=None)
            return residual
        decoded, _ = self._decode_generator(self._encode(x))
        return self.head(decoded)

    def forward_with_aux(
        self,
        x: torch.Tensor,
        current_epoch: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = self._encode(x)
        decoded, generator_features = self._decode_generator(encoded)
        if not self.aux_segmentation_enabled:
            return self.head(decoded), None

        s1, s2, s3, s4, bottleneck = encoded
        g4, g3, g2, g1 = generator_features
        x_seg = self.seg_dec4(bottleneck, s4) + g4
        x_seg = self.seg_dec3(x_seg, s3) + g3
        x_seg = self.seg_dec2(x_seg, s2) + g2
        x_seg = self.seg_dec1(x_seg, s1) + g1
        seg_logits = self.seg_head(x_seg)
        if self.aux_attention_enabled:
            decoded = self.aux_attention(decoded, seg_logits, current_epoch)
        return self.head(decoded), seg_logits
