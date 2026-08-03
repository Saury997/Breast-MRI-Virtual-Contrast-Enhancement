"""Lightweight lesion ROI segmentation model and Lightning module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    """Compact 2-D U-Net that returns lesion-mask logits."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("TinyUNet depth must be >= 2")
        channels = [base_channels * (2**idx) for idx in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for out_ch in channels:
            self.encoders.append(ConvBlock(prev_channels, out_ch))
            prev_channels = out_ch

        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)

        self.decoders = nn.ModuleList()
        decoder_in = channels[-1] * 2
        for skip_channels in reversed(channels):
            self.decoders.append(ConvBlock(decoder_in + skip_channels, skip_channels))
            decoder_in = skip_channels

        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        out = x
        for encoder in self.encoders:
            out = encoder(out)
            skips.append(out)
            out = F.max_pool2d(out, kernel_size=2, stride=2)

        out = self.bottleneck(out)

        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            out = F.interpolate(
                out,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            out = decoder(torch.cat([out, skip], dim=1))
        return self.head(out)


def create_segmenter_model(
    model_type: str | None = None,
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 16,
    depth: int = 4,
    **kwargs: Any,
) -> nn.Module:
    config_type = kwargs.pop("type", None)
    if model_type is None:
        model_type = str(config_type or "tiny_unet")
    elif config_type is not None and str(config_type).lower() != model_type.lower():
        raise ValueError(
            f"Conflicting segmenter model type values: {model_type!r} and "
            f"{config_type!r}."
        )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise ValueError(f"Unsupported segmenter model config field(s): {unknown}")

    normalized = model_type.lower()
    if normalized in {"tiny_unet", "unet_lite", "lite_unet"}:
        return TinyUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
        )
    raise ValueError(
        f"Unsupported segmenter model_type={model_type!r}. "
        "Available: tiny_unet."
    )


def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.expand_as(values).to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def soft_dice_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = (target > 0.5).to(probs.dtype)
    if valid_mask is not None:
        weights = valid_mask.expand_as(probs).to(probs.dtype)
        probs = probs * weights
        target = target * weights

    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


def hard_dice_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).to(probs.dtype)
    target = (target > 0.5).to(probs.dtype)
    if valid_mask is not None:
        weights = valid_mask.expand_as(pred).to(pred.dtype)
        pred = pred * weights
        target = target * weights

    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    denominator = pred.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


@dataclass(frozen=True)
class SegmenterLossWeights:
    bce: float = 0.5
    dice: float = 1.0
    pos_weight: float | None = 4.0


class SegmenterModule(L.LightningModule):
    """Train a lightweight lesion ROI segmenter on post-contrast images."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_name: str = "adamw",
        max_epochs: int = 100,
        image_key: str = "target",
        threshold: float = 0.5,
        loss_weights: SegmenterLossWeights | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.max_epochs = max_epochs
        self.image_key = image_key
        self.threshold = threshold
        self.loss_weights = loss_weights or SegmenterLossWeights()
        if self.loss_weights.pos_weight is None:
            self.register_buffer("pos_weight", None, persistent=False)
        else:
            self.register_buffer(
                "pos_weight",
                torch.tensor([float(self.loss_weights.pos_weight)]),
            )
        self.save_hyperparameters(ignore=["model", "loss_weights"])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def _loss(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = (mask > 0.5).to(logits.dtype)
        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        bce = _masked_mean(bce_map, valid_mask)
        soft_dice = soft_dice_score(logits, target, valid_mask)
        dice_loss = 1.0 - soft_dice
        total = self.loss_weights.bce * bce + self.loss_weights.dice * dice_loss
        return total, {
            "loss_bce": bce.detach(),
            "loss_dice": dice_loss.detach(),
            "soft_dice": soft_dice.detach(),
            "hard_dice": hard_dice_score(
                logits.detach(),
                target,
                valid_mask,
                threshold=self.threshold,
            ),
        }

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        if self.image_key not in batch:
            raise KeyError(f"Batch does not contain image_key={self.image_key!r}")
        image = batch[self.image_key]
        mask = batch["mask"]
        valid_mask = batch["valid_mask"]
        if image.shape != mask.shape:
            raise ValueError(
                f"Segmenter image/mask shape mismatch: image={tuple(image.shape)}, "
                f"mask={tuple(mask.shape)}"
            )
        logits = self(image)
        loss, parts = self._loss(logits, mask, valid_mask)
        batch_size = int(image.shape[0])
        on_step = stage == "train"
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=stage == "train",
            batch_size=batch_size,
            on_step=on_step,
            on_epoch=True,
        )
        for name, value in parts.items():
            self.log(
                f"{stage}/{name}",
                value,
                prog_bar=name.endswith("dice"),
                batch_size=batch_size,
                on_step=on_step,
                on_epoch=True,
            )
        if stage == "val":
            self.log("val_loss", loss, batch_size=batch_size, prog_bar=False)
            self.log(
                "val_dice",
                parts["hard_dice"],
                batch_size=batch_size,
                prog_bar=True,
            )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> dict[str, Any]:
        if self.optimizer_name.lower() != "adamw":
            raise ValueError(
                f"Unsupported optimizer={self.optimizer_name!r}. "
                "Only AdamW is currently implemented."
            )
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.max_epochs),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
