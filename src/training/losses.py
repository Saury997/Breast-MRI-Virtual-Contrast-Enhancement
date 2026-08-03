"""Losses for residual medical image synthesis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from kornia.losses import binary_focal_loss_with_logits
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.kl_loss import KLLoss


@dataclass
class LossWeights:
    l1: float = 1.0
    mse: float = 0.1
    roi_l1: float = 2.0
    ssim: float = 0.2
    gradient: float = 0.1
    roi_ssim: float = 0.0
    roi_gradient: float = 0.0
    perceptual: float = 0.0
    seg: float = 0.0
    warmup_epochs: int = 30
    perceptual_warmup_epochs: int | None = None
    perceptual_clip_sigma: float = 5.0
    perceptual_net: str = "alex"
    kl: float = 0.0
    wavelet: float = 0.0


@dataclass(frozen=True)
class WaveletLossConfig:
    wavelet: str = "haar"
    min_roi_pixels: int = 4
    band_weights: dict[str, float] | None = None
    eps: float = 1e-6


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.expand_as(values).to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    data_range: float = 10.0,
) -> torch.Tensor:
    pred = prediction.clamp(-5.0, 5.0)
    tgt = target.clamp(-5.0, 5.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = F.avg_pool2d(pred, kernel_size=7, stride=1, padding=3)
    mu_y = F.avg_pool2d(tgt, kernel_size=7, stride=1, padding=3)
    sigma_x = F.avg_pool2d(pred * pred, 7, 1, 3) - mu_x * mu_x
    sigma_y = F.avg_pool2d(tgt * tgt, 7, 1, 3) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * tgt, 7, 1, 3) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-6)
    return _masked_mean(1.0 - ssim_map, valid_mask)


def gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    channels = prediction.shape[1]
    sobel_x = prediction.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).view(1, 1, 3, 3)
    sobel_y = prediction.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).view(1, 1, 3, 3)
    sobel_x = sobel_x.repeat(channels, 1, 1, 1) / 8.0
    sobel_y = sobel_y.repeat(channels, 1, 1, 1) / 8.0

    pred_dx = F.conv2d(prediction, sobel_x, padding=1, groups=channels)
    tgt_dx = F.conv2d(target, sobel_x, padding=1, groups=channels)
    pred_dy = F.conv2d(prediction, sobel_y, padding=1, groups=channels)
    tgt_dy = F.conv2d(target, sobel_y, padding=1, groups=channels)
    support_mask = (
        F.avg_pool2d(valid_mask, kernel_size=3, stride=1, padding=1) >= 1.0
    ).to(prediction.dtype)
    return 0.5 * (
        _masked_mean((pred_dx - tgt_dx).abs(), support_mask)
        + _masked_mean((pred_dy - tgt_dy).abs(), support_mask)
    )


def _normalize_for_perceptual(
    image: torch.Tensor,
    clip_sigma: float,
) -> torch.Tensor:
    if clip_sigma <= 0:
        raise ValueError("perceptual_clip_sigma must be > 0")
    image = image.clamp(-clip_sigma, clip_sigma) / clip_sigma
    if image.shape[1] == 1:
        image = image.expand(-1, 3, -1, -1)
    return image.float()


def _advanced_warmup_epochs(weights: LossWeights) -> int:
    if weights.perceptual_warmup_epochs is not None:
        return weights.perceptual_warmup_epochs
    return weights.warmup_epochs


def _mask_to_boundary_distance_map(
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Distance map used by Kervadec et al. BoundaryLoss for binary masks."""
    mask_np = (mask.detach().cpu().numpy() > 0.5)
    valid_np = (valid_mask.detach().cpu().numpy() > 0.5)
    if mask_np.ndim != 4:
        raise ValueError(f"Expected 4-D mask tensor for SDF, got {mask_np.shape}")

    distance_np = np.zeros(mask_np.shape, dtype=np.float32)
    flat_mask = mask_np.reshape(-1, *mask_np.shape[-2:])
    flat_valid = valid_np.reshape(-1, *valid_np.shape[-2:])
    flat_distance = distance_np.reshape(-1, *distance_np.shape[-2:])

    for idx, (mask_2d, valid_2d) in enumerate(zip(flat_mask, flat_valid, strict=True)):
        if not valid_2d.any():
            continue
        rows, cols = np.where(valid_2d)
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        foreground = (mask_2d[y0:y1, x0:x1] & valid_2d[y0:y1, x0:x1]).astype(bool)
        if not foreground.any() or foreground.all():
            continue

        background = ~foreground
        distance_map = (
            ndimage.distance_transform_edt(background) * background
            - (ndimage.distance_transform_edt(foreground) - 1.0) * foreground
        )
        flat_distance[idx, y0:y1, x0:x1] = distance_map.astype(np.float32)

    return torch.from_numpy(distance_np).to(device=mask.device, dtype=mask.dtype)


def _segmentation_loss_from_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    dice_weight: float,
    focal_weight: float,
    focal_alpha: float,
    focal_gamma: float,
    boundary_weight: float,
    eps: float,
    error_prefix: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    target = (mask > 0.5).to(logits.dtype)
    if logits.shape != target.shape:
        raise ValueError(
            f"{error_prefix} shape mismatch: "
            f"logits={tuple(logits.shape)}, mask={tuple(target.shape)}"
        )

    valid = valid_mask.expand_as(logits).to(logits.dtype)
    probs = torch.sigmoid(logits) * valid
    target = target * valid
    reduce_dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=reduce_dims)
    denominator = probs.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice_score = ((2.0 * intersection + eps) / (denominator + eps)).mean()

    dice_loss = 1.0 - dice_score
    focal_map = binary_focal_loss_with_logits(
        logits,
        target,
        alpha=focal_alpha,
        gamma=focal_gamma,
        reduction="none",
    )
    focal_loss = _masked_mean(focal_map, valid)
    distance_map = _mask_to_boundary_distance_map(target, valid)
    boundary_loss = _masked_mean(probs * distance_map, valid)
    region_loss = dice_weight * dice_loss + focal_weight * focal_loss
    loss = region_loss + boundary_weight * boundary_loss
    return (
        loss,
        {
            "loss_seg": loss.detach(),
            "loss_seg_dice": dice_loss.detach(),
            "loss_seg_focal": focal_loss.detach(),
            "loss_seg_boundary": boundary_loss.detach(),
        },
        dice_score,
    )


class SegLoss(nn.Module):
    """Frozen segmenter loss for synthesized DEC-MRI lesion ROI agreement."""

    def __init__(
        self,
        segmenter: nn.Module,
        threshold: float = 0.5,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        boundary_weight: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.segmenter = segmenter
        self.threshold = threshold
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.boundary_weight = boundary_weight
        self.eps = eps
        self._freeze_segmenter()

    def _freeze_segmenter(self) -> None:
        self.segmenter.eval()
        for param in self.segmenter.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True) -> "SegLoss":
        super().train(mode)
        self._freeze_segmenter()
        return self

    def forward(
        self,
        prediction: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._freeze_segmenter()
        logits = self.segmenter(prediction)
        loss, parts, _ = _segmentation_loss_from_logits(
            logits,
            mask,
            valid_mask,
            dice_weight=self.dice_weight,
            focal_weight=self.focal_weight,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
            boundary_weight=self.boundary_weight,
            eps=self.eps,
            error_prefix="SegLoss",
        )
        return loss, parts


class AuxiliarySegmentationLoss(nn.Module):
    """Segmentation supervision for logits produced by the synthesis model."""

    def __init__(
        self,
        threshold: float = 0.5,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        boundary_weight: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.boundary_weight = float(boundary_weight)
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, parts, dice_score = _segmentation_loss_from_logits(
            logits,
            mask,
            valid_mask,
            dice_weight=self.dice_weight,
            focal_weight=self.focal_weight,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
            boundary_weight=self.boundary_weight,
            eps=self.eps,
            error_prefix="Auxiliary segmentation",
        )
        parts["aux_seg_dice"] = dice_score.detach()
        return loss, parts


class WaveletLoss(nn.Module):
    """ROI masked 1-level Haar wavelet loss over LL, LH, HL, and HH bands."""

    _BAND_NAMES = ("ll", "lh", "hl", "hh")

    def __init__(self, config: WaveletLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or WaveletLossConfig()
        if self.config.wavelet.lower() != "haar":
            raise ValueError(
                f"Unsupported wavelet={self.config.wavelet!r}. Only 'haar' is supported."
            )
        if self.config.min_roi_pixels < 1:
            raise ValueError("min_roi_pixels must be >= 1")
        if self.config.eps <= 0:
            raise ValueError("eps must be > 0")

        band_weights = {name: 1.0 for name in self._BAND_NAMES}
        if self.config.band_weights is not None:
            unknown = set(self.config.band_weights) - set(self._BAND_NAMES)
            if unknown:
                raise ValueError(
                    "Unsupported wavelet band weight(s): "
                    + ", ".join(sorted(unknown))
                )
            band_weights.update(
                {name: float(value) for name, value in self.config.band_weights.items()}
            )
        if any(value < 0 for value in band_weights.values()):
            raise ValueError("wavelet band weights must be >= 0")
        weight_sum = sum(band_weights.values())
        if weight_sum <= 0:
            raise ValueError("At least one wavelet band weight must be > 0")
        self.band_weights = band_weights

        low = torch.tensor([2.0**-0.5, 2.0**-0.5], dtype=torch.float32)
        high = torch.tensor([2.0**-0.5, -(2.0**-0.5)], dtype=torch.float32)
        filters = torch.stack(
            (
                torch.outer(low, low),
                torch.outer(low, high),
                torch.outer(high, low),
                torch.outer(high, high),
            )
        ).unsqueeze(1)
        self.register_buffer("haar_filters", filters, persistent=False)

    @staticmethod
    def _pad_image_to_even(image: torch.Tensor) -> torch.Tensor:
        pad_h = image.shape[-2] % 2
        pad_w = image.shape[-1] % 2
        if pad_h == 0 and pad_w == 0:
            return image
        return F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")

    @staticmethod
    def _pad_mask_to_even(mask: torch.Tensor) -> torch.Tensor:
        pad_h = mask.shape[-2] % 2
        pad_w = mask.shape[-1] % 2
        if pad_h == 0 and pad_w == 0:
            return mask
        return F.pad(mask, (0, pad_w, 0, pad_h), value=0.0)

    def _haar_dwt(self, image: torch.Tensor) -> torch.Tensor:
        image = self._pad_image_to_even(image)
        channels = image.shape[1]
        filters = self.haar_filters.to(device=image.device, dtype=image.dtype)
        filters = filters.repeat(channels, 1, 1, 1)
        coeffs = F.conv2d(image, filters, stride=2, groups=channels)
        return coeffs.view(image.shape[0], channels, len(self._BAND_NAMES), *coeffs.shape[-2:])

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        roi_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape:
            raise ValueError(
                "WaveletLoss prediction/target shape mismatch: "
                f"{tuple(prediction.shape)} != {tuple(target.shape)}"
            )

        prediction_f = prediction.float()
        target_f = target.float()
        roi = (
            (roi_mask > 0.5).to(prediction_f.dtype)
            * (valid_mask > 0.5).to(prediction_f.dtype)
        ).expand_as(prediction_f)
        roi = self._pad_mask_to_even(roi)
        subband_roi = F.avg_pool2d(roi, kernel_size=2, stride=2)
        keep = (
            subband_roi.sum(dim=(1, 2, 3)) * 4.0
            >= float(self.config.min_roi_pixels)
        )
        if not torch.any(keep):
            zero = prediction_f.sum() * 0.0
            return zero, {
                "loss_wavelet": zero.detach(),
                "loss_wavelet_ll": zero.detach(),
                "loss_wavelet_lh": zero.detach(),
                "loss_wavelet_hl": zero.detach(),
                "loss_wavelet_hh": zero.detach(),
            }

        pred_coeffs = self._haar_dwt(prediction_f)[keep]
        target_coeffs = self._haar_dwt(target_f)[keep]
        weights = subband_roi[keep]

        band_losses: dict[str, torch.Tensor] = {}
        total = prediction_f.new_tensor(0.0)
        total_weight = 0.0
        for band_idx, band_name in enumerate(self._BAND_NAMES):
            band_loss = _masked_mean(
                (pred_coeffs[:, :, band_idx] - target_coeffs[:, :, band_idx]).abs(),
                weights,
            )
            band_losses[band_name] = band_loss
            band_weight = self.band_weights[band_name]
            if band_weight > 0:
                total = total + float(band_weight) * band_loss
                total_weight += float(band_weight)

        total = total / max(total_weight, self.config.eps)
        return total, {
            "loss_wavelet": total.detach(),
            "loss_wavelet_ll": band_losses["ll"].detach(),
            "loss_wavelet_lh": band_losses["lh"].detach(),
            "loss_wavelet_hl": band_losses["hl"].detach(),
            "loss_wavelet_hh": band_losses["hh"].detach(),
        }


class SynthesisLoss(nn.Module):
    def __init__(
        self,
        weights: LossWeights | None = None,
        seg_loss: SegLoss | None = None,
        aux_seg_loss: AuxiliarySegmentationLoss | None = None,
        kl_loss: KLLoss | None = None,
        wavelet_loss: WaveletLoss | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.seg_loss = seg_loss
        self.aux_seg_loss = aux_seg_loss
        self._perceptual_loss: nn.Module | None = None
        self.kl_loss = kl_loss
        self.wavelet_loss = wavelet_loss

    def advanced_warmup_epochs(self) -> int:
        return _advanced_warmup_epochs(self.weights)

    def advanced_losses_active(self, current_epoch: int) -> bool:
        return current_epoch >= self.advanced_warmup_epochs()

    def _get_perceptual_loss(self, device: torch.device) -> nn.Module:
        if self._perceptual_loss is None:
            from torchmetrics.image.lpip import (
                LearnedPerceptualImagePatchSimilarity,
            )

            self._perceptual_loss = LearnedPerceptualImagePatchSimilarity(
                net_type=self.weights.perceptual_net,
            )
            self._perceptual_loss.eval()
            for param in self._perceptual_loss.parameters():
                param.requires_grad_(False)

        return self._perceptual_loss.to(device)

    @staticmethod
    def _prefix_seg_parts(
        parts: dict[str, torch.Tensor],
        branch: str,
    ) -> dict[str, torch.Tensor]:
        prefixed: dict[str, torch.Tensor] = {}
        for name, value in parts.items():
            if name == "loss_seg":
                prefixed[f"loss_seg_{branch}"] = value
            elif name.startswith("loss_seg_"):
                suffix = name.removeprefix("loss_seg_")
                prefixed[f"loss_seg_{branch}_{suffix}"] = value
            else:
                prefixed[name] = value
        return prefixed

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: torch.Tensor,
        segmentation_logits: torch.Tensor | None = None,
        current_epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        w = self.weights
        advanced_active = self.advanced_losses_active(current_epoch)

        l1 = _masked_mean((prediction - target).abs(), valid_mask)
        mse = _masked_mean((prediction - target).square(), valid_mask)
        roi_weight = (mask > 0.5).to(prediction.dtype) * valid_mask
        roi_l1 = _masked_mean((prediction - target).abs(), roi_weight)
        if roi_weight.sum() <= 0:
            roi_l1 = prediction.new_tensor(0.0)
        ssim = ssim_loss(prediction, target, valid_mask)
        roi_ssim = ssim_loss(prediction, target, roi_weight)
        grad = gradient_loss(prediction, target, valid_mask)
        roi_grad = gradient_loss(prediction, target, roi_weight)

        perceptual = prediction.new_tensor(0.0)
        perceptual_weight = prediction.new_tensor(0.0)
        if w.perceptual > 0 and advanced_active:
            perceptual_fn = self._get_perceptual_loss(prediction.device)
            pred_norm = _normalize_for_perceptual(
                prediction,
                w.perceptual_clip_sigma,
            )
            target_norm = _normalize_for_perceptual(
                target,
                w.perceptual_clip_sigma,
            )
            perceptual = perceptual_fn(pred_norm, target_norm)
            perceptual_weight = prediction.new_tensor(float(w.perceptual))

        total = (
            w.l1 * l1
            + w.mse * mse
            + w.roi_l1 * roi_l1
            + w.ssim * ssim
            + w.gradient * grad
            + w.roi_ssim * roi_ssim
            + w.roi_gradient * roi_grad
            + perceptual_weight * perceptual
        )
        parts = {
            "loss_l1": l1.detach(),
            "loss_mse": mse.detach(),
            #"loss_roi_l1": roi_l1.detach(),
            "loss_ssim": ssim.detach(),
            #"loss_gradient": grad.detach(),
            "loss_roi_ssim": roi_ssim.detach(),
            #"loss_roi_gradient": roi_grad.detach(),
            "loss_perceptual": perceptual.detach(),
        }
        if self.kl_loss is not None and w.kl > 0 and advanced_active:
            kl_value, kl_parts = self.kl_loss(
                prediction,
                target,
                mask,
                valid_mask,
            )
            total = total + w.kl * kl_value
            parts.update(kl_parts)
        if self.wavelet_loss is not None and w.wavelet > 0:
            wavelet_value, wavelet_parts = self.wavelet_loss(
                prediction,
                target,
                mask,
                valid_mask,
            )
            total = total + w.wavelet * wavelet_value
            parts.update(wavelet_parts)
        seg_total = prediction.new_tensor(0.0)
        has_seg_loss = False
        combined_seg = self.seg_loss is not None and self.aux_seg_loss is not None
        if self.seg_loss is not None:
            seg_loss_value, seg_parts = self.seg_loss(prediction, mask, valid_mask)
            seg_total = seg_total + seg_loss_value
            has_seg_loss = True
            parts.update(
                self._prefix_seg_parts(seg_parts, "frozen")
                if combined_seg
                else seg_parts
            )
        if self.aux_seg_loss is not None:
            if segmentation_logits is None:
                raise ValueError(
                    "Auxiliary segmentation loss is enabled, but the model did "
                    "not return segmentation logits."
                )
            seg_loss_value, seg_parts = self.aux_seg_loss(
                segmentation_logits,
                mask,
                valid_mask,
            )
            seg_total = seg_total + seg_loss_value
            has_seg_loss = True
            parts.update(
                self._prefix_seg_parts(seg_parts, "aux")
                if combined_seg
                else seg_parts
            )
        if has_seg_loss:
            total = total + w.seg * seg_total
            if combined_seg:
                parts["loss_seg"] = seg_total.detach()
        return total, parts
