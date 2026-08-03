"""Differentiable ROI intensity-distribution loss based on KL divergence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class KLLossConfig:
    histogram_bins: int = 16
    histogram_sigma: float = 0.15
    clip_sigma: float = 5.0
    min_roi_pixels: int = 8
    symmetric: bool = True
    eps: float = 1e-6


class KLLoss(nn.Module):
    """Match prediction and target soft histograms inside the lesion ROI.

    Images are clipped in z-score space, mapped to [-1, 1], and converted
    into differentiable normalized histograms after masking by
    ``roi_mask * valid_mask``. This is intentionally an ROI grayscale
    histogram divergence, not a full-image distribution loss. By default the
    loss is the symmetric KL divergence, which gives gradients in both
    directions and is less sensitive to the arbitrary choice of prediction or
    target as the reference distribution.
    """

    def __init__(self, config: KLLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or KLLossConfig()
        if self.config.histogram_bins < 2:
            raise ValueError("histogram_bins must be >= 2")
        if self.config.histogram_sigma <= 0:
            raise ValueError("histogram_sigma must be > 0")
        if self.config.clip_sigma <= 0:
            raise ValueError("clip_sigma must be > 0")
        if self.config.min_roi_pixels < 1:
            raise ValueError("min_roi_pixels must be >= 1")
        if self.config.eps <= 0:
            raise ValueError("eps must be > 0")

        centers = torch.linspace(-1.0, 1.0, self.config.histogram_bins)
        self.register_buffer("histogram_centers", centers, persistent=False)

    def _soft_histogram(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        centers = self.histogram_centers.to(image).view(1, 1, 1, 1, -1)
        distances = image.unsqueeze(-1) - centers
        weights = torch.exp(
            -0.5 * distances.square() / (self.config.histogram_sigma**2)
        )
        weights = weights * mask.unsqueeze(-1)
        histogram = weights.sum(dim=(1, 2, 3))
        histogram = histogram + self.config.eps
        return histogram / histogram.sum(dim=1, keepdim=True)

    def _kl_divergence(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        return (p * (p.log() - q.log())).sum(dim=1).mean()

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"KLLoss produced non-finite {name}.")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        roi_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape:
            raise ValueError(
                "KLLoss prediction/target shape mismatch: "
                f"{tuple(prediction.shape)} != {tuple(target.shape)}"
            )

        prediction_f = prediction.float()
        target_f = target.float()
        # Histogram support is restricted to valid lesion ROI pixels only.
        mask = (
            (roi_mask > 0.5).to(prediction_f.dtype)
            * (valid_mask > 0.5).to(prediction_f.dtype)
        ).expand_as(prediction_f)
        keep = mask.sum(dim=(1, 2, 3)) >= float(self.config.min_roi_pixels)
        if not torch.any(keep):
            zero = prediction_f.sum() * 0.0
            return zero, {
                "loss_kl": zero.detach(),
                "loss_kl_forward": zero.detach(),
                "loss_kl_reverse": zero.detach(),
            }

        pred = prediction_f[keep].clamp(
            -self.config.clip_sigma,
            self.config.clip_sigma,
        ) / self.config.clip_sigma
        tgt = target_f[keep].clamp(
            -self.config.clip_sigma,
            self.config.clip_sigma,
        ) / self.config.clip_sigma
        roi = mask[keep]

        pred_distribution = self._soft_histogram(pred, roi)
        target_distribution = self._soft_histogram(tgt, roi)
        self._require_finite("prediction distribution", pred_distribution)
        self._require_finite("target distribution", target_distribution)

        # Forward KL treats the target distribution as the desired reference.
        forward_kl = self._kl_divergence(
            target_distribution,
            pred_distribution,
        )
        reverse_kl = self._kl_divergence(
            pred_distribution,
            target_distribution,
        )
        total = (
            0.5 * (forward_kl + reverse_kl)
            if self.config.symmetric
            else forward_kl
        )
        self._require_finite("total", total)
        return total, {
            "loss_kl": total.detach(),
            "loss_kl_forward": forward_kl.detach(),
            "loss_kl_reverse": reverse_kl.detach(),
        }
