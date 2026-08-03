"""Task-aware paired augmentations for 2-D MAMA-SYNTH slices."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass
class AugmentationConfig:
    hflip_prob: float = 0.5
    vflip_prob: float = 0.1
    affine_prob: float = 0.8
    max_rotate_deg: float = 10.0
    max_translate_frac: float = 0.04
    min_scale: float = 0.95
    max_scale: float = 1.05
    elastic_prob: float = 0.15
    elastic_grid: int = 8
    elastic_magnitude: float = 3.0
    noise_prob: float = 0.15
    noise_std: float = 0.02


class PairedSliceAugmentor:
    """Apply identical geometry to input, target, and mask tensors."""

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()

    def __call__(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        if random.random() < cfg.hflip_prob:
            image = TF.hflip(image)
            target = TF.hflip(target)
            mask = TF.hflip(mask)

        if random.random() < cfg.vflip_prob:
            image = TF.vflip(image)
            target = TF.vflip(target)
            mask = TF.vflip(mask)

        if random.random() < cfg.affine_prob:
            _, h, w = image.shape
            max_dx = int(round(cfg.max_translate_frac * w))
            max_dy = int(round(cfg.max_translate_frac * h))
            angle = random.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg)
            translate = (
                random.randint(-max_dx, max_dx),
                random.randint(-max_dy, max_dy),
            )
            scale = random.uniform(cfg.min_scale, cfg.max_scale)
            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=(0.0, 0.0),
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            target = TF.affine(
                target,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=(0.0, 0.0),
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=(0.0, 0.0),
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )

        if random.random() < cfg.elastic_prob:
            image, target, mask = self._elastic(image, target, mask)

        if random.random() < cfg.noise_prob:
            image = image + torch.randn_like(image) * cfg.noise_std

        return image.contiguous(), target.contiguous(), (mask > 0.5).float()

    def _elastic(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        _, h, w = image.shape
        low_h = max(2, h // cfg.elastic_grid)
        low_w = max(2, w // cfg.elastic_grid)
        disp = torch.randn(1, 2, low_h, low_w, dtype=image.dtype)
        disp = F.interpolate(disp, size=(h, w), mode="bicubic", align_corners=False)
        disp = disp[0].permute(1, 2, 0)
        disp[..., 0] = disp[..., 0] * (2.0 * cfg.elastic_magnitude / max(w - 1, 1))
        disp[..., 1] = disp[..., 1] * (2.0 * cfg.elastic_magnitude / max(h - 1, 1))

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, w, dtype=image.dtype),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1) + disp
        grid = grid.unsqueeze(0)

        def warp(tensor: torch.Tensor, mode: str) -> torch.Tensor:
            return F.grid_sample(
                tensor.unsqueeze(0),
                grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0)

        return warp(image, "bilinear"), warp(target, "bilinear"), warp(mask, "nearest")

