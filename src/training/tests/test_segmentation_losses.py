from pathlib import Path
import sys

import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training.losses import (
    AuxiliarySegmentationLoss,
    LossWeights,
    SegLoss,
    SynthesisLoss,
)


def test_auxiliary_branch_uses_same_logits_loss_as_frozen_segmenter() -> None:
    logits = torch.randn(2, 1, 32, 32)
    mask = torch.zeros_like(logits)
    mask[:, :, 8:18, 10:22] = 1.0
    valid_mask = torch.ones_like(logits)
    kwargs = {
        "threshold": 0.5,
        "dice_weight": 0.5,
        "focal_weight": 0.25,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "boundary_weight": 0.1,
    }

    frozen_loss = SegLoss(segmenter=nn.Identity(), **kwargs)
    aux_loss = AuxiliarySegmentationLoss(**kwargs)

    frozen_value, frozen_parts = frozen_loss(logits, mask, valid_mask)
    aux_value, aux_parts = aux_loss(logits, mask, valid_mask)

    torch.testing.assert_close(aux_value, frozen_value)
    for key in ("loss_seg", "loss_seg_dice", "loss_seg_focal", "loss_seg_boundary"):
        torch.testing.assert_close(aux_parts[key], frozen_parts[key])
    assert "aux_seg_dice" in aux_parts
    assert "loss_aux_seg_bce" not in aux_parts


def test_synthesis_loss_allows_frozen_and_auxiliary_segmentation() -> None:
    logits = torch.randn(2, 1, 32, 32)
    target = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    mask[:, :, 8:18, 10:22] = 1.0
    valid_mask = torch.ones_like(logits)
    kwargs = {
        "threshold": 0.5,
        "dice_weight": 0.5,
        "focal_weight": 0.25,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "boundary_weight": 0.1,
    }
    loss_fn = SynthesisLoss(
        weights=LossWeights(
            l1=0.0,
            mse=0.0,
            roi_l1=0.0,
            ssim=0.0,
            gradient=0.0,
            roi_ssim=0.0,
            roi_gradient=0.0,
            perceptual=0.0,
            seg=0.2,
            kl=0.0,
            wavelet=0.0,
        ),
        seg_loss=SegLoss(segmenter=nn.Identity(), **kwargs),
        aux_seg_loss=AuxiliarySegmentationLoss(**kwargs),
    )

    value, parts = loss_fn(
        logits,
        target,
        mask,
        valid_mask,
        segmentation_logits=logits,
    )

    assert "loss_seg" in parts
    assert "loss_seg_frozen" in parts
    assert "loss_seg_aux" in parts
    assert "loss_seg_frozen_dice" in parts
    assert "loss_seg_aux_dice" in parts
    assert "loss_seg_frozen_focal" in parts
    assert "loss_seg_aux_focal" in parts
    assert "loss_seg_frozen_boundary" in parts
    assert "loss_seg_aux_boundary" in parts
    assert "aux_seg_dice" in parts
    torch.testing.assert_close(
        parts["loss_seg"],
        parts["loss_seg_frozen"] + parts["loss_seg_aux"],
    )
    torch.testing.assert_close(value, 0.2 * parts["loss_seg"])
