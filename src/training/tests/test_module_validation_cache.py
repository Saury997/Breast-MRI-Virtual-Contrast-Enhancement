from pathlib import Path
import sys

import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training.losses import LossWeights
from training.modules import ResidualTranslationModule


class ZeroResidual(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(image)


def test_validation_predictions_not_cached_when_all_consumers_disabled() -> None:
    module = ResidualTranslationModule(
        model=ZeroResidual(),
        loss_weights=LossWeights(l1=1.0, mse=0.0, ssim=0.0),
        visualize_val_samples=False,
        official_eval_every_n_epochs=0,
    )

    assert module.should_collect_validation_predictions() is False


def test_validation_predictions_cached_when_visualization_enabled() -> None:
    module = ResidualTranslationModule(
        model=ZeroResidual(),
        loss_weights=LossWeights(l1=1.0, mse=0.0, ssim=0.0),
        visualize_val_samples=True,
        num_visualization_cases=1,
        official_eval_every_n_epochs=0,
    )

    assert module.should_collect_validation_predictions() is True
