from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training.official_eval import ValidationPrediction
from training.visualization import close_validation_grid, make_validation_grid


def _prediction(aux_segmentation: np.ndarray | None = None) -> ValidationPrediction:
    image = np.zeros((16, 16), dtype=np.float32)
    mask = np.zeros_like(image)
    mask[4:10, 5:11] = 1.0
    return ValidationPrediction(
        case_id="case_001",
        prediction=image,
        target=image,
        mask=mask,
        precontrast=image,
        input_path="input.mha",
        target_path="target.mha",
        mask_path="mask.mha",
        aux_segmentation=aux_segmentation,
    )


def test_validation_grid_keeps_base_columns_without_aux_segmentation() -> None:
    fig = make_validation_grid([_prediction()])
    try:
        assert len(fig.axes) == 5
        assert [axis.get_title() for axis in fig.axes] == [
            "Input",
            "Prediction",
            "GT",
            "Diff",
            "Tumor Mask",
        ]
    finally:
        close_validation_grid(fig)


def test_validation_grid_adds_aux_segmentation_column() -> None:
    fig = make_validation_grid([
        _prediction(aux_segmentation=np.ones((16, 16), dtype=np.float32) * 0.75)
    ])
    try:
        assert len(fig.axes) == 6
        assert fig.axes[-1].get_title() == "Aux Seg"
    finally:
        close_validation_grid(fig)
