"""Adapter for calling the official MAMA-SYNTH evaluation from training."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass
class ValidationPrediction:
    case_id: str
    prediction: np.ndarray
    target: np.ndarray
    mask: np.ndarray
    precontrast: np.ndarray
    input_path: str
    target_path: str
    mask_path: str
    aux_segmentation: np.ndarray | None = None


def write_prediction_like_input(
    prediction: np.ndarray,
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    input_image = sitk.ReadImage(str(input_path))
    out_image = sitk.GetImageFromArray(prediction.astype(np.float32))
    out_image.CopyInformation(input_image)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_image, str(output_path), useCompression=True)


def save_validation_predictions(
    predictions: list[ValidationPrediction],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save validation predictions as MHA files and return case-to-path mapping."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for item in predictions:
        pred_path = output_dir / f"{item.case_id}.mha"
        write_prediction_like_input(item.prediction, item.input_path, pred_path)
        output_paths[item.case_id] = pred_path
    return output_paths


def run_official_evaluation(
    predictions: list[ValidationPrediction],
    output_dir: str | Path,
    models_dir: str | Path | None = None,
) -> dict:
    src_dir = Path(__file__).resolve().parents[1]
    evaluation_dir = src_dir / "evaluation"
    for path in (str(evaluation_dir), str(src_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from evaluate import run_evaluation  # type: ignore
    from evaluators import Case  # type: ignore

    output_dir = Path(output_dir)
    prediction_paths = save_validation_predictions(predictions, output_dir)
    cases = []
    for item in predictions:
        cases.append(
            Case(
                case_id=item.case_id,
                prediction=item.prediction.astype(np.float64),
                ground_truth=item.target.astype(np.float64),
                mask=(item.mask > 0).astype(bool),
                precontrast=item.precontrast.astype(np.float64),
                prediction_path=str(prediction_paths[item.case_id]),
                ground_truth_path=item.target_path,
                mask_path=item.mask_path,
            )
        )

    resolved_models = Path(models_dir) if models_dir else None
    if resolved_models is not None and not resolved_models.exists():
        resolved_models = None
    return run_evaluation(cases, resolved_models)
