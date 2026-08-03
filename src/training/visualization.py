"""Validation visualization utilities for synthesis training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from training.official_eval import ValidationPrediction

if TYPE_CHECKING:
    import lightning.pytorch as L


BASE_COLUMN_TITLES = ("Input", "Prediction", "GT", "Diff", "Tumor Mask")


def select_visualization_cases(
    predictions: list[ValidationPrediction],
    count: int,
    seed: int,
) -> list[ValidationPrediction]:
    """Select a stable random subset from validation predictions."""
    if count <= 0 or not predictions:
        return []
    rng = random.Random(seed)
    indices = list(range(len(predictions)))
    selected = rng.sample(indices, k=min(count, len(indices)))
    return [predictions[idx] for idx in sorted(selected)]


def _robust_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        center = float(np.nanmean(finite))
        return center - 1.0, center + 1.0
    return float(vmin), float(vmax)


def _case_images(item: ValidationPrediction) -> tuple[np.ndarray, ...]:
    input_image = np.asarray(item.precontrast, dtype=np.float32)
    prediction = np.asarray(item.prediction, dtype=np.float32)
    target = np.asarray(item.target, dtype=np.float32)
    diff = np.abs(prediction - target)
    mask = (np.asarray(item.mask) > 0).astype(np.float32)
    return input_image, prediction, target, diff, mask


def make_validation_grid(
    predictions: list[ValidationPrediction],
) -> plt.Figure:
    """Create validation rows with an optional auxiliary segmentation column."""
    if not predictions:
        raise ValueError("Cannot create validation grid without predictions.")

    include_aux_seg = any(item.aux_segmentation is not None for item in predictions)
    column_titles = (
        (*BASE_COLUMN_TITLES, "Aux Seg") if include_aux_seg else BASE_COLUMN_TITLES
    )
    rows = len(predictions)
    fig, axes = plt.subplots(
        rows,
        len(column_titles),
        figsize=(3.2 * len(column_titles), 3.0 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    for row, item in enumerate(predictions):
        input_image, prediction, target, diff, mask = _case_images(item)
        image_limits = _robust_limits(np.stack([input_image, prediction, target]))
        diff_limits = _robust_limits(diff)
        values = [input_image, prediction, target, diff, mask]
        cmaps = ["gray", "gray", "gray", "magma", "gray"]
        limits = [image_limits, image_limits, image_limits, diff_limits, (0.0, 1.0)]
        if include_aux_seg:
            aux_seg = (
                np.zeros_like(mask, dtype=np.float32)
                if item.aux_segmentation is None
                else np.asarray(item.aux_segmentation, dtype=np.float32)
            )
            values.append(np.clip(aux_seg, 0.0, 1.0))
            cmaps.append("viridis")
            limits.append((0.0, 1.0))

        for col, (value, cmap, (vmin, vmax)) in enumerate(zip(values, cmaps, limits)):
            ax = axes[row, col]
            ax.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis("off")
            if row == 0:
                ax.set_title(column_titles[col], fontsize=11)
            if col == 0:
                ax.text(
                    0.01,
                    0.99,
                    item.case_id,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
                )
    return fig


def save_validation_grid(fig: plt.Figure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")


def close_validation_grid(fig: plt.Figure) -> None:
    plt.close(fig)


def log_validation_grid_to_tensorboard(
    trainer: "L.Trainer",
    fig: plt.Figure,
    tag: str,
    epoch: int,
) -> None:
    """Log a matplotlib figure to any TensorBoard-compatible Lightning logger."""
    loggers = trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]
    for logger in loggers:
        experiment = getattr(logger, "experiment", None)
        if hasattr(experiment, "add_figure"):
            experiment.add_figure(tag, fig, global_step=epoch)
            experiment.flush()
