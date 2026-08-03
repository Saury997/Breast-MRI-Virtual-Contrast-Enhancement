#!/usr/bin/env python3
"""Evaluate a trained synthesis checkpoint on the configured validation set.

This script is a thin offline wrapper around the official ``evaluate.py``:
it reconstructs the training validation split from a YAML config, writes
checkpoint predictions as .mha files, then calls the official local
evaluation functions on those files.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any


EVALUATION_DIR = Path(__file__).resolve().parent
SRC_DIR = EVALUATION_DIR.parent
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import lightning.pytorch as L
import torch
from tqdm import tqdm

from evaluate import load_cases_local, run_evaluation, write_metrics
from training.augmentations import AugmentationConfig
from training.config import get_nested, load_yaml_config
from training.data import MamaSynthDataModule
from training.models import create_model, is_smp_model_type
from training.modules import ResidualTranslationModule
from training.official_eval import write_prediction_like_input


DEFAULT_CONFIG_PATH = SRC_DIR / "training" / "configs" / "resunet_baseline.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained MAMA-SYNTH checkpoint on the YAML validation set "
            "and score it with the official evaluation code."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_dir/predictions before writing new predictions.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path | None, base_dir: Path = REPO_DIR) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def default_output_dir(config_path: Path, checkpoint_path: Path) -> Path:
    return (
        REPO_DIR
        / "outputs"
        / "eval"
        / config_path.stem
        / checkpoint_path.stem
    )


def require_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {path}")


def prepare_predictions_dir(predictions_dir: Path, overwrite: bool) -> None:
    if predictions_dir.exists():
        existing = list(predictions_dir.glob("*.mha"))
        if existing and not overwrite:
            raise FileExistsError(
                f"Prediction directory already contains {len(existing)} .mha file(s): "
                f"{predictions_dir}. Pass --overwrite to regenerate them."
            )
        if overwrite:
            shutil.rmtree(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)


def build_data_module(config: dict[str, Any], data_root: Path) -> MamaSynthDataModule:
    seed = int(get_nested(config, "experiment.seed", 42))
    augmentation_config = AugmentationConfig(
        **dict(get_nested(config, "data.augmentation", {}) or {})
    )
    split_file = resolve_path(get_nested(config, "data.split_file", None))
    data_module = MamaSynthDataModule(
        data_root=data_root,
        batch_size=int(get_nested(config, "data.batch_size", 8)),
        num_workers=int(get_nested(config, "data.num_workers", 4)),
        val_fraction=float(get_nested(config, "data.val_fraction", 0.1)),
        seed=seed,
        augment=bool(get_nested(config, "data.augment", True)),
        augmentation_config=augmentation_config,
        pad_multiple=int(get_nested(config, "data.pad_multiple", 16)),
        split_mode=str(get_nested(config, "data.split_mode", "random")),
        split_file=split_file,
    )
    data_module.setup("fit")
    if not data_module.val_records:
        raise RuntimeError("Validation split is empty; nothing to evaluate.")
    return data_module


def load_module(
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> ResidualTranslationModule:
    module_type = str(get_nested(config, "module.type", "translation")).lower()
    supported_module_types = {"translation", "adversarial_translation"}
    if module_type not in supported_module_types:
        raise ValueError(
            f"Unsupported module.type={module_type!r}. "
            "eval.py currently supports 'translation' and "
            "'adversarial_translation'."
        )

    model_config = get_nested(config, "model", {"type": "resunet"})
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a mapping")
    model_config = dict(model_config)
    model_type = str(model_config.get("type", "resunet")).lower()
    if is_smp_model_type(model_type):
        # Checkpoint loading replaces model weights; skipping pretrained
        # initialization avoids downloads and keeps evaluation usable offline.
        model_config["encoder_weights"] = None
    model = create_model(**model_config)
    module = ResidualTranslationModule(model=model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no valid state_dict: {checkpoint_path}")
    model_state = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(("loss_fn.", "discriminator.", "adv_loss."))
    }
    module.load_state_dict(model_state, strict=True)
    module.eval().to(device)
    return module


def write_validation_predictions(
    module: ResidualTranslationModule,
    data_module: MamaSynthDataModule,
    predictions_dir: Path,
    device: torch.device,
) -> None:
    with torch.no_grad():
        for batch in tqdm(
            data_module.val_dataloader(),
            desc="Predicting validation",
            unit="batch",
        ):
            inputs = batch["input"].to(device, non_blocking=True)
            outputs = module(inputs)
            for idx, case_id in enumerate(batch["case_id"]):
                h, w = [int(v) for v in batch["original_size"][idx].tolist()]
                prediction = outputs[idx, 0, :h, :w].detach().cpu().numpy()
                output_path = predictions_dir / f"{case_id}.mha"
                write_prediction_like_input(
                    prediction,
                    input_path=batch["input_path"][idx],
                    output_path=output_path,
                )


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    if config_path is None:
        raise ValueError("--config is required")
    if checkpoint_path is None:
        raise ValueError("--checkpoint is required")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else default_output_dir(config_path, checkpoint_path)
    )
    if output_dir is None:
        raise ValueError("Could not resolve output directory")
    predictions_dir = output_dir / "predictions"
    metrics_path = output_dir / "metrics.json"

    config = load_yaml_config(config_path)
    data_root = resolve_path(get_nested(config, "data.root", None))
    if data_root is None:
        raise ValueError("data.root is required in the YAML config")
    input_dir = data_root / "input"
    gt_dir = data_root / "ground_truth"
    masks_dir = data_root / "mask"
    for label, path in (
        ("Pre-contrast input", input_dir),
        ("Ground-truth", gt_dir),
        ("Mask", masks_dir),
    ):
        require_dir(path, label)

    official_models_dir = resolve_path(
        get_nested(
            config,
            "evaluation.official_models_dir",
            SRC_DIR / "evaluation" / "models",
        )
    )
    if official_models_dir is not None and not official_models_dir.exists():
        print(
            "Warning: official evaluation models directory not found; "
            f"classification and segmentation metrics will be disabled: {official_models_dir}"
        )
        official_models_dir = None

    L.seed_everything(int(get_nested(config, "experiment.seed", 42)), workers=True)
    device = resolve_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_predictions_dir(predictions_dir, args.overwrite)

    print(f"Config      : {config_path}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Data root   : {data_root}")
    print(f"Output      : {output_dir}")
    print(f"Device      : {device}")

    data_module = build_data_module(config, data_root)
    print(
        f"Discovered {len(data_module.train_records)} train and "
        f"{len(data_module.val_records)} validation cases."
    )
    module = load_module(config, checkpoint_path, device)
    write_validation_predictions(module, data_module, predictions_dir, device)

    cases = load_cases_local(
        pred_dir=predictions_dir,
        gt_dir=gt_dir,
        masks_dir=masks_dir,
        precon_dir=input_dir,
    )
    if not cases:
        raise RuntimeError(
            "Official evaluator could not load any valid validation cases. "
            "Check prediction filenames, image shapes, and data.root paths."
        )
    metrics = run_evaluation(cases, official_models_dir)
    write_metrics(metrics, metrics_path)
    print(f"Metrics written to: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
