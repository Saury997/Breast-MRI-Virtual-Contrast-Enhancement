#!/usr/bin/env python3
"""Train a Lightning residual Res-UNet baseline for MAMA-SYNTH."""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parent
SRC_DIR = TRAINING_DIR.parent
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import lightning.pytorch as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from training.augmentations import AugmentationConfig
from training.config import get_nested, load_yaml_config, save_yaml_config
from training.data import MamaSynthDataModule
from training.losses import (
    AuxiliarySegmentationLoss,
    LossWeights,
    SegLoss,
    WaveletLoss,
    WaveletLossConfig,
)
from training.models import create_model
from training.modules import create_lightning_module
from training.segmenter import SegmenterModule, create_segmenter_model
from training.kl_loss import KLLoss, KLLossConfig


DEFAULT_CONFIG_PATH = TRAINING_DIR / "configs" / "resunet_baseline.yaml"


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def launch_tensorboard(logdir: Path, port: int) -> subprocess.Popen | None:
    if is_port_in_use(port):
        print(
            f"TensorBoard was not launched because port {port} is already in use. "
            f"Open the existing service or pass --tensorboard_port to choose another port."
        )
        return None

    tensorboard_exe = shutil.which("tensorboard")
    command = (
        [tensorboard_exe]
        if tensorboard_exe is not None
        else [sys.executable, "-m", "tensorboard.main"]
    )
    command.extend(
        [
            "--logdir",
            str(logdir),
            "--port",
            str(port),
            "--host",
            "localhost",
        ]
    )
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    print(f"TensorBoard launched at http://localhost:{port} (logdir: {logdir})")
    return process


def next_logger_version(exp_dir: Path) -> int:
    if not exp_dir.exists():
        return 0
    versions: list[int] = []
    for child in exp_dir.iterdir():
        if child.is_dir() and child.name.startswith("version_"):
            suffix = child.name.removeprefix("version_")
            if suffix.isdigit():
                versions.append(int(suffix))
    return max(versions, default=-1) + 1


def resolve_path(value: str | Path | None, base_dir: Path = REPO_DIR) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def resolve_validation_enabled(config: dict, val_record_count: int) -> bool:
    enabled = get_nested(config, "trainer.enable_validation", None)
    if enabled is None:
        return val_record_count > 0
    if not isinstance(enabled, bool):
        raise ValueError("trainer.enable_validation must be true, false, or omitted")
    if enabled and val_record_count <= 0:
        raise ValueError(
            "trainer.enable_validation=true but the data split produced no "
            "validation cases."
        )
    return enabled


def normalize_seg_modes(seg_config: dict) -> tuple[str, ...]:
    raw_modes = seg_config.get("modes", seg_config.get("mode", "frozen_segmenter"))
    if isinstance(raw_modes, str):
        value = raw_modes.lower().replace("+", ",").replace("|", ",")
        aliases = {
            "both": "frozen_segmenter,auxiliary_branch",
            "combined": "frozen_segmenter,auxiliary_branch",
            "frozen_and_auxiliary": "frozen_segmenter,auxiliary_branch",
            "frozen_segmenter_and_auxiliary_branch": (
                "frozen_segmenter,auxiliary_branch"
            ),
        }
        value = aliases.get(value, value)
        modes = [mode.strip() for mode in value.split(",") if mode.strip()]
    elif isinstance(raw_modes, (list, tuple)):
        modes = [str(mode).lower().strip() for mode in raw_modes if str(mode).strip()]
    else:
        raise ValueError("loss.seg.mode or loss.seg.modes must be a string or list")

    supported = {"frozen_segmenter", "auxiliary_branch"}
    unknown = sorted(set(modes) - supported)
    if unknown:
        raise ValueError(
            "Unsupported loss.seg mode(s): "
            + ", ".join(unknown)
            + ". Available: auxiliary_branch, frozen_segmenter."
        )

    deduped: list[str] = []
    for mode in modes:
        if mode not in deduped:
            deduped.append(mode)
    if not deduped:
        raise ValueError("loss.seg.enabled=true requires at least one mode")
    return tuple(deduped)


def create_seg_losses_from_config(
    config: dict,
) -> tuple[SegLoss | None, AuxiliarySegmentationLoss | None]:
    seg_config = get_nested(config, "loss.seg", {}) or {}
    if not isinstance(seg_config, dict):
        raise ValueError("loss.seg config must be a mapping")
    if not bool(seg_config.get("enabled", False)):
        return None, None

    modes = normalize_seg_modes(seg_config)
    use_frozen = "frozen_segmenter" in modes
    use_auxiliary = "auxiliary_branch" in modes

    aux_seg_loss = None
    if use_auxiliary:
        aux_seg_loss = AuxiliarySegmentationLoss(
            threshold=float(seg_config.get("threshold", 0.5)),
            dice_weight=float(seg_config.get("dice_weight", 0.5)),
            focal_weight=float(seg_config.get("focal_weight", 0.5)),
            focal_alpha=float(seg_config.get("focal_alpha", 0.25)),
            focal_gamma=float(seg_config.get("focal_gamma", 2.0)),
            boundary_weight=float(seg_config.get("boundary_weight", 0.05)),
        )

    if not use_frozen:
        if "checkpoint" in seg_config or "model" in seg_config:
            raise ValueError(
                "loss.seg auxiliary-only mode must not set checkpoint or model. "
                "Add frozen_segmenter to loss.seg.modes to use the pretrained S path."
            )
        return None, aux_seg_loss

    checkpoint = resolve_path(seg_config.get("checkpoint"))
    if checkpoint is None:
        raise ValueError(
            "loss.seg.checkpoint is required when frozen_segmenter is enabled"
        )
    if not checkpoint.exists():
        raise FileNotFoundError(f"SegLoss segmenter checkpoint not found: {checkpoint}")

    model_config = seg_config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError(
            "loss.seg.model must be a mapping when frozen_segmenter is enabled"
        )
    model = create_segmenter_model(**dict(model_config))
    segmenter = SegmenterModule.load_from_checkpoint(
        str(checkpoint),
        model=model,
        map_location="cpu",
    )
    print(f"Loaded frozen SegLoss segmenter from {checkpoint}")
    return (
        SegLoss(
            segmenter=segmenter,
            threshold=float(seg_config.get("threshold", 0.5)),
            dice_weight=float(seg_config.get("dice_weight", 0.5)),
            focal_weight=float(seg_config.get("focal_weight", 0.5)),
            focal_alpha=float(seg_config.get("focal_alpha", 0.25)),
            focal_gamma=float(seg_config.get("focal_gamma", 2.0)),
            boundary_weight=float(seg_config.get("boundary_weight", 0.05)),
        ),
        aux_seg_loss,
    )


def create_kl_loss_from_config(
    config: dict,
    weight: float,
) -> KLLoss | None:
    kl_config = get_nested(config, "loss.kl", {}) or {}
    if not isinstance(kl_config, dict):
        raise ValueError("loss.kl config must be a mapping")
    if not bool(kl_config.get("enabled", weight > 0)) or weight <= 0:
        return None

    values = dict(kl_config)
    values.pop("enabled", None)
    return KLLoss(KLLossConfig(**values))


def create_wavelet_loss_from_config(
    config: dict,
    weight: float,
) -> WaveletLoss | None:
    wavelet_config = get_nested(config, "loss.wavelet", {}) or {}
    if not isinstance(wavelet_config, dict):
        raise ValueError("loss.wavelet config must be a mapping")
    if not bool(wavelet_config.get("enabled", weight > 0)) or weight <= 0:
        return None

    values = dict(wavelet_config)
    values.pop("enabled", None)
    return WaveletLoss(WaveletLossConfig(**values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MAMA-SYNTH residual Res-UNet baseline."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)

    seed = int(get_nested(config, "experiment.seed", 42))
    experiment_name = str(get_nested(config, "experiment.name", "resunet_baseline"))
    experiments_dir = resolve_path(get_nested(config, "experiment.experiments_dir", "experiments"))
    checkpoints_dir = resolve_path(get_nested(config, "experiment.checkpoints_dir", "checkpoints"))
    if experiments_dir is None or checkpoints_dir is None:
        raise ValueError("experiment.experiments_dir and experiment.checkpoints_dir are required")
    exp_dir = experiments_dir / experiment_name
    ckpt_dir = checkpoints_dir / experiment_name
    logger_version = next_logger_version(exp_dir)
    version_name = f"version_{logger_version}"
    log_version_dir = exp_dir / version_name
    ckpt_version_dir = ckpt_dir / version_name

    L.seed_everything(seed, workers=True)

    augmentation_config = AugmentationConfig(
        **dict(get_nested(config, "data.augmentation", {}) or {})
    )
    data_root = resolve_path(get_nested(config, "data.root"))
    if data_root is None:
        raise ValueError("data.root is required in the training YAML config")
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
    validation_enabled = resolve_validation_enabled(config, len(data_module.val_records))
    print(
        f"Discovered {len(data_module.train_records)} train and "
        f"{len(data_module.val_records)} validation cases."
    )
    if not validation_enabled:
        reason = (
            "the data split has no validation cases"
            if not data_module.val_records
            else "trainer.enable_validation=false"
        )
        print(f"Validation is disabled because {reason}. Training will save last.ckpt only.")

    model_config = get_nested(config, "model", {"type": "resunet"})
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a mapping")
    model = create_model(**dict(model_config))
    module_type = str(get_nested(config, "module.type", "translation"))
    gradient_clip_val = get_nested(config, "trainer.gradient_clip_val", None)
    loss_weights = LossWeights(
        **dict(get_nested(config, "loss.weights", {}) or {})
    )
    kl_loss = create_kl_loss_from_config(
        config,
        weight=loss_weights.kl,
    )
    wavelet_loss = create_wavelet_loss_from_config(
        config,
        weight=loss_weights.wavelet,
    )
    seg_loss, aux_seg_loss = create_seg_losses_from_config(config)
    if aux_seg_loss is not None and not bool(
        getattr(model, "aux_segmentation_enabled", False)
    ):
        raise ValueError(
            "loss.seg auxiliary_branch mode requires "
            "model.aux_segmentation.enabled=true."
        )
    if bool(getattr(model, "aux_attention_enabled", False)) and aux_seg_loss is None:
        raise ValueError(
            "model.aux_segmentation.attention.enabled=true requires "
            "loss.seg auxiliary_branch mode so the attention mask is supervised."
        )
    adversarial_config = get_nested(config, "adversarial", {}) or {}
    if not isinstance(adversarial_config, dict):
        raise ValueError("adversarial config must be a mapping")
    adversarial_config = dict(adversarial_config)
    if module_type.lower() == "adversarial_translation":
        adversarial_config.setdefault("gradient_clip_val", gradient_clip_val)
    official_eval_every_n_epochs = int(
        get_nested(config, "evaluation.official_eval_every_n_epochs", 10)
    )
    visualize_val_samples = bool(
        get_nested(config, "logging.visualize_val_samples", True)
    )
    if not validation_enabled:
        official_eval_every_n_epochs = 0
        visualize_val_samples = False
    module = create_lightning_module(
        module_type,
        model=model,
        learning_rate=float(get_nested(config, "optimizer.learning_rate", 2e-4)),
        weight_decay=float(get_nested(config, "optimizer.weight_decay", 1e-4)),
        optimizer_name=str(get_nested(config, "optimizer.type", "adamw")),
        max_epochs=int(get_nested(config, "trainer.max_epochs", 500)),
        loss_weights=loss_weights,
        official_models_dir=str(
            resolve_path(
                get_nested(config, "evaluation.official_models_dir", SRC_DIR / "evaluation" / "models")
            )
        ),
        official_eval_predictions_dir=log_version_dir / "official_eval_predictions",
        val_visualizations_dir=exp_dir / "val_visualizations",
        official_eval_every_n_epochs=official_eval_every_n_epochs,
        visualize_val_samples=visualize_val_samples,
        num_visualization_cases=int(get_nested(config, "logging.num_visualization_cases", 4)),
        visualization_seed=seed,
        seg_loss=seg_loss,
        aux_seg_loss=aux_seg_loss,
        adversarial_config=adversarial_config,
        kl_loss=kl_loss,
        wavelet_loss=wavelet_loss,
    )

    csv_logger = CSVLogger(
        save_dir=str(experiments_dir),
        name=experiment_name,
        version=logger_version,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(experiments_dir),
        name=experiment_name,
        version=logger_version,
    )
    if bool(get_nested(config, "logging.launch_tensorboard", True)):
        launch_tensorboard(exp_dir, int(get_nested(config, "logging.tensorboard_port", 6006)))

    checkpoint_filename = str(
        get_nested(
            config,
            "checkpoint.filename",
            "{v_num:02d}-{epoch:03d}-{val_loss:.5f}",
        )
    ).replace("{v_num:02d}", f"{logger_version:02d}")
    checkpoint_monitor = get_nested(config, "checkpoint.monitor", "val_loss")
    checkpoint_save_top_k = int(get_nested(config, "checkpoint.save_top_k", 3))
    if not validation_enabled:
        checkpoint_monitor = None
        checkpoint_save_top_k = 0
    elif checkpoint_monitor is not None:
        checkpoint_monitor = str(checkpoint_monitor)
    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_version_dir),
        filename=checkpoint_filename,
        monitor=checkpoint_monitor,
        mode=str(get_nested(config, "checkpoint.mode", "min")),
        save_top_k=checkpoint_save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [
        checkpoint,
        LearningRateMonitor(logging_interval=str(get_nested(config, "logging.lr_logging_interval", "epoch"))),
    ]
    if validation_enabled and bool(get_nested(config, "early_stopping.enabled", True)):
        callbacks.append(
            EarlyStopping(
                monitor=str(get_nested(config, "early_stopping.monitor", "val_loss")),
                patience=int(get_nested(config, "early_stopping.patience", 25)),
                mode=str(get_nested(config, "early_stopping.mode", "min")),
                min_delta=float(get_nested(config, "early_stopping.min_delta", 0.0)),
                strict=True,
                verbose=True,
            )
        )
    if bool(get_nested(config, "logging.rich_progress", True)):
        callbacks.append(RichProgressBar())
    model_summary_depth = get_nested(config, "logging.model_summary_depth", 2)
    if model_summary_depth is not None:
        callbacks.append(RichModelSummary(max_depth=int(model_summary_depth)))

    trainer_gradient_clip_val = (
        None if module_type.lower() == "adversarial_translation" else gradient_clip_val
    )
    limit_val_batches = (
        get_nested(config, "trainer.limit_val_batches", 1.0)
        if validation_enabled
        else 0
    )
    num_sanity_val_steps = (
        int(get_nested(config, "trainer.num_sanity_val_steps", 2))
        if validation_enabled
        else 0
    )
    trainer = L.Trainer(
        max_epochs=int(get_nested(config, "trainer.max_epochs", 500)),
        accelerator=str(get_nested(config, "trainer.accelerator", "auto")),
        devices=get_nested(config, "trainer.devices", "auto"),
        precision=str(get_nested(config, "trainer.precision", "16-mixed")),
        logger=[csv_logger, tensorboard_logger],
        callbacks=callbacks,
        default_root_dir=str(exp_dir),
        limit_train_batches=get_nested(config, "trainer.limit_train_batches", 1.0),
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=num_sanity_val_steps,
        fast_dev_run=bool(get_nested(config, "trainer.fast_dev_run", False)),
        log_every_n_steps=int(get_nested(config, "logging.log_every_n_steps", 10)),
        check_val_every_n_epoch=int(get_nested(config, "trainer.check_val_every_n_epoch", 1)),
        accumulate_grad_batches=int(get_nested(config, "trainer.accumulate_grad_batches", 1)),
        gradient_clip_val=trainer_gradient_clip_val,
    )
    resume_checkpoint = resolve_path(get_nested(config, "trainer.resume_checkpoint", None))
    trainer.fit(module, datamodule=data_module, ckpt_path=resume_checkpoint)
    save_yaml_config(config, log_version_dir / "config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
