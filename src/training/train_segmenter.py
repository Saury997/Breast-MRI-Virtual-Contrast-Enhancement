#!/usr/bin/env python3
"""Pretrain a lightweight lesion ROI segmenter for downstream validation."""

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
from training.segmenter import (
    SegmenterLossWeights,
    SegmenterModule,
    create_segmenter_model,
)


DEFAULT_CONFIG_PATH = TRAINING_DIR / "configs" / "segmenter_unet_lite.yaml"


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def launch_tensorboard(logdir: Path, port: int) -> subprocess.Popen | None:
    if is_port_in_use(port):
        print(
            f"TensorBoard was not launched because port {port} is already in use. "
            "Open the existing service or choose another logging.tensorboard_port."
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain a lightweight MAMA-SYNTH lesion ROI segmenter."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)

    seed = int(get_nested(config, "experiment.seed", 42))
    experiment_name = str(get_nested(config, "experiment.name", "segmenter_unet_lite"))
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

    data_root = resolve_path(get_nested(config, "data.root"))
    if data_root is None:
        raise ValueError("data.root is required in the segmenter YAML config")
    split_file = resolve_path(get_nested(config, "data.split_file", None))
    augmentation_config = AugmentationConfig(
        **dict(get_nested(config, "data.augmentation", {}) or {})
    )
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
    print(
        f"Discovered {len(data_module.train_records)} train and "
        f"{len(data_module.val_records)} validation cases."
    )

    model_config = get_nested(config, "model", {"type": "tiny_unet"})
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a mapping")
    model = create_segmenter_model(**dict(model_config))

    loss_config = get_nested(config, "loss", {}) or {}
    if not isinstance(loss_config, dict):
        raise ValueError("loss config must be a mapping")
    module = SegmenterModule(
        model=model,
        learning_rate=float(get_nested(config, "optimizer.learning_rate", 1e-3)),
        weight_decay=float(get_nested(config, "optimizer.weight_decay", 1e-4)),
        optimizer_name=str(get_nested(config, "optimizer.type", "adamw")),
        max_epochs=int(get_nested(config, "trainer.max_epochs", 100)),
        image_key=str(get_nested(config, "data.image_key", "target")),
        threshold=float(get_nested(config, "metric.threshold", 0.5)),
        loss_weights=SegmenterLossWeights(**dict(loss_config)),
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
        launch_tensorboard(exp_dir, int(get_nested(config, "logging.tensorboard_port", 6007)))

    checkpoint_filename = str(
        get_nested(config, "checkpoint.filename", "{v_num:02d}-{epoch:03d}-{val_dice:.5f}")
    ).replace("{v_num:02d}", f"{logger_version:02d}")
    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_version_dir),
        filename=checkpoint_filename,
        monitor=str(get_nested(config, "checkpoint.monitor", "val_dice")),
        mode=str(get_nested(config, "checkpoint.mode", "max")),
        save_top_k=int(get_nested(config, "checkpoint.save_top_k", 1)),
        save_last=True,
        auto_insert_metric_name=False,
    )

    callbacks = [
        checkpoint,
        LearningRateMonitor(logging_interval=str(get_nested(config, "logging.lr_logging_interval", "epoch"))),
    ]
    if bool(get_nested(config, "early_stopping.enabled", True)):
        callbacks.append(
            EarlyStopping(
                monitor=str(get_nested(config, "early_stopping.monitor", "val_dice")),
                patience=int(get_nested(config, "early_stopping.patience", 20)),
                mode=str(get_nested(config, "early_stopping.mode", "max")),
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

    trainer = L.Trainer(
        max_epochs=int(get_nested(config, "trainer.max_epochs", 100)),
        accelerator=str(get_nested(config, "trainer.accelerator", "auto")),
        devices=get_nested(config, "trainer.devices", "auto"),
        precision=str(get_nested(config, "trainer.precision", "16-mixed")),
        logger=[csv_logger, tensorboard_logger],
        callbacks=callbacks,
        default_root_dir=str(exp_dir),
        limit_train_batches=get_nested(config, "trainer.limit_train_batches", 1.0),
        limit_val_batches=get_nested(config, "trainer.limit_val_batches", 1.0),
        fast_dev_run=bool(get_nested(config, "trainer.fast_dev_run", False)),
        log_every_n_steps=int(get_nested(config, "logging.log_every_n_steps", 10)),
        check_val_every_n_epoch=int(get_nested(config, "trainer.check_val_every_n_epoch", 1)),
        accumulate_grad_batches=int(get_nested(config, "trainer.accumulate_grad_batches", 1)),
        gradient_clip_val=get_nested(config, "trainer.gradient_clip_val", None),
    )
    resume_checkpoint = resolve_path(get_nested(config, "trainer.resume_checkpoint", None))
    trainer.fit(module, datamodule=data_module, ckpt_path=resume_checkpoint)
    save_yaml_config(config, log_version_dir / "config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
