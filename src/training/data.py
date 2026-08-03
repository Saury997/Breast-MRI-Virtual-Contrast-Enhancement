"""Data loading for paired MAMA-SYNTH 2-D MHA slices."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning.pytorch as L
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from training.augmentations import AugmentationConfig, PairedSliceAugmentor


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    input_path: Path
    target_path: Path
    mask_path: Path


def discover_cases(data_root: str | Path) -> list[CaseRecord]:
    root = Path(data_root)
    input_dir = root / "input"
    target_dir = root / "ground_truth"
    mask_dir = root / "mask"
    for directory in (input_dir, target_dir, mask_dir):
        if not directory.exists():
            raise FileNotFoundError(f"Required data directory not found: {directory}")

    inputs = {p.stem: p for p in input_dir.glob("*.mha")}
    targets = {p.stem: p for p in target_dir.glob("*.mha")}
    masks = {p.stem: p for p in mask_dir.glob("*.mha")}
    common = sorted(set(inputs) & set(targets) & set(masks))
    missing = (set(inputs) ^ set(targets)) | (set(inputs) ^ set(masks))
    if missing:
        examples = ", ".join(sorted(missing)[:10])
        raise RuntimeError(
            "Input, ground-truth, and mask stems must match exactly. "
            f"Found {len(missing)} mismatched stem(s), e.g. {examples}"
        )
    if not common:
        raise RuntimeError(f"No paired .mha cases found under {root}")

    return [
        CaseRecord(
            case_id=case_id,
            input_path=inputs[case_id],
            target_path=targets[case_id],
            mask_path=masks[case_id],
        )
        for case_id in common
    ]


def split_records(
    records: list[CaseRecord],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[CaseRecord], list[CaseRecord]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_records = sorted(shuffled[:n_val], key=lambda r: r.case_id)
    train_records = sorted(shuffled[n_val:], key=lambda r: r.case_id)
    return train_records, val_records


def _require_case_id_list(data: dict[str, Any], key: str, path: Path) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Split file {path} must contain a string list at '{key}'")
    return value


def _find_duplicates(case_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case_id in case_ids:
        if case_id in seen:
            duplicates.add(case_id)
        seen.add(case_id)
    return sorted(duplicates)


def load_split_file(path: str | Path) -> tuple[list[str], list[str]]:
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if split_path.suffix.lower() != ".json":
        raise ValueError(f"Split file must be a .json file: {split_path}")

    try:
        with split_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON split file: {split_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Split file must contain a JSON object: {split_path}")
    train_ids = _require_case_id_list(data, "train", split_path)
    val_ids = _require_case_id_list(data, "val", split_path)
    if not train_ids:
        raise ValueError(f"Split file has no training cases: {split_path}")
    if not val_ids:
        raise ValueError(f"Split file has no validation cases: {split_path}")

    duplicate_train = _find_duplicates(train_ids)
    duplicate_val = _find_duplicates(val_ids)
    if duplicate_train:
        raise ValueError(
            f"Duplicate training case IDs in {split_path}: "
            f"{', '.join(duplicate_train[:10])}"
        )
    if duplicate_val:
        raise ValueError(
            f"Duplicate validation case IDs in {split_path}: "
            f"{', '.join(duplicate_val[:10])}"
        )

    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(
            f"Split train/val overlap in {split_path}: {', '.join(overlap[:10])}"
        )
    return train_ids, val_ids


def split_records_fixed(
    records: list[CaseRecord],
    split_file: str | Path,
) -> tuple[list[CaseRecord], list[CaseRecord]]:
    train_ids, val_ids = load_split_file(split_file)
    by_case_id = {record.case_id: record for record in records}
    discovered = set(by_case_id)
    requested = set(train_ids) | set(val_ids)

    unknown = sorted(requested - discovered)
    if unknown:
        raise ValueError(
            f"Split file references unknown case IDs: {', '.join(unknown[:10])}"
        )

    missing = sorted(discovered - requested)
    if missing:
        raise ValueError(
            "Split file does not cover all discovered cases. "
            f"Missing examples: {', '.join(missing[:10])}"
        )

    train_records = sorted(
        (by_case_id[case_id] for case_id in train_ids),
        key=lambda r: r.case_id,
    )
    val_records = sorted(
        (by_case_id[case_id] for case_id in val_ids),
        key=lambda r: r.case_id,
    )
    return train_records, val_records


def read_mha_array(path: str | Path, *, is_mask: bool = False) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D MHA slice at {path}, got shape {array.shape}")
    if is_mask:
        return (array > 0).astype(np.float32)
    return array.astype(np.float32)


class MamaSynthPairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[CaseRecord],
        augment: bool = False,
        augmentation_config: AugmentationConfig | None = None,
    ) -> None:
        self.records = records
        self.augmentor = PairedSliceAugmentor(augmentation_config) if augment else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = read_mha_array(record.input_path)
        target = read_mha_array(record.target_path)
        mask = read_mha_array(record.mask_path, is_mask=True)
        if image.shape != target.shape or image.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch for {record.case_id}: "
                f"input={image.shape}, target={target.shape}, mask={mask.shape}"
            )

        image_t = torch.from_numpy(image).unsqueeze(0)
        target_t = torch.from_numpy(target).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        if self.augmentor is not None:
            image_t, target_t, mask_t = self.augmentor(image_t, target_t, mask_t)

        h, w = image_t.shape[-2:]
        return {
            "input": image_t,
            "target": target_t,
            "mask": mask_t,
            "valid_mask": torch.ones_like(mask_t),
            "case_id": record.case_id,
            "input_path": str(record.input_path),
            "target_path": str(record.target_path),
            "mask_path": str(record.mask_path),
            "original_size": torch.tensor([h, w], dtype=torch.long),
        }


def pad_collate(
    batch: list[dict[str, Any]],
    pad_multiple: int = 16,
) -> dict[str, Any]:
    max_h = max(int(item["input"].shape[-2]) for item in batch)
    max_w = max(int(item["input"].shape[-1]) for item in batch)
    pad_h = int(math.ceil(max_h / pad_multiple) * pad_multiple)
    pad_w = int(math.ceil(max_w / pad_multiple) * pad_multiple)

    def pad_tensor(tensor: torch.Tensor) -> torch.Tensor:
        h, w = tensor.shape[-2:]
        return F.pad(tensor, (0, pad_w - w, 0, pad_h - h), mode="constant", value=0.0)

    return {
        "input": torch.stack([pad_tensor(item["input"]) for item in batch]),
        "target": torch.stack([pad_tensor(item["target"]) for item in batch]),
        "mask": torch.stack([pad_tensor(item["mask"]) for item in batch]),
        "valid_mask": torch.stack([pad_tensor(item["valid_mask"]) for item in batch]),
        "case_id": [item["case_id"] for item in batch],
        "input_path": [item["input_path"] for item in batch],
        "target_path": [item["target_path"] for item in batch],
        "mask_path": [item["mask_path"] for item in batch],
        "original_size": torch.stack([item["original_size"] for item in batch]),
    }


@dataclass(frozen=True)
class PadCollate:
    """Pickle-safe collate callable for Windows DataLoader workers."""

    pad_multiple: int = 16

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return pad_collate(batch, self.pad_multiple)


class MamaSynthDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str | Path,
        batch_size: int = 4,
        num_workers: int = 4,
        val_fraction: float = 0.1,
        seed: int = 42,
        augment: bool = True,
        augmentation_config: AugmentationConfig | None = None,
        pad_multiple: int = 16,
        split_mode: str = "random",
        split_file: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.augment = augment
        self.augmentation_config = augmentation_config
        self.pad_multiple = pad_multiple
        self.split_mode = split_mode
        self.split_file = Path(split_file) if split_file is not None else None
        self.train_records: list[CaseRecord] = []
        self.val_records: list[CaseRecord] = []

    def setup(self, stage: str | None = None) -> None:
        records = discover_cases(self.data_root)
        split_mode = self.split_mode.lower()
        if split_mode == "random":
            self.train_records, self.val_records = split_records(
                records,
                val_fraction=self.val_fraction,
                seed=self.seed,
            )
        elif split_mode == "fixed":
            if self.split_file is None:
                raise ValueError("data.split_file is required when split_mode is fixed")
            self.train_records, self.val_records = split_records_fixed(
                records,
                self.split_file,
            )
        elif split_mode == "all_train":
            self.train_records = sorted(records, key=lambda r: r.case_id)
            self.val_records = []
        else:
            raise ValueError(
                f"Unsupported split_mode={self.split_mode!r}. "
                "Expected 'random', 'fixed', or 'all_train'."
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            MamaSynthPairDataset(
                self.train_records,
                augment=self.augment,
                augmentation_config=self.augmentation_config,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=PadCollate(self.pad_multiple),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            MamaSynthPairDataset(self.val_records, augment=False),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=PadCollate(self.pad_multiple),
        )
