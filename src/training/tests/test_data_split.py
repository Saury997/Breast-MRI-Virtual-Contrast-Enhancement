from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training.data import MamaSynthDataModule


def _touch_case(root: Path, case_id: str) -> None:
    for subdir in ("input", "ground_truth", "mask"):
        directory = root / subdir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{case_id}.mha").touch()


def test_all_train_split_uses_all_cases_without_validation(tmp_path: Path) -> None:
    for case_id in ("CASE_001", "CASE_002", "CASE_003"):
        _touch_case(tmp_path, case_id)

    data_module = MamaSynthDataModule(
        data_root=tmp_path,
        batch_size=1,
        num_workers=0,
        split_mode="all_train",
        val_fraction=0.0,
    )
    data_module.setup("fit")

    assert [record.case_id for record in data_module.train_records] == [
        "CASE_001",
        "CASE_002",
        "CASE_003",
    ]
    assert data_module.val_records == []


def test_unsupported_split_mode_mentions_all_train(tmp_path: Path) -> None:
    _touch_case(tmp_path, "CASE_001")
    data_module = MamaSynthDataModule(data_root=tmp_path, split_mode="bad")

    with pytest.raises(ValueError, match="all_train"):
        data_module.setup("fit")
