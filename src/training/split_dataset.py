#!/usr/bin/env python3
"""Create a fixed train/validation split JSON for MAMA-SYNTH training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed train/validation split from MHA case files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing pre-contrast input .mha files.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Output JSON split file path.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Validation fraction. Default: 0.2 for val:train = 2:8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before freezing the split.",
    )
    return parser.parse_args()


def discover_case_ids(input_dir: Path) -> list[str]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    case_ids = sorted(p.stem for p in input_dir.glob("*.mha"))
    if not case_ids:
        raise RuntimeError(f"No .mha input files found under {input_dir}")
    return case_ids


def make_split(
    case_ids: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    if n_val >= len(shuffled):
        raise ValueError(
            "val_fraction leaves no training cases; choose a smaller value."
        )

    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    return train_ids, val_ids


def write_split(
    output_path: Path,
    input_dir: Path,
    seed: int,
    val_fraction: float,
    train_ids: list[str],
    val_ids: list[str],
) -> None:
    split = {
        "version": 1,
        "source_input_dir": str(input_dir.resolve()),
        "seed": seed,
        "val_fraction": val_fraction,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "train": train_ids,
        "val": val_ids,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    case_ids = discover_case_ids(args.input_dir)
    train_ids, val_ids = make_split(case_ids, args.val_fraction, args.seed)
    write_split(
        output_path=args.output_path,
        input_dir=args.input_dir,
        seed=args.seed,
        val_fraction=args.val_fraction,
        train_ids=train_ids,
        val_ids=val_ids,
    )
    print(
        f"Wrote split to {args.output_path} "
        f"({len(train_ids)} train, {len(val_ids)} val)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
