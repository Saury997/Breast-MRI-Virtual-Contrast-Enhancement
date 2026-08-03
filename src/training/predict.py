#!/usr/bin/env python3
"""Run a trained residual synthesis checkpoint on flat pre-contrast MHA files."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from training.config import get_nested, load_yaml_config
from training.data import read_mha_array
from training.models import create_model, is_smp_model_type
from training.modules import ResidualTranslationModule
from training.official_eval import write_prediction_like_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict post-contrast MHA slices.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model_type", type=str, default="resunet")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--encoder_weights", type=str, default=None)
    parser.add_argument("--encoder_depth", type=int, default=5)
    parser.add_argument("--decoder_channels", type=str, default=None)
    parser.add_argument("--decoder_use_norm", type=str, default=None)
    parser.add_argument("--decoder_attention_type", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pad_multiple", type=int, default=16)
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _pad_to_multiple(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    h, w = tensor.shape[-2:]
    pad_h = int(math.ceil(h / multiple) * multiple)
    pad_w = int(math.ceil(w / multiple) * multiple)
    padded = F.pad(tensor, (0, pad_w - w, 0, pad_h - h), mode="constant", value=0.0)
    return padded, h, w


def _parse_decoder_channels(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    channels = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not channels:
        raise ValueError("--decoder_channels must contain at least one channel value")
    return channels


def _load_model_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.config is not None:
        config = load_yaml_config(args.config)
        model_config = get_nested(config, "model", None)
        if not isinstance(model_config, dict):
            raise ValueError(f"model config must be a mapping in {args.config}")
        config_for_prediction = dict(model_config)
        model_type = str(config_for_prediction.get("type", "resunet")).lower()
        if is_smp_model_type(model_type):
            # Checkpoint loading immediately replaces model weights; skipping
            # pretrained initialization keeps prediction usable offline.
            config_for_prediction["encoder_weights"] = None
        return config_for_prediction

    model_config: dict[str, Any] = {
        "type": args.model_type,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
    }
    if args.model_type.lower() == "resunet":
        model_config["base_channels"] = args.base_channels
    else:
        model_config.update(
            {
                "encoder_name": args.encoder_name,
                "encoder_weights": args.encoder_weights,
                "encoder_depth": args.encoder_depth,
                "decoder_channels": _parse_decoder_channels(args.decoder_channels),
                "decoder_use_norm": args.decoder_use_norm,
                "decoder_attention_type": args.decoder_attention_type,
            }
        )
        if args.backbone is not None:
            model_config["backbone"] = args.backbone
    return model_config


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)
    model = create_model(**_load_model_config(args))
    module = ResidualTranslationModule.load_from_checkpoint(
        str(args.checkpoint),
        model=model,
        map_location=device,
    )
    module.eval().to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(args.input_dir.glob("*.mha"))
    if not input_files:
        raise FileNotFoundError(f"No .mha files found in {args.input_dir}")

    with torch.no_grad():
        for input_path in tqdm(input_files, desc="Predicting", unit="case"):
            array = read_mha_array(input_path)
            tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)
            tensor, h, w = _pad_to_multiple(tensor, args.pad_multiple)
            prediction = module(tensor)[0, 0, :h, :w].detach().cpu().numpy()
            write_prediction_like_input(
                prediction,
                input_path=input_path,
                output_path=args.output_dir / input_path.name,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
