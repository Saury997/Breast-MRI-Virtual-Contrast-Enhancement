#!/usr/bin/env python3
"""MAMA-SYNTH Grand Challenge inference.

Input:
    /input/images/pre-contrast-dce-mri-slice-breast/<uuid>.mha

Output:
    /output/images/synthetic-contrast-dce-mri-slice-breast/output.mha

The model is an SMP U-Net with an EfficientNet-B4 encoder. It predicts a
post-contrast residual, and the final output is pre_contrast + residual.
Training-only TinyUNet segmentation supervision, perceptual-loss weights, and
auxiliary segmentation-head weights in the checkpoint are ignored during
inference.
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


INPUT_PATH = Path(os.environ.get("MAMA_INPUT_DIR", "/input"))
OUTPUT_PATH = Path(os.environ.get("MAMA_OUTPUT_DIR", "/output"))
INPUT_SLUG = os.environ.get("MAMA_INPUT_SLUG", "pre-contrast-dce-mri-slice-breast")
OUTPUT_SLUG = os.environ.get(
    "MAMA_PREDICTION_SLUG", "synthetic-contrast-dce-mri-slice-breast"
)
MODEL_PATH = Path(os.environ.get("MAMA_MODEL_PATH", "/opt/app/model.ckpt"))
REQUESTED_DEVICE = os.environ.get("MAMA_DEVICE", "cuda:0")
PAD_MULTIPLE = int(os.environ.get("MAMA_PAD_MULTIPLE", "32"))


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(requested)


def _diagnostic_tree(root: Path) -> str:
    if not root.exists():
        return f"  {root} does not exist"
    lines: list[str] = []
    for idx, entry in enumerate(sorted(root.rglob("*"))):
        if idx >= 80:
            lines.append("  ...")
            break
        lines.append(f"  {entry}")
    return "\n".join(lines) if lines else f"  {root} is empty"


def find_input_image() -> Path:
    search_dir = INPUT_PATH / "images" / INPUT_SLUG
    candidates: list[str] = []
    for ext in ("*.mha", "*.nii.gz", "*.nii"):
        candidates.extend(glob(str(search_dir / ext)))

    if not candidates:
        raise FileNotFoundError(
            "\n".join(
                [
                    f"No input image found in: {search_dir}",
                    f"INPUT_SLUG: {INPUT_SLUG!r}",
                    "",
                    "Directory tree under /input/images:",
                    _diagnostic_tree(INPUT_PATH / "images"),
                ]
            )
        )

    candidates = sorted(candidates)
    if len(candidates) > 1:
        print(
            f"WARNING: {len(candidates)} input images found; using {candidates[0]}",
            file=sys.stderr,
        )
    return Path(candidates[0])


def read_input_array(path: Path) -> tuple[sitk.Image, np.ndarray, bool]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    had_singleton_z = False
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
        had_singleton_z = True
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D MHA slice, got shape {array.shape}")
    return image, array, had_singleton_z


def _torch_load(path: Path, device: torch.device) -> object:
    # Lightning checkpoints saved on Windows can pickle pathlib.WindowsPath
    # values in hyperparameters. Linux containers cannot instantiate them.
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[attr-defined, assignment]
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def create_network() -> torch.nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation-models-pytorch is required for loss-no-val inference."
        ) from exc

    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None,
        encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_use_norm="batchnorm",
        decoder_attention_type=None,
    )


def extract_model_state(state_dict: dict[str, object]) -> dict[str, torch.Tensor]:
    model_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("model.aux_segmentation_head."):
            continue
        if key.startswith("model.model."):
            model_state[key[len("model.model.") :]] = value
        elif key.startswith("module.model.model."):
            model_state[key[len("module.model.model.") :]] = value
        elif key.startswith("model.") and not key.startswith("model.aux_segmentation_head."):
            model_state[key[len("model.") :]] = value

    if not model_state:
        model_state = {
            key: value
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor)
        }
    return model_state


def load_network(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = _torch_load(checkpoint_path, device)
    state_dict = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")

    network = create_network()
    model_state = extract_model_state(state_dict)
    missing, unexpected = network.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match loss-no-val SMP U-Net generator.\n"
            f"Missing keys: {missing[:10]}\n"
            f"Unexpected keys: {unexpected[:10]}"
        )
    network.to(device)
    network.eval()
    return network


def pad_to_multiple(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    if multiple <= 0:
        raise ValueError(f"MAMA_PAD_MULTIPLE must be positive, got {multiple}")
    h, w = tensor.shape[-2:]
    pad_h = int(math.ceil(h / multiple) * multiple)
    pad_w = int(math.ceil(w / multiple) * multiple)
    padded = F.pad(tensor, (0, pad_w - w, 0, pad_h - h), mode="constant", value=0.0)
    return padded, h, w


def write_output(
    prediction: np.ndarray,
    input_image: sitk.Image,
    output_path: Path,
    had_singleton_z: bool,
) -> None:
    prediction = np.nan_to_num(
        prediction.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if had_singleton_z and input_image.GetDimension() == 3:
        out_image = sitk.GetImageFromArray(prediction[np.newaxis, :, :])
    else:
        out_image = sitk.GetImageFromArray(prediction)

    if out_image.GetDimension() == input_image.GetDimension():
        out_image.CopyInformation(input_image)
    else:
        print(
            "WARNING: input/output image dimensions differ; writing prediction "
            "without copying spatial metadata.",
            file=sys.stderr,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_image, str(output_path), useCompression=True)


def run() -> int:
    print("=" * 60)
    print("MAMA-SYNTH final test")
    print("=" * 60)

    device = _resolve_device(REQUESTED_DEVICE)
    print(f"Device: {device}")
    print(f"Model : {MODEL_PATH}")

    input_file = find_input_image()
    print(f"Input : {input_file}")
    input_image, array, had_singleton_z = read_input_array(input_file)
    print(
        f"  Shape: {array.shape}  dtype: {array.dtype}  "
        f"range: [{float(array.min()):.3f}, {float(array.max()):.3f}]"
    )

    network = load_network(MODEL_PATH, device)
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)
    tensor, h, w = pad_to_multiple(tensor, PAD_MULTIPLE)

    with torch.inference_mode():
        residual = network(tensor)
        prediction = (tensor + residual)[0, 0, :h, :w].detach().cpu().numpy()

    print(
        f"Output range: [{float(np.min(prediction)):.3f}, "
        f"{float(np.max(prediction)):.3f}]"
    )

    output_file = OUTPUT_PATH / "images" / OUTPUT_SLUG / "output.mha"
    write_output(prediction, input_image, output_file, had_singleton_z)
    print(f"Output: {output_file}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
