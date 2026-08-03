#!/usr/bin/env python3
"""Smoke tests for the loss-no-val Grand Challenge container."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from glob import glob
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk


SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = (
    SCRIPT_DIR
    / "test"
    / "input"
    / "images"
    / "pre-contrast-dce-mri-slice-breast"
)
OUTPUT_DIR = (
    SCRIPT_DIR
    / "test"
    / "output"
    / "images"
    / "synthetic-contrast-dce-mri-slice-breast"
)
OUTPUT_FILE = OUTPUT_DIR / "output.mha"
IMAGE_NAME = os.environ.get("IMAGE_NAME", "mama-synth-loss-no-val")
IMAGE_TAG = os.environ.get("IMAGE_TAG", "pytest")
IMAGE_REF = f"{IMAGE_NAME}:{IMAGE_TAG}"
DEFAULT_CHECKPOINT = (SCRIPT_DIR / "resources" / "model.ckpt").resolve()
Z_SCORE_MAX_ABS = 100.0


def _run_docker() -> subprocess.CompletedProcess[str]:
    docker = shutil.which("docker")
    assert docker, "docker is required to run the submission container"

    checkpoint = Path(os.environ.get("BASELINE_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"
    target = SCRIPT_DIR / "resources" / "model.ckpt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.resolve() != target.resolve():
        shutil.copy2(checkpoint, target)

    if (SCRIPT_DIR / "test" / "output").exists():
        shutil.rmtree(SCRIPT_DIR / "test" / "output")
    (SCRIPT_DIR / "test" / "output").mkdir(parents=True, exist_ok=True)

    commands = [
        [
            docker,
            "build",
            "--provenance=false",
            "--sbom=false",
            "-t",
            IMAGE_REF,
            str(SCRIPT_DIR),
        ]
    ]
    run_cmd = [
        docker,
        "run",
        "--rm",
        "--network=none",
        "--memory=8g",
    ]
    if os.environ.get("USE_GPU", "0") == "1":
        run_cmd.extend(["--gpus", "device=0", "-e", "MAMA_DEVICE=cuda:0"])
    else:
        run_cmd.extend(["-e", "MAMA_DEVICE=cpu"])
    run_cmd.extend(
        [
            "-v",
            f"{SCRIPT_DIR / 'test' / 'input'}:/input:ro",
            "-v",
            f"{SCRIPT_DIR / 'test' / 'output'}:/output",
            IMAGE_REF,
        ]
    )
    commands.append(run_cmd)

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    for command in commands:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(SCRIPT_DIR),
            env=os.environ.copy(),
        )
        stdout_parts.append(result.stdout or "")
        stderr_parts.append(result.stderr or "")
        if result.returncode != 0:
            return subprocess.CompletedProcess(
                args=command,
                returncode=result.returncode,
                stdout="\n".join(stdout_parts),
                stderr="\n".join(stderr_parts),
            )
    return subprocess.CompletedProcess(
        args=commands[-1],
        returncode=0,
        stdout="\n".join(stdout_parts),
        stderr="\n".join(stderr_parts),
    )


@pytest.fixture(scope="session")
def container_run() -> subprocess.CompletedProcess[str]:
    assert glob(str(INPUT_DIR / "*.mha")), f"No .mha input found in {INPUT_DIR}"
    return _run_docker()


def test_container_runs_successfully(container_run: subprocess.CompletedProcess[str]) -> None:
    assert container_run.returncode == 0, (
        f"container exited with code {container_run.returncode}\n"
        f"STDOUT:\n{container_run.stdout}\n"
        f"STDERR:\n{container_run.stderr}"
    )


def test_output_file_created(container_run: subprocess.CompletedProcess[str]) -> None:
    assert container_run.returncode == 0
    assert OUTPUT_FILE.exists(), f"Expected output not found at {OUTPUT_FILE}"


def _read_output_array() -> np.ndarray:
    image = sitk.ReadImage(str(OUTPUT_FILE))
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return array


def test_output_is_readable_2d_image(container_run: subprocess.CompletedProcess[str]) -> None:
    assert container_run.returncode == 0
    assert _read_output_array().ndim == 2


def test_output_matches_input_shape(container_run: subprocess.CompletedProcess[str]) -> None:
    assert container_run.returncode == 0
    input_file = glob(str(INPUT_DIR / "*.mha"))[0]
    input_array = sitk.GetArrayFromImage(sitk.ReadImage(input_file))
    if input_array.ndim == 3 and input_array.shape[0] == 1:
        input_array = input_array[0]
    assert _read_output_array().shape == input_array.shape


def test_output_is_float32(container_run: subprocess.CompletedProcess[str]) -> None:
    assert container_run.returncode == 0
    assert _read_output_array().dtype == np.float32


def test_output_zscore_range_plausible(container_run: subprocess.CompletedProcess[str]) -> None:
    assert container_run.returncode == 0
    array = _read_output_array().astype(np.float64)
    assert np.isfinite(array).all()
    assert float(np.max(np.abs(array))) < Z_SCORE_MAX_ABS


def test_saved_archive_has_one_image() -> None:
    # Helper assertion for local package inspection when a tar.gz is present.
    archives = sorted(SCRIPT_DIR.glob("mama-synth-loss-no-val-*.tar.gz"))
    if not archives:
        pytest.skip("No exported archive found yet")
    import gzip
    import tarfile

    with gzip.open(archives[-1], "rb") as gz, tarfile.open(fileobj=gz, mode="r|*") as tf:
        for member in tf:
            if member.name == "manifest.json":
                data = json.loads(tf.extractfile(member).read().decode())
                assert len(data) == 1
                return
    raise AssertionError("manifest.json not found in exported Docker archive")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
