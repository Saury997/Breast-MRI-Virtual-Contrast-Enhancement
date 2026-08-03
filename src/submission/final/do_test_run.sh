#!/usr/bin/env bash
# Run loss-no-val locally against test/input/.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
IMAGE_NAME="${IMAGE_NAME:-mama-synth-loss-no-val}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_REF="$IMAGE_NAME:$IMAGE_TAG"

bash "$SCRIPT_DIR/do_build.sh"

rm -rf "$SCRIPT_DIR/test/output"
mkdir -p "$SCRIPT_DIR/test/output"

INPUT_DIR="$SCRIPT_DIR/test/input/images/pre-contrast-dce-mri-slice-breast"
if [ -z "$(ls -A "$INPUT_DIR"/*.mha 2>/dev/null)" ]; then
    echo ""
    echo "ERROR: No .mha file found in:"
    echo "  $INPUT_DIR"
    echo ""
    exit 1
fi

USE_GPU="${USE_GPU:-1}"
if [ "$USE_GPU" = "1" ]; then
    GPU_FLAG="--gpus device=0"
    DEVICE_ENV="-e MAMA_DEVICE=cuda:0"
    echo "Running with GPU (USE_GPU=1)"
else
    GPU_FLAG=""
    DEVICE_ENV="-e MAMA_DEVICE=cpu"
    echo "Running on CPU (USE_GPU=0)"
fi

if command -v cygpath >/dev/null 2>&1; then
    INPUT_MOUNT=$(cygpath -w "$SCRIPT_DIR/test/input")
    OUTPUT_MOUNT=$(cygpath -w "$SCRIPT_DIR/test/output")
else
    INPUT_MOUNT="$SCRIPT_DIR/test/input"
    OUTPUT_MOUNT="$SCRIPT_DIR/test/output"
fi

MSYS_NO_PATHCONV=1 docker run --rm \
    --network=none \
    --memory=8g \
    $GPU_FLAG \
    $DEVICE_ENV \
    -v "$INPUT_MOUNT:/input:ro" \
    -v "$OUTPUT_MOUNT:/output" \
    "$IMAGE_REF"

echo ""
echo "=== Output files ==="
find "$SCRIPT_DIR/test/output" -type f
