#!/usr/bin/env bash
# Export the loss-no-val container for Grand Challenge upload.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

VERSION="${VERSION:-20260710-loss-no-val-v2-r2}"
IMAGE_NAME="${IMAGE_NAME:-mama-synth-loss-no-val}"
IMAGE_TAG="${IMAGE_TAG:-$VERSION}"
IMAGE_REF="$IMAGE_NAME:$IMAGE_TAG"
OUT_FILE="$SCRIPT_DIR/mama-synth-loss-no-val-${VERSION}.tar.gz"

NO_REBUILD="${NO_REBUILD:-0}"
for arg in "$@"; do
    [ "$arg" = "--no-rebuild" ] && NO_REBUILD=1
done

if [ "$NO_REBUILD" != "1" ]; then
    IMAGE_NAME="$IMAGE_NAME" IMAGE_TAG="$IMAGE_TAG" SUBMISSION_VERSION="$VERSION" bash "$SCRIPT_DIR/do_build.sh"
else
    echo "Skipping rebuild (--no-rebuild / NO_REBUILD=1)."
    if ! docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
        echo "ERROR: Image $IMAGE_REF not found. Run do_build.sh first."
        exit 1
    fi
fi

if [ -f "$OUT_FILE" ]; then
    echo "ERROR: Refusing to overwrite existing file:"
    echo "  $OUT_FILE"
    echo "Set VERSION to a new value and rerun do_save.sh."
    exit 1
fi

docker save "$IMAGE_REF" | gzip -c > "$OUT_FILE"
echo ""
echo "Saved to $OUT_FILE"
echo "Upload this file to Grand Challenge -> Algorithm -> Container Management."
