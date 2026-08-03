#!/usr/bin/env bash
# Build the loss-no-val Grand Challenge container.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
DEFAULT_CHECKPOINT="$SCRIPT_DIR/resources/model.ckpt"
CHECKPOINT_PATH="${BASELINE_CHECKPOINT:-${MODEL_CHECKPOINT:-$DEFAULT_CHECKPOINT}}"
RESOURCES_DIR="$SCRIPT_DIR/resources"
TARGET_CHECKPOINT="$RESOURCES_DIR/model.ckpt"
IMAGE_NAME="${IMAGE_NAME:-mama-synth-loss-no-val}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_REF="$IMAGE_NAME:$IMAGE_TAG"
SUBMISSION_VERSION="${SUBMISSION_VERSION:-$IMAGE_TAG}"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo ""
    echo "ERROR: checkpoint not found at:"
    echo "  $CHECKPOINT_PATH"
    echo ""
    echo "Set BASELINE_CHECKPOINT to a valid Lightning checkpoint:"
    echo "  BASELINE_CHECKPOINT=/path/to/model.ckpt bash do_build.sh"
    echo ""
    exit 1
fi

mkdir -p "$RESOURCES_DIR"
if [ -f "$TARGET_CHECKPOINT" ] && cmp -s "$CHECKPOINT_PATH" "$TARGET_CHECKPOINT"; then
    echo "Checkpoint already staged at $TARGET_CHECKPOINT"
else
    cp "$CHECKPOINT_PATH" "$TARGET_CHECKPOINT.tmp"
    mv "$TARGET_CHECKPOINT.tmp" "$TARGET_CHECKPOINT"
fi

CHECKPOINT_SHA=$(python - "$TARGET_CHECKPOINT" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
h = hashlib.sha256()
with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest().upper())
PY
)
CREATED=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
REVISION="loss-no-val-${IMAGE_TAG}-${CHECKPOINT_SHA:0:12}"

echo "Staged checkpoint:"
ls -lh "$TARGET_CHECKPOINT"
echo "Checkpoint SHA256: $CHECKPOINT_SHA"
echo ""
echo "Building Docker image: $IMAGE_REF"
docker build ${DOCKER_NO_CACHE:+--no-cache} \
    --provenance=false \
    --sbom=false \
    --build-arg "SUBMISSION_VERSION=$SUBMISSION_VERSION" \
    --label "org.opencontainers.image.title=loss-no-val" \
    --label "org.opencontainers.image.version=$IMAGE_TAG" \
    --label "org.opencontainers.image.created=$CREATED" \
    --label "org.opencontainers.image.revision=$REVISION" \
    --label "mama-synth.model.name=loss-no-val" \
    --label "mama-synth.checkpoint.sha256=$CHECKPOINT_SHA" \
    --label "mama-synth.checkpoint.name=loss-no-val-version-2-last.ckpt" \
    -t "$IMAGE_REF" "$SCRIPT_DIR"
echo "Build complete."
