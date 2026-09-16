#!/usr/bin/env bash
# Fetch the HITNET ONNX models into models/hitnet/.
#
# Google's original frozen TensorFlow graphs
# (storage.googleapis.com/tensorflow-graphics/models/hitnet/default_models/*.pb)
# now return HTTP 403 to anonymous callers, so the TensorFlow reference
# implementation can no longer be fed a model. PINTO0309 converted the same
# trained weights to ONNX in PINTO_model_zoo/142_HITNET before that happened;
# this pulls that archive and keeps the sizes worth running on this hardware.
#
# Usage: ./download_hitnet_models.sh

set -euo pipefail

ARCHIVE_URL="https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/142_HITNET/resources.tar.gz"
DEST="models/hitnet"
FAMILIES=(eth3d middlebury_d400 flyingthings_finalpass_xl)
SIZES=(240x320 480x640 720x1280)

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

echo "Downloading HITNET model archive (~440 MB)..."
curl -fL --progress-bar "$ARCHIVE_URL" -o "$workdir/resources.tar.gz"

echo "Extracting ONNX models..."
tar -xzf "$workdir/resources.tar.gz" -C "$workdir" --wildcards '*.onnx'

for family in "${FAMILIES[@]}"; do
  for size in "${SIZES[@]}"; do
    mkdir -p "$DEST/$family/$size"
    cp "$workdir/$family/saved_model_$size/model_float32.onnx" \
       "$DEST/$family/$size/model_float32.onnx"
  done
done

echo
echo "Installed:"
find "$DEST" -name '*.onnx' | sort
echo
echo "Verify with: python check_hitnet_setup.py"
