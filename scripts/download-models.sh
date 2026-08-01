#!/usr/bin/env bash
# Fetch the third-party model weights that this repository does not mirror.
#
# w600k_mbf.onnx is part of InsightFace's "buffalo_s" model pack. InsightFace
# provides its pretrained models for NON-COMMERCIAL RESEARCH purposes only, so
# the file is not redistributed here — you download it directly from upstream
# and accept their terms.  https://github.com/deepinsight/insightface
#
# Without this file the extension still runs: it automatically falls back to
# the bundled MIT-licensed `faceres` model (lower recognition accuracy).

set -euo pipefail

cd "$(dirname "$0")/.."
DEST="models/w600k_mbf.onnx"
URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"

if [ -f "$DEST" ]; then
  echo "✓ $DEST already present — nothing to do."
  exit 0
fi

echo "Downloading InsightFace buffalo_s pack…"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -fL "$URL" -o "$TMP/buffalo_s.zip"
unzip -o -j "$TMP/buffalo_s.zip" '*w600k_mbf.onnx' -d models/

if [ -f "$DEST" ]; then
  echo "✓ Installed $DEST"
else
  echo "✗ w600k_mbf.onnx was not found inside the archive." >&2
  exit 1
fi
