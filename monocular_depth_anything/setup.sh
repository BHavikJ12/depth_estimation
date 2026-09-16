#!/usr/bin/env bash
# Creates ./venv and installs dependencies, picking the right ONNX Runtime for
# this machine.
#
#   ./setup.sh              auto: GPU build if an NVIDIA card is present
#   ./setup.sh --cpu        force the CPU build (smaller, ~3 fps)
#   ./setup.sh --gpu        force the GPU build, even if nvidia-smi is missing
#   ./setup.sh --metric     also torch + transformers, for the V2-Metric models
#
# Flags combine: ./setup.sh --cpu --metric
set -euo pipefail
cd "$(dirname "$0")"

WANT=auto
METRIC=0
for arg in "$@"; do
    case "$arg" in
        --cpu) WANT=cpu ;;
        --gpu) WANT=gpu ;;
        --metric) METRIC=1 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

# Some machines (this project's original one included) have an NVIDIA PyIndex
# extra-index-url in pip.conf pointing at pypi.ngc.nvidia.com, which does not
# resolve everywhere and makes every pip call retry for a minute before
# failing. Bypass pip's config only when that is actually what is configured --
# blanket-ignoring it would also discard a legitimate corporate mirror.
if pip config list 2>/dev/null | grep -q "pypi.ngc.nvidia.com"; then
    echo "==> pip.conf points at pypi.ngc.nvidia.com; bypassing it for this run"
    export PIP_CONFIG_FILE=/dev/null
fi

if [ "$WANT" = auto ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
        WANT=gpu
    else
        WANT=cpu
    fi
fi

# ONNX Runtime ships GPU wheels for Linux x86_64 and Windows x64 only.
ARCH="$(uname -m 2>/dev/null || echo unknown)"
OS="$(uname -s 2>/dev/null || echo unknown)"
if [ "$WANT" = gpu ] && { [ "$OS" = Darwin ] || { [ "$ARCH" != x86_64 ] && [ "$ARCH" != amd64 ]; }; }; then
    echo "==> no GPU wheels exist for $OS/$ARCH; using the CPU build instead"
    WANT=cpu
fi

echo "==> ONNX Runtime: $WANT build  ($OS/$ARCH)"

[ -d venv ] || python3 -m venv venv
./venv/bin/pip install --upgrade pip wheel
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -r "requirements-$WANT.txt"
[ "$METRIC" = 1 ] && ./venv/bin/pip install -r requirements-metric.txt

./venv/bin/python selftest.py

cat <<'MSG'

Done.  Try it:
    ./venv/bin/python run.py --source 0

Weights download on first use (the default model is ~100 MB).
MSG
