#!/usr/bin/env bash
# Creates ./venv and installs dependencies.
# Everything needed for all backends; local inference is a docker server, not a
# pip package, so there is nothing heavy to install here. See ./serve_local.sh.
set -euo pipefail
cd "$(dirname "$0")"

# This machine has an NVIDIA PyIndex extra-index-url in ~/.config/pip/pip.conf
# pointing at pypi.ngc.nvidia.com, which does not resolve here and makes every
# pip call retry for a minute before failing. Bypass the config file.
export PIP_CONFIG_FILE=/dev/null

[ -d venv ] || python3 -m venv venv
./venv/bin/pip install --upgrade pip wheel
./venv/bin/pip install -r requirements.txt
./venv/bin/python selftest.py
./venv/bin/python pipeline_test.py
echo
echo "Done. Set your key, then run:"
echo
echo "  export ROBOFLOW_API_KEY=xxxxxxxx      # https://app.roboflow.com/settings/api"
echo "  ./venv/bin/python run.py --source 0 \\"
echo "      --model-id drone-detection-rchy7/8 --keep-classes 1 \\"
echo "      --conf 0.40 --hfov 70 --target-size 0.5"
echo
echo "--source takes a camera index, video file, stream URL, image, or folder."
echo "See README.md to run it, SETUP.md to install it elsewhere."
