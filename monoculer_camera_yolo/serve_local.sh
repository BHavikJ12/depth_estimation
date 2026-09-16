#!/usr/bin/env bash
# Runs Roboflow's inference server on this machine so run.py can hit
# http://localhost:9001 instead of the cloud: no per-frame round trip, and
# offline once the model is cached.
#
#   ./serve_local.sh          GPU image (needs the NVIDIA container toolkit)
#   ./serve_local.sh --cpu    CPU image
#
# Then:  ./venv/bin/python run.py --source 0 --endpoint http://localhost:9001 --hfov 70
set -euo pipefail

: "${ROBOFLOW_API_KEY:?export ROBOFLOW_API_KEY first (https://app.roboflow.com/settings/api)}"

NAME=roboflow-inference
if [ "${1:-}" = "--cpu" ]; then
    IMAGE=roboflow/roboflow-inference-server-cpu:latest
    GPU_ARGS=()
else
    IMAGE=roboflow/roboflow-inference-server-gpu:latest
    GPU_ARGS=(--gpus all)
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" "${GPU_ARGS[@]}" \
    -p 9001:9001 \
    -e ROBOFLOW_API_KEY="$ROBOFLOW_API_KEY" \
    -v "$HOME/.inference/cache:/tmp/cache" \
    "$IMAGE"

echo -n "waiting for the server"
for _ in $(seq 1 60); do
    if curl -fsS http://localhost:9001/info >/dev/null 2>&1; then
        echo " up"
        curl -s http://localhost:9001/info
        echo
        exit 0
    fi
    echo -n "."
    sleep 2
done
echo
echo "server did not come up; check: docker logs $NAME" >&2
exit 1
