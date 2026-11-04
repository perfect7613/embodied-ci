#!/usr/bin/env bash
# EmbodiedCI on your own GPU box. Run it on the box, or from your laptop:
#   ssh you@gpu-box 'BASETEN_API_KEY=... bash -s' < deploy/ssh/bootstrap.sh
# Needs: Ubuntu 22.04+, an NVIDIA GPU with >= 40 GB (V-JEPA 2-AC ViT-g + predictor in bf16), CUDA 12 driver.
# The API binds to 127.0.0.1:8700; reach it through a tunnel:  ssh -N -L 8700:localhost:8700 you@gpu-box
set -euo pipefail

REPO=${REPO:-https://github.com/perfect7613/embodied-ci}
DIR=${DIR:-$HOME/embodied-ci}
PORT=${PORT:-8700}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "no NVIDIA GPU visible" >&2; exit 1; }

if command -v apt-get >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq git build-essential clang linux-libc-dev libgl1 libegl1 libgles2 libglib2.0-0 ffmpeg
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

[ -d "$DIR" ] || git clone --depth 1 "$REPO" "$DIR"
[ -d /opt/vjepa2 ] || sudo git clone --depth 1 https://github.com/facebookresearch/vjepa2 /opt/vjepa2
cd "$DIR"
uv sync --extra engine

# Your Baseten key stays on this box, in a 600 file.
if [ -n "${BASETEN_API_KEY:-}" ]; then
  umask 077
  printf 'BASETEN_API_KEY=%s\nBASETEN_MODEL=%s\n' "$BASETEN_API_KEY" "${BASETEN_MODEL:-meta-models/muse-spark-open-1026}" > .env
fi

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl VJEPA_REPO=/opt/vjepa2 TORCH_HOME="$DIR/data/torch"
nohup uv run python -m infra.local_server --port "$PORT" --data "$DIR/data" > "$DIR/server.log" 2>&1 &
echo "EmbodiedCI engine starting on 127.0.0.1:$PORT (log: $DIR/server.log)."
echo "From your laptop: ssh -N -L $PORT:localhost:$PORT $(whoami)@$(hostname)"
