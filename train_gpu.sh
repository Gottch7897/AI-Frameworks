#!/usr/bin/env bash
# Lanza el entrenamiento en GPU. Conveniencia para máquinas con la GPU configurada
# vía un venv `venv-gpu` (con `tensorflow[and-cuda]`) junto al repo. Fija el
# LD_LIBRARY_PATH que TensorFlow necesita en WSL2. Si no hay venv-gpu, usa `python`.
#
# Uso:  ./train_gpu.sh --epochs 30
# (Sin GPU, basta con:  python train.py --epochs 30)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV_GPU="$HERE/../venv-gpu"
PY="python"

if [ -x "$VENV_GPU/bin/python" ]; then
    PY="$VENV_GPU/bin/python"
    NV="$VENV_GPU/lib/python3.12/site-packages/nvidia"
    LDP=""
    for d in "$NV"/*/lib; do LDP="$LDP:$d"; done
    export LD_LIBRARY_PATH="${LDP#:}:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$PY" "$HERE/train.py" "$@"
