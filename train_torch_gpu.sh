#!/usr/bin/env bash
# Lanza el entrenamiento de PyTorch usando el venv `venv-torch` (con torch+CUDA)
# junto al repo. PyTorch encuentra la GPU en WSL2 sin configuración extra.
# Si no hay venv-torch, usa `python`.
#
# Uso:  ./train_torch_gpu.sh --epochs 30
# (Sin GPU, basta con:  python train_torch.py --epochs 30)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV_TORCH="$HERE/../venv-torch"
PY="python"
[ -x "$VENV_TORCH/bin/python" ] && PY="$VENV_TORCH/bin/python"

exec "$PY" "$HERE/train_torch.py" "$@"
