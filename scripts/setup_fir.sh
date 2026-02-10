#!/bin/bash
# Prepare a FIR/Compute Alliance environment for B.A.I.L.I.F.F.
# Usage:
#   bash scripts/setup_fir.sh
# Optional env vars:
#   INSTALL_LOCAL_BACKEND=1
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
#   FIR_MODULES="python/3.10 cuda/12.2"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if command -v module >/dev/null 2>&1; then
  MODULES="${FIR_MODULES:-python/3.10 cuda/12.2}"
  for m in $MODULES; do
    module load "$m" || true
  done
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,agent]"

if [ "${INSTALL_LOCAL_BACKEND:-1}" = "1" ]; then
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
  python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
  python -m pip install transformers accelerate bitsandbytes "huggingface_hub[cli]"
fi

python - <<'PY'
import bailiff
print("bailiff import OK:", bailiff.__file__)
PY

if [ "${INSTALL_LOCAL_BACKEND:-1}" = "1" ]; then
  if command -v hf >/dev/null 2>&1; then
    echo "Hugging Face CLI detected. Run: hf auth login"
  else
    echo "Hugging Face CLI not on PATH. Run: python -m huggingface_hub.commands.huggingface_cli login"
  fi
fi

echo "FIR setup complete."
