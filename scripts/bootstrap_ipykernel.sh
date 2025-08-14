#!/bin/bash
# Ensure ipykernel is installed in the active conda env and a kernel spec exists
# Usage: source scripts/bootstrap_ipykernel.sh [KERNEL_NAME]

set -euo pipefail

KERNEL_NAME=${1:-qrisp-jax}

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] python not found in PATH. Activate your conda env first." >&2
  return 1 2>/dev/null || exit 1
fi

# Try importing ipykernel; if missing, install via conda first, then fallback to pip
python - <<'PY' 2>/dev/null || MISSING=1
import ipykernel
print('ipykernel-present')
PY

if [[ "${MISSING:-0}" == "1" ]]; then
  echo "[INFO] Installing ipykernel into current env..."
  if command -v conda >/dev/null 2>&1; then
    conda install -y -c conda-forge ipykernel || true
  fi
  python -m pip install -q --upgrade ipykernel
fi

echo "[INFO] Registering Jupyter kernel spec: ${KERNEL_NAME}"
python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "Python (${KERNEL_NAME})" 1>/dev/null
echo "[INFO] Kernel spec installed for ${KERNEL_NAME}"


