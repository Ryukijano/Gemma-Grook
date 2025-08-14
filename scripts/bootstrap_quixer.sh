#!/bin/bash
# Install Quixer and qujax into the active environment (expects conda env active)
# Usage: source scripts/bootstrap_quixer.sh

set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] python not found in PATH. Activate your conda env first." >&2
  return 1 2>/dev/null || exit 1
fi

echo "[INFO] Installing qujax and Quixer from GitHub..."
python -m pip install -q --upgrade pip setuptools wheel
python -m pip install -q "git+https://github.com/CQCL/qujax.git"
python -m pip install -q "git+https://github.com/CQCL/Quixer.git"

python - <<'PY'
import importlib
for m in ("qujax", "quixer"):
    assert importlib.util.find_spec(m) is not None, f"{m} not importable after install"
print("[INFO] qujax & quixer ready")
PY


