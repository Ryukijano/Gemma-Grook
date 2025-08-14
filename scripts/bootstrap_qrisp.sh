#!/bin/bash
# Ensure Qrisp and required python deps are available in the active env
# Usage: source scripts/bootstrap_qrisp.sh

set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] python not found in PATH. Activate your conda env first." >&2
  return 1 2>/dev/null || exit 1
fi

need_install_qrisp=0
python - <<'PY' 2>/dev/null || need_install_qrisp=1
import importlib
import sys
assert importlib.util.find_spec('qrisp') is not None
assert importlib.util.find_spec('qrisp.vqe') is not None
print('qrisp-present')
PY

if [[ "$need_install_qrisp" == "1" ]]; then
  echo "[INFO] Installing Qrisp (pip) into current env..."
  python -m pip install -q --upgrade pip
  python -m pip install -q qrisp
fi

# Ensure auxiliary deps used in the notebook exist
for mod in networkx matplotlib scipy; do
  python - <<PY 2>/dev/null || python -m pip install -q ${mod}
import importlib; assert importlib.util.find_spec('${mod}') is not None
print('${mod}-present')
PY
done

echo "[INFO] Qrisp and deps ready."


