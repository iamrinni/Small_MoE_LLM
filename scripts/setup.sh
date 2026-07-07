#!/usr/bin/env bash
# Clean-environment bootstrap: install deps, run tests, and (once present) a smoke
# train + eval. Designed so a fresh clone reproduces the project end to end.
#
#   bash scripts/setup.sh            # install + test (+ smoke if scripts exist)
#   SKIP_INSTALL=1 bash scripts/setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> [1/4] Python version check (need >= 3.10)"
python - <<'PY'
import sys
assert sys.version_info >= (3, 10), f"Python >= 3.10 required, found {sys.version}"
print("OK:", sys.version.split()[0])
PY

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  echo "==> [2/4] Installing dependencies (requirements.txt)"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
else
  echo "==> [2/4] SKIP_INSTALL=1 set — skipping pip install"
fi

echo "==> [3/4] Running unit tests"
python -m pytest -q

echo "==> [4/4] Smoke run (tiny config)"
if [[ -f scripts/train.py && -f scripts/evaluate.py ]]; then
  python scripts/train.py --config configs/smoke.yaml
  python scripts/evaluate.py --config configs/smoke.yaml
  echo "Smoke train + eval complete."
else
  echo "scripts/train.py / evaluate.py not present yet (added in Phases 4-5). Skipping smoke run."
fi

echo "==> Setup complete."
