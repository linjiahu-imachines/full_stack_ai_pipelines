#!/bin/bash
# CPU smoke: small batch, short generation (uses sglang_venv if present).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
DEF_PY="$ROOT/sglang_venv/bin/python"
if [[ -z "${SGLANG_PYTHON:-}" && -x "$DEF_PY" ]]; then
  PY="$DEF_PY"
else
  PY="${SGLANG_PYTHON:-python3}"
fi
exec "$PY" "$DIR/run_offline_batch_experiment.py" --device cpu --synthetic --num-prompts 2 --batch-size 2 --max-new-tokens 16 "$@"
