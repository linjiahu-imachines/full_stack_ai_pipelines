#!/bin/bash
# Run the same offline batch experiment on CPU, then on GPU if CUDA is available.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
DEF_PY="$ROOT/sglang_venv/bin/python"
if [[ -z "${SGLANG_PYTHON:-}" && -x "$DEF_PY" ]]; then
  PY="$DEF_PY"
else
  PY="${SGLANG_PYTHON:-python3}"
fi
NP="${BATCH_SIZE:-4}"
COMMON=(--synthetic --num-prompts "$NP" --batch-size "$NP" --max-new-tokens "${MAX_NEW_TOKENS:-32}")
if [[ -n "${MODEL:-}" ]]; then
  COMMON+=(--model "$MODEL")
fi

echo "=== SGLang offline batch — CPU ==="
"$PY" "$DIR/run_offline_batch_experiment.py" --device cpu "${COMMON[@]}" "$@"

echo ""
if "$PY" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "=== SGLang offline batch — CUDA ==="
  "$PY" "$DIR/run_offline_batch_experiment.py" --device cuda "${COMMON[@]}" "$@"
else
  echo "=== Skip GPU: torch not available or CUDA not available ==="
fi
