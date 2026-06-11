#!/bin/bash
# GPU only: same defaults as run_offline_batch_experiment.py (512 queries, --batch-size per generate()).
# Usage:
#   ./run_experiment_gpu_batch.sh           # batch size 8
#   ./run_experiment_gpu_batch.sh 32      # batch size 32
#   BATCH_SIZE=16 ./run_experiment_gpu_batch.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
DEF_PY="$ROOT/sglang_venv/bin/python"
if [[ -z "${SGLANG_PYTHON:-}" && -x "$DEF_PY" ]]; then
  PY="$DEF_PY"
else
  PY="${SGLANG_PYTHON:-python3}"
fi

if ! "$PY" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "CUDA is not available; this script is GPU-only." >&2
  exit 2
fi

BS="${BATCH_SIZE:-8}"
if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
  BS="$1"
  shift
fi

ARGS=(--device cuda --batch-size "$BS")
[[ -n "${MODEL:-}" ]] && ARGS+=(--model "$MODEL")
[[ -n "${OUTPUT:-}" ]] && ARGS+=(--output "$OUTPUT")
[[ -n "${MAX_NEW_TOKENS:-}" ]] && ARGS+=(--max-new-tokens "$MAX_NEW_TOKENS")

exec "$PY" "$DIR/run_offline_batch_experiment.py" "${ARGS[@]}" "$@"
