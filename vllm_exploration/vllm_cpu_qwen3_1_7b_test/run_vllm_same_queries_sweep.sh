#!/bin/bash
# Run vLLM with the same 512-query input set used in sglang_exploration.
# Batch size here means prompts per generate() call (chunk size).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
PY="${VLLM_CPU_PYTHON:-$ROOT/vllm_cpu_venv/bin/python3}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

PROMPTS_FILE_DEFAULT="/home/linhu/projects/sglang_exploration/sglang_test/data/queries_512.jsonl"
PROMPTS_FILE="${PROMPTS_FILE:-$PROMPTS_FILE_DEFAULT}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
OUT_DIR="${OUT_DIR:-$DIR/results_same_queries}"
BATCH_SIZES="${BATCH_SIZES:-8,16,32,64,128,256,512}"

mkdir -p "$OUT_DIR"

IFS=',' read -r -a BS_ARR <<< "$BATCH_SIZES"
for bs in "${BS_ARR[@]}"; do
  bs="$(echo "$bs" | xargs)"
  [[ -z "$bs" ]] && continue
  if [[ ! "$bs" =~ ^[0-9]+$ ]]; then
    echo "Invalid batch size: $bs" >&2
    exit 2
  fi
  out="$OUT_DIR/vllm_cpu_bs${bs}.json"
  echo "=== vLLM CPU same-queries run: batch_size=$bs ==="
  "$PY" "$DIR/test_qwen3_vllm_cpu.py" \
    --model "$MODEL" \
    --batch-size "$bs" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-num-seqs "$bs" \
    --prompts-file "$PROMPTS_FILE" \
    --num-prompts 512 \
    --output "$out"
done

echo "Saved results to: $OUT_DIR"
