#!/usr/bin/env bash
# Compare single-core vs multi-core performance from QEMU test results
# Usage: ./compare_qemu_performance.sh [single_file] [multi_file]
#   If files not provided, uses most recent files matching the pattern

set -euo pipefail

RESULTS_DIR="/home/linhu/repo/iminn-tools/results"

# If files provided as arguments, use them; otherwise find most recent
if [[ $# -ge 2 ]]; then
    SINGLE_FILE="$1"
    MULTI_FILE="$2"
else
    echo "Finding most recent test files..."
    SINGLE_FILE=$(ls -t "${RESULTS_DIR}"/test_q4_0_stories_system_mode_single_*.txt 2>/dev/null | head -1)
    MULTI_FILE=$(ls -t "${RESULTS_DIR}"/test_q4_0_stories_system_mode_multi_*.txt 2>/dev/null | head -1)
    
    if [[ -z "$SINGLE_FILE" ]]; then
        echo "ERROR: No single-core test file found in ${RESULTS_DIR}" >&2
        echo "Looking for: test_q4_0_stories_system_mode_single_*.txt" >&2
        exit 1
    fi
    
    if [[ -z "$MULTI_FILE" ]]; then
        echo "ERROR: No multi-core test file found in ${RESULTS_DIR}" >&2
        echo "Looking for: test_q4_0_stories_system_mode_multi_*.txt" >&2
        exit 1
    fi
fi

if [[ ! -f "$SINGLE_FILE" ]]; then
    echo "ERROR: Single-core file not found: $SINGLE_FILE" >&2
    exit 1
fi

if [[ ! -f "$MULTI_FILE" ]]; then
    echo "ERROR: Multi-core file not found: $MULTI_FILE" >&2
    exit 1
fi

echo "=== Single-core Test File ==="
echo "$SINGLE_FILE"
echo ""

echo "=== Multi-core Test File ==="
echo "$MULTI_FILE"
echo ""

# Extract thread count
echo "=== Thread Configuration ==="
SINGLE_THREADS=$(grep "n_threads = " "$SINGLE_FILE" | head -1 | grep -oP 'n_threads = \K\d+' || echo "unknown")
MULTI_THREADS=$(grep "n_threads = " "$MULTI_FILE" | head -1 | grep -oP 'n_threads = \K\d+' || echo "unknown")
echo "Single-core: $SINGLE_THREADS threads"
echo "Multi-core:  $MULTI_THREADS threads"
echo ""

# Extract eval time metrics
echo "=== Eval Time Performance ==="
SINGLE_EVAL=$(grep "eval time" "$SINGLE_FILE" | tail -1)
MULTI_EVAL=$(grep "eval time" "$MULTI_FILE" | tail -1)

if [[ -n "$SINGLE_EVAL" ]]; then
    echo "Single-core: $SINGLE_EVAL"
else
    echo "WARNING: Could not find eval time in single-core file" >&2
fi

if [[ -n "$MULTI_EVAL" ]]; then
    echo "Multi-core:  $MULTI_EVAL"
else
    echo "WARNING: Could not find eval time in multi-core file" >&2
fi
echo ""

# Extract tokens per second
echo "=== Throughput Comparison ==="
SINGLE_TPS=$(echo "$SINGLE_EVAL" | grep -oP '\d+\.\d+ tokens per second' | grep -oP '\d+\.\d+' || echo "")
MULTI_TPS=$(echo "$MULTI_EVAL" | grep -oP '\d+\.\d+ tokens per second' | grep -oP '\d+\.\d+' || echo "")

if [[ -n "$SINGLE_TPS" && -n "$MULTI_TPS" ]]; then
    SPEEDUP_TPS=$(echo "scale=2; $MULTI_TPS / $SINGLE_TPS" | bc)
    echo "Single-core: $SINGLE_TPS tokens/second"
    echo "Multi-core:  $MULTI_TPS tokens/second"
    echo "Speedup:     ${SPEEDUP_TPS}x"
else
    echo "Could not extract tokens/second metrics"
fi
echo ""

# Extract eval time per token
echo "=== Eval Time Per Token ==="
SINGLE_MS_PER_TOKEN=$(echo "$SINGLE_EVAL" | grep -oP '\(\s*\K\d+\.\d+ ms per token' | grep -oP '\d+\.\d+' || echo "")
MULTI_MS_PER_TOKEN=$(echo "$MULTI_EVAL" | grep -oP '\(\s*\K\d+\.\d+ ms per token' | grep -oP '\d+\.\d+' || echo "")

if [[ -n "$SINGLE_MS_PER_TOKEN" && -n "$MULTI_MS_PER_TOKEN" ]]; then
    SPEEDUP_MS=$(echo "scale=2; $SINGLE_MS_PER_TOKEN / $MULTI_MS_PER_TOKEN" | bc)
    echo "Single-core: $SINGLE_MS_PER_TOKEN ms per token"
    echo "Multi-core:  $MULTI_MS_PER_TOKEN ms per token"
    echo "Speedup:     ${SPEEDUP_MS}x (lower is better)"
else
    echo "Could not extract ms per token metrics"
fi
echo ""

# Extract total eval time
echo "=== Total Eval Time ==="
SINGLE_TOTAL_MS=$(echo "$SINGLE_EVAL" | grep -oP 'eval time = \s*\K\d+\.\d+' | head -1 || echo "")
MULTI_TOTAL_MS=$(echo "$MULTI_EVAL" | grep -oP 'eval time = \s*\K\d+\.\d+' | head -1 || echo "")

if [[ -n "$SINGLE_TOTAL_MS" && -n "$MULTI_TOTAL_MS" ]]; then
    SINGLE_TOTAL_SEC=$(echo "scale=2; $SINGLE_TOTAL_MS / 1000" | bc)
    MULTI_TOTAL_SEC=$(echo "scale=2; $MULTI_TOTAL_MS / 1000" | bc)
    SPEEDUP_TOTAL=$(echo "scale=2; $SINGLE_TOTAL_MS / $MULTI_TOTAL_MS" | bc)
    echo "Single-core: ${SINGLE_TOTAL_SEC} seconds (${SINGLE_TOTAL_MS} ms)"
    echo "Multi-core:  ${MULTI_TOTAL_SEC} seconds (${MULTI_TOTAL_MS} ms)"
    echo "Speedup:     ${SPEEDUP_TOTAL}x"
else
    echo "Could not extract total eval time"
fi
echo ""

# Summary
echo "=== Summary ==="
if [[ -n "$SINGLE_TPS" && -n "$MULTI_TPS" ]]; then
    if (( $(echo "$SPEEDUP_TPS > 1.3" | bc -l) )); then
        echo "✅ Good multi-core speedup detected (${SPEEDUP_TPS}x)"
    elif (( $(echo "$SPEEDUP_TPS > 1.0" | bc -l) )); then
        echo "⚠️  Modest multi-core speedup (${SPEEDUP_TPS}x) - may indicate limited parallelism"
    else
        echo "❌ No speedup or slowdown detected - check if multi-threading is working"
    fi
fi

