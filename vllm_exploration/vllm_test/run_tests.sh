#!/bin/bash

# Script to run vLLM CPU tests

echo "=================================="
echo "vLLM CPU Testing Suite"
echo "=================================="
echo ""

# Get script directory to use relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON="${SCRIPT_DIR}/venv/bin/python3"

# Verify Python exists
if [ ! -f "$PYTHON" ]; then
    echo "Error: Virtual environment not found at ${SCRIPT_DIR}/venv"
    echo "Please ensure the virtual environment is properly set up."
    exit 1
fi

# Check which test to run
if [ "$1" == "with" ]; then
    echo "Running tests WITH vLLM..."
    $PYTHON test_with_vllm.py
elif [ "$1" == "without" ]; then
    echo "Running tests WITHOUT vLLM (transformers only)..."
    $PYTHON test_without_vllm.py
elif [ "$1" == "compare" ]; then
    echo "Running comparison tests..."
    $PYTHON test_comparison.py
elif [ "$1" == "pytest" ]; then
    echo "Running pytest comparison suite..."
    ${SCRIPT_DIR}/venv/bin/pytest test_comparison.py -v -s
else
    echo "Usage: ./run_tests.sh [with|without|compare|pytest]"
    echo ""
    echo "Options:"
    echo "  with     - Run tests WITH vLLM"
    echo "  without  - Run tests WITHOUT vLLM (transformers only)"
    echo "  compare  - Run full comparison"
    echo "  pytest   - Run comparison using pytest"
    echo ""
    echo "Running full comparison by default..."
    $PYTHON test_comparison.py
fi
