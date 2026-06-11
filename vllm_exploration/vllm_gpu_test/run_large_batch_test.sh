#!/bin/bash
#
# Run Large Batch Performance Test (Batch Size 500)
#

echo "======================================================================"
echo "VLLM LARGE BATCH PERFORMANCE TEST"
echo "Batch Size: 500 (vs original 5)"
echo "======================================================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: venv not found"
    echo "Please create virtual environment first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install vllm transformers torch psutil"
    exit 1
fi

# Activate venv
source venv/bin/activate

pip install psutil -q

if ! python3 "$SCRIPT_DIR/verify_gpu_env.py"; then
  echo "Fix CUDA PyTorch first (see README / install_jetson_gpu_deps.sh)."
  exit 1
fi

echo ""
echo "System Information:"
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  vLLM: $(python3 -c 'import vllm; print(vllm.__version__)')"
echo "  CPU Cores: $(python3 -c 'import psutil; print(psutil.cpu_count())')"
echo "  GPU Available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo ""

echo "======================================================================"
echo "RUNNING TESTS"
echo "======================================================================"
echo ""
echo "This will test:"
echo "  1. Transformers on CPU (batch=500) with core monitoring"
echo "  2. vLLM on CPU (batch=500) with core monitoring"
echo "  3. Transformers on GPU (batch=500)"
echo "  4. vLLM on GPU (batch=500)"
echo ""
echo "Expected duration: 5-15 minutes"
echo ""

read -p "Press Enter to start tests..."

# Run tests
python3 test_large_batch.py --batch-size 500

echo ""
echo "======================================================================"
echo "TEST COMPLETE"
echo "======================================================================"
echo ""
echo "Results summary:"
echo "  - CPU core usage monitored and reported"
echo "  - Batch size: 500 (100× larger than original)"
echo "  - Full comparison completed"
echo ""
echo "Next steps:"
echo "  1. Review the output above"
echo "  2. Check LARGE_BATCH_RESULTS_500.md (if generated)"
echo "  3. Compare with original results (batch=5)"
echo ""
