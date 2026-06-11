#!/bin/bash

# Verification script for vLLM exploration folder after reorganization
# This script checks that all components are working correctly

echo "════════════════════════════════════════════════════════════"
echo "  vLLM Exploration Folder - Post-Move Verification"
echo "════════════════════════════════════════════════════════════"
echo ""

VLLM_ROOT="/home/linhu/projects/vllm_exploration"
ERRORS=0

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if main folder exists
echo "1. Checking main folder structure..."
if [ -d "$VLLM_ROOT" ]; then
    echo -e "   ${GREEN}✓${NC} Main folder exists: $VLLM_ROOT"
else
    echo -e "   ${RED}✗${NC} Main folder not found: $VLLM_ROOT"
    ERRORS=$((ERRORS + 1))
fi

# Check for subfolder
echo ""
echo "2. Checking subfolders..."
if [ -d "$VLLM_ROOT/vllm_test" ]; then
    echo -e "   ${GREEN}✓${NC} CPU test folder exists"
else
    echo -e "   ${RED}✗${NC} CPU test folder not found"
    ERRORS=$((ERRORS + 1))
fi

if [ -d "$VLLM_ROOT/vllm_gpu_test" ]; then
    echo -e "   ${GREEN}✓${NC} GPU test folder exists"
else
    echo -e "   ${RED}✗${NC} GPU test folder not found"
    ERRORS=$((ERRORS + 1))
fi

# Check for reports
echo ""
echo "3. Checking reports..."
if [ -f "$VLLM_ROOT/docs/README.md" ]; then
    echo -e "   ${GREEN}✓${NC} Master README.md exists in docs/"
else
    echo -e "   ${RED}✗${NC} Master README.md not found in docs/"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "$VLLM_ROOT/docs/EXECUTIVE_REPORT.md" ]; then
    echo -e "   ${GREEN}✓${NC} Executive report exists in docs/"
else
    echo -e "   ${RED}✗${NC} Executive report not found in docs/"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "$VLLM_ROOT/README.md" ]; then
    echo -e "   ${GREEN}✓${NC} Root README.md exists"
else
    echo -e "   ${RED}✗${NC} Root README.md not found"
    ERRORS=$((ERRORS + 1))
fi

# Check CPU test environment
echo ""
echo "4. Checking CPU test environment..."
if [ -d "$VLLM_ROOT/vllm_test/venv" ]; then
    echo -e "   ${GREEN}✓${NC} CPU virtual environment exists"
    
    # Test Python import
    cd "$VLLM_ROOT/vllm_test"
    if ./venv/bin/python3 -c "import vllm, transformers, torch" 2>/dev/null; then
        echo -e "   ${GREEN}✓${NC} CPU dependencies installed correctly"
    else
        echo -e "   ${RED}✗${NC} CPU dependencies missing"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "   ${RED}✗${NC} CPU virtual environment not found"
    ERRORS=$((ERRORS + 1))
fi

# Check CPU test scripts
if [ -f "$VLLM_ROOT/vllm_test/run_tests.sh" ] && [ -x "$VLLM_ROOT/vllm_test/run_tests.sh" ]; then
    echo -e "   ${GREEN}✓${NC} CPU run script exists and is executable"
else
    echo -e "   ${RED}✗${NC} CPU run script missing or not executable"
    ERRORS=$((ERRORS + 1))
fi

# Check GPU test environment
echo ""
echo "5. Checking GPU test environment..."
if [ -d "$VLLM_ROOT/vllm_gpu_test/venv" ]; then
    echo -e "   ${GREEN}✓${NC} GPU virtual environment exists"
    
    # Test Python import
    cd "$VLLM_ROOT/vllm_gpu_test"
    if ./venv/bin/python3 -c "import vllm, transformers, torch" 2>/dev/null; then
        echo -e "   ${GREEN}✓${NC} GPU dependencies installed correctly"
        
        # Check CUDA
        CUDA_CHECK=$(./venv/bin/python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "$CUDA_CHECK" == "True" ]; then
            echo -e "   ${GREEN}✓${NC} CUDA available in GPU environment"
        else
            echo -e "   ${RED}✗${NC} CUDA not available in GPU environment"
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo -e "   ${RED}✗${NC} GPU dependencies missing"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "   ${RED}✗${NC} GPU virtual environment not found"
    ERRORS=$((ERRORS + 1))
fi

# Check GPU test scripts
if [ -f "$VLLM_ROOT/vllm_gpu_test/run_gpu_tests.sh" ] && [ -x "$VLLM_ROOT/vllm_gpu_test/run_gpu_tests.sh" ]; then
    echo -e "   ${GREEN}✓${NC} GPU run script exists and is executable"
else
    echo -e "   ${RED}✗${NC} GPU run script missing or not executable"
    ERRORS=$((ERRORS + 1))
fi

# Check GPU hardware
echo ""
echo "6. Checking GPU hardware..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "   ${GREEN}✓${NC} nvidia-smi available"
    echo -e "   ${GREEN}✓${NC} $GPU_COUNT GPU(s) detected"
else
    echo -e "   ${RED}✗${NC} nvidia-smi not available"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo ""
echo "════════════════════════════════════════════════════════════"
  if [ $ERRORS -eq 0 ]; then
    echo -e "  ${GREEN}✓ ALL CHECKS PASSED${NC}"
    echo "  The vLLM exploration folder is properly set up!"
    echo ""
    echo "  You can now:"
    echo "    - View reports: ls $VLLM_ROOT/docs/"
    echo "    - Run CPU tests: cd $VLLM_ROOT/vllm_test && ./run_tests.sh"
    echo "    - Run GPU tests: cd $VLLM_ROOT/vllm_gpu_test && ./run_gpu_tests.sh"
else
    echo -e "  ${RED}✗ VERIFICATION FAILED${NC}"
    echo "  Found $ERRORS error(s)"
    echo "  Please check the errors above and fix them."
fi
echo "════════════════════════════════════════════════════════════"
echo ""

exit $ERRORS
