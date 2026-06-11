"""
Verification script to check if vLLM and dependencies are properly installed
"""
import sys

def check_installation():
    """Check if all required packages are installed"""
    print("=" * 80)
    print("vLLM CPU Setup Verification")
    print("=" * 80)
    print()
    
    errors = []
    
    # Check Python version
    print(f"✓ Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        errors.append("Python 3.8 or higher is required")
    
    # Check vLLM
    try:
        import vllm
        print(f"✓ vLLM version: {vllm.__version__}")
    except ImportError as e:
        print(f"✗ vLLM not found: {e}")
        errors.append("vLLM is not installed")
    
    # Check transformers
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers not found: {e}")
        errors.append("Transformers is not installed")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        print(f"  - CPU available: True")
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        errors.append("PyTorch is not installed")
    
    # Check pytest
    try:
        import pytest
        print(f"✓ Pytest version: {pytest.__version__}")
    except ImportError as e:
        print(f"✗ Pytest not found: {e}")
        errors.append("Pytest is not installed")
    
    print()
    print("=" * 80)
    
    if errors:
        print("ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        print()
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("✓ All dependencies are properly installed!")
        print()
        print("You can now run the tests:")
        print("  ./run_tests.sh compare    # Run full comparison")
        print("  python test_with_vllm.py  # Test with vLLM only")
        print("  python test_without_vllm.py  # Test without vLLM")
        return True


if __name__ == "__main__":
    success = check_installation()
    sys.exit(0 if success else 1)
