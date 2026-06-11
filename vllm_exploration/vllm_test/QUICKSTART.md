# Quick Start Guide

## Setup Complete! ✓

Your vLLM CPU testing environment is ready to use.

## What's Been Set Up

1. ✓ Virtual environment created in `venv/`
2. ✓ vLLM v0.15.1 installed
3. ✓ PyTorch, Transformers, and test dependencies installed
4. ✓ Test scripts created

## Project Structure

```
vllm_test/
├── venv/                    # Virtual environment
├── requirements.txt         # Python dependencies
├── verify_setup.py          # Verify installation
├── test_with_vllm.py        # Tests WITH vLLM on CPU
├── test_without_vllm.py     # Tests WITHOUT vLLM (transformers)
├── test_comparison.py       # Comparison suite
├── run_tests.sh             # Convenience script to run tests
├── README.md                # Detailed documentation
└── QUICKSTART.md            # This file
```

## Running Tests

### Option 1: Use the convenience script (recommended)

```bash
cd /home/linhu/projects/vllm_test

# Activate virtual environment
source venv/bin/activate

# Run full comparison (default)
./run_tests.sh compare

# Or run individual tests:
./run_tests.sh with       # Test WITH vLLM
./run_tests.sh without    # Test WITHOUT vLLM
./run_tests.sh pytest     # Run with pytest
```

### Option 2: Run Python scripts directly

```bash
cd /home/linhu/projects/vllm_test
source venv/bin/activate

# Test WITH vLLM
python test_with_vllm.py

# Test WITHOUT vLLM
python test_without_vllm.py

# Run comparison
python test_comparison.py

# Run with pytest
pytest test_comparison.py -v -s
```

## What the Tests Do

### test_with_vllm.py
- Initializes vLLM with CPU-only mode
- Runs inference on sample prompts
- Tests single and batch inference
- Uses facebook/opt-125m model (small, suitable for CPU)

### test_without_vllm.py
- Uses HuggingFace transformers directly
- Same prompts and parameters for fair comparison
- No vLLM optimizations

### test_comparison.py
- Runs both tests back-to-back
- Compares performance metrics
- Shows speedup/slowdown percentages

## Important Notes

### CPU-Only Mode
Even though your system has CUDA available, the tests explicitly use `device="cpu"` to test CPU-only performance. This is intentional for the comparison.

### First Run
The first run will:
- Download the model (~500MB for opt-125m)
- Cache it locally for future use
- Take longer than subsequent runs

### Performance Expectations
- CPU inference is much slower than GPU
- vLLM optimizations provide some benefits even on CPU
- For production CPU workloads, vLLM can improve throughput

## Example Output

When you run the comparison, you'll see:
- Execution time for each approach
- Generated text samples
- Performance comparison (speedup factor)
- Summary statistics

## Troubleshooting

### If you get an import error
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### If you run out of memory
- The tests use a small model (125M parameters)
- Close other applications to free up RAM
- Consider using an even smaller model

### To verify everything is working
```bash
source venv/bin/activate
python verify_setup.py
```

## Next Steps

1. Run the verification script to confirm setup:
   ```bash
   source venv/bin/activate
   python verify_setup.py
   ```

2. Start with a quick test:
   ```bash
   ./run_tests.sh with
   ```

3. Then run the full comparison:
   ```bash
   ./run_tests.sh compare
   ```

## Customization

You can modify the tests to:
- Use different models (change `facebook/opt-125m` to another model)
- Test with different prompts
- Adjust generation parameters (temperature, max_tokens, etc.)
- Change batch sizes

## Getting Help

- See README.md for detailed documentation
- Check test files for implementation details
- vLLM docs: https://docs.vllm.ai/

Enjoy testing! 🚀
