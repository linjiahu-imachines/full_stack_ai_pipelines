# vLLM CPU Testing Suite

This project compares LLM inference performance with and without vLLM on a CPU-only setup.

## Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Test Files

- **test_with_vllm.py**: Tests using vLLM for inference on CPU
- **test_without_vllm.py**: Tests using transformers directly (without vLLM) on CPU
- **test_comparison.py**: Comparison suite that runs both tests and shows performance differences

## Running Tests

### Run individual tests

```bash
# Test WITH vLLM
python test_with_vllm.py

# Test WITHOUT vLLM (using transformers)
python test_without_vllm.py

# Run comparison tests
python test_comparison.py
```

### Run with pytest

```bash
# Run all comparison tests
pytest test_comparison.py -v -s

# Run specific test
pytest test_comparison.py::TestComparison::test_single_inference_comparison -v -s
```

## What's Being Tested

1. **Single Inference**: Generating text for multiple prompts sequentially
2. **Batch Inference**: Processing multiple prompts in a batch

## Model Used

- **facebook/opt-125m**: A small 125M parameter model suitable for CPU testing
- Larger models can be used but will require more memory and time on CPU

## Expected Results

vLLM provides several optimizations even on CPU:
- PagedAttention for better memory management
- Continuous batching for improved throughput
- Optimized kernels and scheduling

However, note that:
- vLLM is primarily optimized for GPU inference
- CPU performance gains may be modest compared to GPU
- For CPU-only production, consider the tradeoff between setup complexity and performance gains

## CPU-Specific Settings

In the tests, we use:
- vLLM CPU build: Automatically uses CPU (no device parameter needed)
- `enforce_eager=True`: Disable CUDA graphs (not applicable for CPU)
- Small batch sizes: CPU has limited parallel processing compared to GPU

**Note**: This setup uses the vLLM CPU build (installed via `vllm-{version}+cpu` wheel), which is specifically compiled for CPU inference.

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use a smaller model
- Close other applications

### Slow Performance
- CPU inference is inherently slower than GPU
- Consider using smaller models
- Reduce max_tokens for generation

## Notes

- First run will download the model (~500MB for opt-125m)
- Subsequent runs will use cached model
- CPU inference is significantly slower than GPU inference
- vLLM's benefits are more pronounced with GPU acceleration
