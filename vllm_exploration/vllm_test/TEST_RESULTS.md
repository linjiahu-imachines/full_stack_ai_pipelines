# vLLM CPU Testing Results

## Test Date: February 11, 2026

## Environment
- **System**: Linux (60 CPU cores)
- **vLLM Version**: 0.15.1+cpu (CPU build)
- **PyTorch Version**: 2.10.0+cpu
- **Model**: facebook/opt-125m (125M parameters)
- **Device**: CPU only

## Test Results

### Single Inference Test (3 prompts sequentially)

| Method | Total Time | Avg Time per Prompt | Speed |
|--------|-----------|---------------------|-------|
| **WITH vLLM** | 21.79s | 7.26s | Input: 0.87 tok/s, Output: 5.33 tok/s |
| **WITHOUT vLLM** (transformers) | 2.67s | 0.89s | ~8x faster |

### Batch Inference Test (5 prompts in batch)

| Method | Total Time | Avg Time per Prompt | Speed |
|--------|-----------|---------------------|-------|
| **WITH vLLM** | 9.70s | 1.94s | Input: 3.09 tok/s, Output: 15.47 tok/s |
| **WITHOUT vLLM** (transformers) | 0.87s | 0.17s | ~11x faster |

## Analysis

### Key Findings

1. **Direct transformers approach is significantly faster on CPU**
   - Single inference: 8.16x faster without vLLM
   - Batch inference: 11.15x faster without vLLM

2. **Why transformers is faster on CPU for this use case:**
   - **Overhead**: vLLM adds significant overhead for features like PagedAttention, continuous batching, and scheduling
   - **Optimization target**: vLLM is primarily optimized for GPU inference where its benefits shine
   - **Small model**: For a 125M parameter model, the direct PyTorch approach is more efficient on CPU
   - **Simple workload**: Single-user, sequential inference doesn't benefit from vLLM's concurrent request handling

3. **vLLM batch processing did improve**
   - vLLM batch test was 2.2x faster than its single test
   - Shows that vLLM's batching features do work, but still slower than direct transformers

### When to Use vLLM on CPU

Based on these results, vLLM on CPU might be beneficial when:
- **High concurrency**: Many concurrent users/requests where vLLM's continuous batching helps
- **Large models**: Larger models where PagedAttention's memory efficiency matters
- **Production serving**: When you need vLLM's OpenAI-compatible API server
- **Memory constraints**: When you need better KV cache management for long sequences

### When to Use Direct Transformers on CPU

Use direct transformers when:
- **Simple inference**: Single-user or low-concurrency scenarios
- **Small models**: Models under 1B parameters
- **Minimal latency**: When you need the fastest possible inference time
- **Prototyping**: Quick experiments and development

## Sample Outputs

### WITH vLLM
```
Prompt: Hello, my name is
Generated:  Shane and I live in the South Bay. I'm very happy to see you. 
I hope you are not in a hurry.
```

### WITHOUT vLLM (transformers)
```
Prompt: Hello, my name is
Generated: Hello, my name is Tashan. I'm 23 years old and have a 7 year old 
daughter who is a wonderful mother and sister. I'm a successful business owner 
with over 50+ years of experience.
```

## Recommendations

1. **For CPU-only deployment of small models**: Use direct transformers/PyTorch for better performance

2. **For production serving with multiple users**: Consider vLLM for its:
   - OpenAI-compatible API
   - Better concurrent request handling
   - Continuous batching capabilities

3. **For GPU deployment**: vLLM shows much better performance gains on GPU where its optimizations are designed to work

4. **For this specific test setup**: Direct transformers is the clear winner for CPU-only inference

## Conclusion

This testing demonstrates that vLLM's optimizations are primarily beneficial for GPU inference and high-concurrency scenarios. For simple CPU-only inference with small models, the direct transformers approach provides significantly better performance with lower latency.

The choice between vLLM and direct transformers should be based on your specific use case:
- **CPU + Small Model + Low Concurrency** → Use transformers directly
- **GPU + Any Model Size** → Use vLLM
- **CPU + High Concurrency + Production API** → Consider vLLM for features
- **CPU + Large Model + Memory Constraints** → Consider vLLM for memory efficiency
