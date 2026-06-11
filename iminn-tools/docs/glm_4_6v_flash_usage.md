# GLM-4.6V-Flash Model Usage Guide

This document provides a comprehensive guide for using the GLM-4.6V-Flash multimodal vision-language model in the iminn-tools project.

**Date:** January 5, 2025  
**Model Repository:** [ggml-org/GLM-4.6V-Flash-GGUF](https://huggingface.co/ggml-org/GLM-4.6V-Flash-GGUF)  
**Base Model:** [zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Model Architecture](#2-model-architecture)
3. [Model Files](#3-model-files)
4. [Setup & Installation](#4-setup--installation)
5. [Running the Model](#5-running-the-model)
6. [Test Configurations](#6-test-configurations)
7. [Usage Examples](#7-usage-examples)
8. [Troubleshooting](#8-troubleshooting)
9. [References](#9-references)

---

## 1. Overview

GLM-4.6V-Flash is a **multimodal vision-language model** that can process both text and images. It's based on the GLM (General Language Model) architecture and has been converted to GGUF format for use with llama.cpp.

### Key Features

- **Multimodal Capabilities**: Processes both text and images
- **Quantized Formats**: Available in Q4_K_M quantization (6.17 GB)
- **Vision Projection**: Separate vision projection file for image processing
- **llama.cpp Compatible**: Works with all llama.cpp targets (x86, RISC-V, IMI, etc.)

### Model Specifications

- **Main Model**: `GLM-4.6V-Flash-Q4_K_M.gguf` (6.17 GB)
- **Vision Projection**: `mmproj-GLM-4.6V-Flash-Q8_0.gguf` (980 MB)
- **Total Size**: ~7.15 GB
- **Quantization**: Q4_K_M (main model), Q8_0 (vision projection)

---

## 2. Model Architecture

### Two-Component Architecture

GLM-4.6V-Flash uses a two-file architecture:

```
┌─────────────────────────────────────────────────────────┐
│  Main Model (GLM-4.6V-Flash-Q4_K_M.gguf)               │
│  - Language model weights                                │
│  - Transformer layers                                     │
│  - Attention mechanisms                                   │
│  - Feed-forward networks                                  │
└─────────────────────────────────────────────────────────┘
                          ↑
                          │
┌─────────────────────────────────────────────────────────┐
│  Vision Projection (mmproj-GLM-4.6V-Flash-Q8_0.gguf)   │
│  - Vision encoder projection weights                     │
│  - Image-to-text embedding conversion                    │
│  - Multimodal bridge                                     │
└─────────────────────────────────────────────────────────┘
```

### Why Two Files?

1. **Modularity**: Use language model without vision if needed
2. **Different Quantization**: Vision projection uses higher precision (Q8_0) for better image understanding
3. **Flexibility**: Swap vision encoders without changing the main model
4. **Smaller Downloads**: Download only what you need

### Processing Flow

```
Text Input → Tokenizer → Language Model → Text Output
                                    ↑
Image Input → Vision Encoder → mmproj (projection) ──┘
```

The vision projection (`mmproj`) converts image embeddings into the same embedding space as text tokens, allowing the language model to process both modalities together.

---

## 3. Model Files

### File Locations

Model files are stored in the standard models directory:

```
/home/linhu/repo/iminn-tools/dev_env/llama.cpp/models/
├── GLM-4.6V-Flash-Q4_K_M.gguf          (6.17 GB)
└── mmproj-GLM-4.6V-Flash-Q8_0.gguf     (980 MB)
```

### Download Status

The model files are automatically downloaded during initialization or can be manually downloaded:

```bash
cd /home/linhu/repo/iminn-tools/dev_env/llama.cpp/models/

# Main model
wget https://huggingface.co/ggml-org/GLM-4.6V-Flash-GGUF/resolve/main/GLM-4.6V-Flash-Q4_K_M.gguf

# Vision projection
wget https://huggingface.co/ggml-org/GLM-4.6V-Flash-GGUF/resolve/main/mmproj-GLM-4.6V-Flash-Q8_0.gguf
```

### Verification

Check if files are present:

```bash
ls -lh /home/linhu/repo/iminn-tools/dev_env/llama.cpp/models/GLM-4.6V-Flash-*
```

Expected output:
```
-rw-rw-r-- 1 linhu linhu 5.8G Jan  5 14:18 GLM-4.6V-Flash-Q4_K_M.gguf
-rw-rw-r-- 1 linhu linhu 935M Jan  5 14:18 mmproj-GLM-4.6V-Flash-Q8_0.gguf
```

---

## 4. Setup & Installation

### Prerequisites

1. **llama.cpp built**: Ensure llama.cpp is initialized and built
   ```bash
   iminnt -t llama_imi init
   iminnt -t llama_imi build
   ```

2. **Model files downloaded**: Verify model files exist (see [Model Files](#3-model-files))

3. **Test configurations added**: Test configurations are already added to `src/iminnt/llamacpp.py`

### Configuration Location

Test configurations are defined in:

**File**: `src/iminnt/llamacpp.py:403-413`

```python
# GLM-4.6V-Flash model tests (multimodal vision-language model)
"test_glm_4_6v_text": {"bin": f"{self.install_dir}/bin/llama-cli", 
    "args": f"-m {self.root}/models/GLM-4.6V-Flash-Q4_K_M.gguf \
                --seed 42 \
                -t 1 -ngl 0 -n 32 \
                -no-cnv -st --no-warmup \
                --file {PROMPTS_DIR}/hello-world.txt"},
"test_glm_4_6v_multimodal": {"bin": f"{self.install_dir}/bin/llama-cli", 
    "args": f"-m {self.root}/models/GLM-4.6V-Flash-Q4_K_M.gguf \
                --mmproj {self.root}/models/mmproj-GLM-4.6V-Flash-Q8_0.gguf \
                --seed 42 \
                -t 1 -ngl 0 -n 32 \
                -no-cnv -st --no-warmup \
                --file {PROMPTS_DIR}/hello-world.txt"},
```

---

## 5. Running the Model

### Method 1: Using Default Test Commands

The easiest way to run the model is using the pre-configured test commands:

#### Text-Only Mode

```bash
# x86 target (native, no QEMU) - RECOMMENDED for GLM-4.6V-Flash
iminnt -t llama_x86 run -d test_glm_4_6v_text

# IMI target (RISC-V with QEMU) - May hang due to model size
# iminnt -t llama_imi run -d test_glm_4_6v_text

# RVV target (RISC-V Vector) - May also have performance issues
# iminnt -t llama_rvv run -d test_glm_4_6v_text
```

**⚠️ Important Note**: GLM-4.6V-Flash (9.4B parameters) is a large model that may hang or run extremely slowly when executed through QEMU emulation (`llama_imi`). For best results, use the native `llama_x86` target.

#### Multimodal Mode (with Vision Projection)

```bash
# IMI target
iminnt -t llama_imi run -d test_glm_4_6v_multimodal

# x86 target
iminnt -t llama_x86 run -d test_glm_4_6v_multimodal
```

### Method 2: Using Custom Arguments

For more control, use custom arguments:

```bash
# Text-only
iminnt -t llama_imi run --bin llama-cli --args "-m dev_env/llama.cpp/models/GLM-4.6V-Flash-Q4_K_M.gguf --seed 42 -t 1 -ngl 0 -n 32 -no-cnv -st --no-warmup --file src/iminnt/resources/prompts/hello-world.txt"

# Multimodal
iminnt -t llama_imi run --bin llama-cli --args "-m dev_env/llama.cpp/models/GLM-4.6V-Flash-Q4_K_M.gguf --mmproj dev_env/llama.cpp/models/mmproj-GLM-4.6V-Flash-Q8_0.gguf --seed 42 -t 1 -ngl 0 -n 32 -no-cnv -st --no-warmup --file src/iminnt/resources/prompts/hello-world.txt"
```

### Method 3: Direct Execution

For quick testing without the iminnt wrapper:

```bash
cd /home/linhu/repo/iminn-tools/dev_env/llama.cpp/llamacpp-imi-install/bin

# Text-only
./llama-cli -m ../../models/GLM-4.6V-Flash-Q4_K_M.gguf \
            --seed 42 -t 1 -ngl 0 -n 32 \
            -no-cnv -st --no-warmup \
            --file ../../../../src/iminnt/resources/prompts/hello-world.txt

# Multimodal
./llama-cli -m ../../models/GLM-4.6V-Flash-Q4_K_M.gguf \
            --mmproj ../../models/mmproj-GLM-4.6V-Flash-Q8_0.gguf \
            --seed 42 -t 1 -ngl 0 -n 32 \
            -no-cnv -st --no-warmup \
            --file ../../../../src/iminnt/resources/prompts/hello-world.txt
```

### Command-Line Arguments Explained

| Argument | Value | Description |
|----------|-------|-------------|
| `-m` | `GLM-4.6V-Flash-Q4_K_M.gguf` | Main model file path |
| `--mmproj` | `mmproj-GLM-4.6V-Flash-Q8_0.gguf` | Vision projection file (multimodal only) |
| `--seed` | `42` | Random seed for reproducibility |
| `-t` | `1` | Number of threads |
| `-ngl` | `0` | Number of GPU layers (0 = CPU only) |
| `-n` | `32` | Number of tokens to generate |
| `-no-cnv` | - | Disable conversation mode |
| `-st` | - | Simple tokenization |
| `--no-warmup` | - | Skip warmup iterations |
| `--file` | `hello-world.txt` | Input prompt file |

---

## 6. Test Configurations

### Available Test Configurations

Two test configurations are available in `src/iminnt/llamacpp.py`:

1. **`test_glm_4_6v_text`**: Text-only mode
   - Uses only the main model file
   - Faster execution
   - Suitable for text generation tasks

2. **`test_glm_4_6v_multimodal`**: Full multimodal mode
   - Uses both main model and vision projection
   - Enables image processing capabilities
   - Required for vision-language tasks

### Adding Custom Configurations

To add custom test configurations, edit `src/iminnt/llamacpp.py` in the `default_runs` property:

```python
"test_glm_4_6v_custom": {"bin": f"{self.install_dir}/bin/llama-cli", 
    "args": f"-m {self.root}/models/GLM-4.6V-Flash-Q4_K_M.gguf \
                --mmproj {self.root}/models/mmproj-GLM-4.6V-Flash-Q8_0.gguf \
                --seed 42 \
                -t 4 -ngl 0 -n 64 \
                -no-cnv -st --no-warmup \
                --file {PROMPTS_DIR}/hello-world.txt"},
```

---

## 7. Usage Examples

### Example 1: Basic Text Generation

```bash
# Run text-only generation
iminnt -t llama_imi run -d test_glm_4_6v_text
```

**Expected Output**: Text generation based on the prompt in `hello-world.txt`

### Example 2: Multimodal Processing

```bash
# Run with vision projection enabled
iminnt -t llama_imi run -d test_glm_4_6v_multimodal
```

**Note**: For actual image processing, you would need to provide an image file using the `--image` flag (if supported by your llama.cpp version).

### Example 3: Multi-threaded Execution

Create a custom configuration for multi-threaded execution:

```bash
iminnt -t llama_imi run --bin llama-cli --args "-m dev_env/llama.cpp/models/GLM-4.6V-Flash-Q4_K_M.gguf --seed 42 -t 4 -ngl 0 -n 64 -no-cnv -st --no-warmup --file src/iminnt/resources/prompts/hello-world.txt"
```

### Example 4: System Mode (QEMU Full System)

For system mode execution, see the [QEMU All-in-One Guide](qemu_all_in_one_guide.md) for details on running models in QEMU system mode.

---

## 8. Troubleshooting

### Issue: Model Hangs During Generation on IMI (QEMU)

**Symptom**: Model loads successfully but hangs at generation step when using `llama_imi` target

**Observed Behavior**:
- ✅ Works on `llama_x86` (native x86, no QEMU)
- ❌ Hangs on `llama_imi` (RISC-V with QEMU emulation)
- Model loads successfully, but generation never completes

**Root Cause**:
- GLM-4.6V-Flash is a large model (9.4B parameters)
- QEMU emulation overhead makes generation extremely slow or causes hangs
- The model may be too large for efficient QEMU emulation

**Solutions**:
1. **Use x86 target instead** (recommended for this model):
   ```bash
   iminnt -t llama_x86 run -d test_glm_4_6v_text
   ```

2. **Reduce context size** (if you must use IMI):
   - Add `-c 512` to reduce context window
   - This may help but may still be very slow

3. **Reduce tokens to generate**:
   - Change `-n 32` to `-n 8` or `-n 4` for testing
   - This reduces the amount of computation needed

4. **Check if process is actually running**:
   ```bash
   ps aux | grep llama-cli
   ```
   - If process exists, it may just be very slow (wait longer)
   - If process is stuck, use Ctrl+C and try x86 target

**Note**: This is a known limitation when running very large models through QEMU emulation. For production use with GLM-4.6V-Flash, prefer native x86 execution.

### Issue: Model File Not Found

**Error**: `llama_model_load: failed to load model`

**Solution**: 
1. Verify model files exist:
   ```bash
   ls -lh /home/linhu/repo/iminn-tools/dev_env/llama.cpp/models/GLM-4.6V-Flash-*
   ```
2. Re-download if missing (see [Model Files](#3-model-files))

### Issue: Vision Projection Not Loading

**Error**: `failed to load mmproj file`

**Solution**:
1. Check that `mmproj-GLM-4.6V-Flash-Q8_0.gguf` exists
2. Verify file permissions
3. For text-only tasks, you can omit `--mmproj` flag

### Issue: Out of Memory

**Error**: `out of memory` or `failed to allocate`

**Solution**:
- The model requires ~6-7 GB of RAM/VRAM
- Reduce context size with `-c` flag
- Use fewer threads with `-t 1`
- Ensure sufficient system memory

### Issue: Unsupported Architecture

**Error**: Model architecture not recognized

**Solution**:
- Ensure llama.cpp is up to date
- Check that GLM architecture is supported in your llama.cpp version
- Verify model file integrity

### Issue: Slow Performance

**Solutions**:
- Use fewer threads for small models: `-t 1`
- Increase threads for larger contexts: `-t 4` or `-t 8`
- Use GPU acceleration if available: `-ngl <layers>`
- Reduce generation length: `-n 16` instead of `-n 32`
- **For large models like GLM-4.6V-Flash**: Prefer native x86 execution over QEMU emulation

---

## 9. References

### Model Information

- **Hugging Face Repository**: [ggml-org/GLM-4.6V-Flash-GGUF](https://huggingface.co/ggml-org/GLM-4.6V-Flash-GGUF)
- **Base Model**: [zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash)
- **README**: [Model README](https://huggingface.co/ggml-org/GLM-4.6V-Flash-GGUF/blob/main/README.md)

### Related Documentation

- [llama.cpp Lifecycle](llamacpp_lifecycle.md) - Complete lifecycle management
- [llama.cpp Execution Flow](llama_cpp_execution_flow.md) - Execution flow details
- [QEMU All-in-One Guide](qemu_all_in_one_guide.md) - QEMU system mode usage
- [llama.cpp Internal Architecture](llama_cpp_internal_architecture.md) - Architecture details

### Code References

- **Test Configurations**: `src/iminnt/llamacpp.py:403-413`
- **Model Directory**: `dev_env/llama.cpp/models/`
- **Prompts Directory**: `src/iminnt/resources/prompts/`

### External Resources

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp) - Main llama.cpp repository
- [GGUF Format Documentation](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - GGUF format specification

---

## Appendix: Quick Reference

### Quick Start Commands

```bash
# 1. Verify model files
ls -lh dev_env/llama.cpp/models/GLM-4.6V-Flash-*

# 2. Run text-only test
iminnt -t llama_imi run -d test_glm_4_6v_text

# 3. Run multimodal test
iminnt -t llama_imi run -d test_glm_4_6v_multimodal
```

### File Paths

- **Models**: `/home/linhu/repo/iminn-tools/dev_env/llama.cpp/models/`
- **Binaries**: `/home/linhu/repo/iminn-tools/dev_env/llama.cpp/llamacpp-imi-install/bin/`
- **Prompts**: `/home/linhu/repo/iminn-tools/src/iminnt/resources/prompts/`
- **Config**: `/home/linhu/repo/iminn-tools/src/iminnt/llamacpp.py`

---

**Last Updated**: January 5, 2025

