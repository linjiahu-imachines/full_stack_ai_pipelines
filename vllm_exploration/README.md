# vLLM Performance Exploration

**Comprehensive performance evaluation of vLLM across CPU and GPU hardware**

---

## 🎯 Quick Navigation

### 📖 Documentation (in `docs/` folder)

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[EXECUTIVE_REPORT.md](docs/EXECUTIVE_REPORT.md)** ⭐⭐⭐ | Leadership presentation | 20 min |
| **[README.md](docs/README.md)** | Complete project overview | 10 min |
| **[QUICKSTART.md](docs/QUICKSTART.md)** | Essential commands & results | 2 min |
| **[REORGANIZATION_COMPLETE.md](docs/REORGANIZATION_COMPLETE.md)** | Folder organization details | 5 min |

### 🧪 Test Environments

| Folder | Purpose | Hardware |
|--------|---------|----------|
| **[vllm_test/](vllm_test/)** | CPU performance tests | 60-core CPU |
| **[vllm_cpu_test/](vllm_cpu_test/)** | CPU-only batch-500 vLLM assets | CPU-only Azure VM |
| **[vllm_gpu_test/](vllm_gpu_test/)** | GPU performance tests | 4x NVIDIA TITAN RTX |

---

## 🏆 Key Results

### The Bottom Line

| Hardware | Transformers | vLLM | Winner | Performance Gap |
|----------|--------------|------|--------|-----------------|
| **CPU** | 2.67s | 21.79s | Transformers ✅ | **8.16x faster** |
| **GPU** | 1.27s | 0.31s | vLLM ✅ | **4.07x faster** |

### Recommendation

```
📌 For CPU deployment → Use HuggingFace Transformers (8-11x faster)
📌 For GPU deployment → Use vLLM (1.5-4x faster)
```

---

## ⚡ Quick Start

### 1. Verify Setup
```bash
./verify_all.sh
```

### 2. View Executive Report
```bash
cat docs/EXECUTIVE_REPORT.md
```

### 3. Run CPU Tests
```bash
cd vllm_test
./run_tests.sh compare
```

### 4. Run GPU Tests
```bash
cd vllm_gpu_test
./run_gpu_tests.sh compare
```

---

## 📊 Experimental Setup

### Computing Resources

**CPU Environment:**
- 60 physical cores (Intel/AMD x86)
- 62.88 GiB KV cache
- vLLM 0.15.1+cpu, PyTorch 2.10.0+cpu

**GPU Environment:**
- 4x NVIDIA TITAN RTX (24GB VRAM each)
- CUDA 12.4, Compute Capability 7.5
- vLLM 0.15.1, PyTorch 2.6.0+cu124

### Test Model

- **Model**: facebook/opt-125m
- **Parameters**: 125 million
- **Size**: ~500 MB

---

## 📁 Folder Structure

```
vllm_exploration/
├── README.md                    ← You are here
├── verify_all.sh                ← Verification script
│
├── docs/                        ← All documentation
│   ├── EXECUTIVE_REPORT.md      ← Leadership report ⭐
│   ├── README.md                ← Project overview
│   ├── QUICKSTART.md            ← Quick reference
│   └── REORGANIZATION_COMPLETE.md
│
├── vllm_test/                   ← CPU Tests
│   ├── test_*.py                ← Test scripts
│   ├── run_tests.sh             ← Run CPU tests
│   ├── *.md                     ← CPU reports
│   ├── outputs/                 ← Saved results
│   └── venv/                    ← CPU environment
│
├── vllm_cpu_test/               ← CPU-only batch-500 assets
│   ├── test2_vllm_cpu.py        ← Standalone vLLM CPU benchmark
│   ├── test_vllm_cpu_quick.sh   ← CPU quick runner
│   └── *.md                     ← CPU-only reports
│
└── vllm_gpu_test/               ← GPU Tests
    ├── test_*.py                ← Test scripts
    ├── run_gpu_tests.sh         ← Run GPU tests
    ├── *.md                     ← GPU reports
    └── venv/                    ← GPU environment
```

---

## 🎓 For Different Audiences

### For Leadership
→ Read `docs/EXECUTIVE_REPORT.md`

### For Technical Teams
→ Read `vllm_test/COMPARISON_REPORT.md` (CPU)
→ Read `vllm_gpu_test/GPU_TEST_RESULTS.md` (GPU)

### For Quick Decision
→ Read `docs/QUICKSTART.md`

---

## ✅ Status

**All components verified and operational:**

- ✅ Folder structure organized
- ✅ CPU tests: Working
- ✅ GPU tests: Working
- ✅ CUDA: Available (4 GPUs detected)
- ✅ Documentation: Consolidated in `docs/`
- ✅ Scripts: Updated and executable

**Everything is runnable and testable!**

---

## 📞 Support

**Project Location**: `/home/linhu/projects/vllm_exploration/`

**Quick Help:**
- Verify setup: `./verify_all.sh`
- View docs: `ls docs/`
- Run CPU tests: `cd vllm_test && ./run_tests.sh`
- Run GPU tests: `cd vllm_gpu_test && ./run_gpu_tests.sh`

---

**Project Status**: ✅ Complete  
**Last Updated**: February 11, 2026  
**Hardware**: 60-core CPU + 4x NVIDIA TITAN RTX (24GB)
