# SGLang exploration

Performance and behavior experiments with [SGLang](https://github.com/sgl-project/sglang), parallel to the sibling [vllm_exploration](../vllm_exploration) project under `/home/linhu/projects/`.

## Documentation

| Location | Purpose |
|----------|---------|
| [docs/README.md](docs/README.md) | Doc index |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Commands and workflow (stub) |

## Test environments

| Folder | Purpose |
|--------|---------|
| [sglang_test/](sglang_test/) | General SGLang tests and smoke checks |
| [sglang_cpu_test/](sglang_cpu_test/) | CPU-oriented runs and assets |
| [sglang_gpu_test/](sglang_gpu_test/) | GPU-oriented runs and assets |

## SGLang Python environment (Jetson Thor / aarch64)

```bash
./setup_sglang_env.sh          # creates sglang_venv, installs PyTorch cu130 + SGLang 0.5.9 + CUDA 12 user libs
source ./activate_sglang_env.sh
```

Then run experiments under [sglang_test/](sglang_test/) (see that README for **Qwen3-1.7B** and known platform limits).

## Verify layout

```bash
./verify_all.sh
```

Add install notes, benchmarks, and scripts under the folders above as experiments grow.
