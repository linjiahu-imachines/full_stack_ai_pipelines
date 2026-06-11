# Quick start

## Install (Jetson Thor / aarch64)

```bash
cd /home/linhu/projects/sglang_exploration
./setup_sglang_env.sh
source ./activate_sglang_env.sh
```

## Run

```bash
python sglang_test/run_offline_batch_experiment.py --device cuda --model Qwen/Qwen3-1.7B --batch-size 1 --max-new-tokens 16
```

See [sglang_test/README.md](../sglang_test/README.md) for platform notes (Thor / `sm_110a`, `nvcc`, and the `torch_memory_saver` stub).
