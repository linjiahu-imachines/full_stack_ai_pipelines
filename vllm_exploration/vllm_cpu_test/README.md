# vLLM CPU Test Assets

CPU-only tests and reports have been separated into this folder.

## Contents

- `test2_vllm_cpu.py` - standalone vLLM CPU batch-500 benchmark used by the 4-way comparison runner
- `test_vllm_cpu_quick.sh` - helper script to run the CPU benchmark with a CPU-capable Python environment
- `CPU_ONLY_BATCH500_NEW_MACHINE_SUMMARY.md` - CPU-only benchmark summary report

## Related Runner

`vllm_gpu_test/test_batch500_complete.py` invokes `test2_vllm_cpu.py` from this folder for its CPU vLLM test phase.
