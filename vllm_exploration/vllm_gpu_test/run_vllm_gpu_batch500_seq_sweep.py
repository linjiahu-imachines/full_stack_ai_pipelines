#!/usr/bin/env python3
"""
vLLM on GPU: 500 prompts, sweep max_num_seqs (--max-num-seqs-list; default 64,128,256).

Uses vllm_gpu_test/venv (CUDA vLLM) and gpu_env.py (Jetson memory / patches).
Writes JSON next to CPU sweep results by default (vllm_cpu_qwen3_1_7b_test/).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

GPU_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(GPU_SCRIPT_DIR, ".."))
QWEN_TEST_DIR = os.path.join(PROJECT_ROOT, "vllm_cpu_qwen3_1_7b_test")
sys.path.insert(0, QWEN_TEST_DIR)
from test_qwen3_cpu_compare import collect_system_info  # noqa: E402


def resolve_gpu_python() -> str:
    candidates = [
        os.environ.get("GPU_VLLM_PYTHON", ""),
        os.environ.get("GPU_TEST_PYTHON", ""),
        os.path.join(GPU_SCRIPT_DIR, "venv", "bin", "python3"),
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise RuntimeError(
        "No CUDA Python found. Set GPU_VLLM_PYTHON or use "
        f"{GPU_SCRIPT_DIR}/venv/bin/python3 (run install_jetson_gpu_deps.sh if missing)."
    )


def cuda_meta() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "device_0_name": torch.cuda.get_device_name(0),
            "device_0_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM GPU batch-500 max_num_seqs sweep")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--max-num-seqs-list", type=str, default="64,128,256")
    parser.add_argument("--tag", type=str, default="imu_thor")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=f"Default: {QWEN_TEST_DIR}",
    )
    args = parser.parse_args()

    seq_values = [int(x.strip()) for x in args.max_num_seqs_list.split(",") if x.strip()]
    if not seq_values:
        raise SystemExit("No values in --max-num-seqs-list")

    python_bin = resolve_gpu_python()
    worker = os.path.join(GPU_SCRIPT_DIR, "test_qwen3_vllm_gpu_batch.py")
    out_dir = os.path.abspath(args.output_dir.strip() or QWEN_TEST_DIR)

    print(f"Using Python: {python_bin}")
    print(f"Worker: {worker}")
    print(f"Model: {args.model} | batch_size={args.batch_size} | max_new_tokens={args.max_new_tokens}")
    print(f"max_num_seqs sweep: {seq_values}")
    print(f"Output directory: {out_dir}")
    print()

    env = os.environ.copy()

    for max_num_seqs in seq_values:
        effective = max(1, min(args.batch_size, max_num_seqs))
        fname = f"qwen3_1_7b_gpu_results_batch{args.batch_size}_seq{effective}_{args.tag}.json"
        out_path = os.path.join(out_dir, fname)

        started = datetime.now(timezone.utc).isoformat()
        system_info = collect_system_info()
        system_info["gpu"] = cuda_meta()

        cmd = [
            python_bin,
            worker,
            "--model",
            args.model,
            "--batch-size",
            str(args.batch_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-num-seqs",
            str(max_num_seqs),
            "--output",
            out_path + ".partial",
        ]

        print(
            f"=== max_num_seqs={max_num_seqs} (effective {effective}) -> {fname} ===",
            flush=True,
        )
        proc = subprocess.run(
            cmd,
            cwd=GPU_SCRIPT_DIR,
            env=env,
            text=True,
            timeout=7200,
        )

        partial = out_path + ".partial"
        if proc.returncode != 0:
            if os.path.isfile(partial):
                os.remove(partial)
            raise RuntimeError(f"vLLM GPU worker failed with exit code {proc.returncode}")

        if not os.path.isfile(partial):
            raise RuntimeError(f"Expected partial output missing: {partial}")

        with open(partial, encoding="utf-8") as f:
            inner = json.load(f)
        os.remove(partial)

        doc = {
            "meta": {
                "started_at_utc": started,
                "model": args.model,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "max_num_seqs_requested": max_num_seqs,
                "max_num_seqs_effective": inner.get("max_num_seqs"),
                "experiment": "vLLM GPU (batch 500, max_num_seqs sweep); same prompts as CPU harness",
                "system_info": system_info,
            },
            "qwen3-1.7B with vLLM (GPU)": inner,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)

        print(f"Saved: {out_path}", flush=True)
        print()


if __name__ == "__main__":
    main()
