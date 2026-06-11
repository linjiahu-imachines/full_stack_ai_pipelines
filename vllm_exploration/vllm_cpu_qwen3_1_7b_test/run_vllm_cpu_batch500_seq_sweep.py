#!/usr/bin/env python3
"""
vLLM CPU only: batch size 500, sweep max_num_seqs (--max-num-seqs-list; default 64,128,256).

Writes one JSON per seq value, named like:
  qwen3_1_7b_cpu_results_batch500_seq64_imu_thor.json

Does not run Hugging Face Transformers (compare that separately if needed).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Import after path setup
sys.path.insert(0, SCRIPT_DIR)
from test_qwen3_cpu_compare import collect_system_info  # noqa: E402


def resolve_cpu_python() -> str:
    candidates = [
        os.environ.get("CPU_TEST_PYTHON", ""),
        os.environ.get("VLLM_CPU_PYTHON", ""),
        os.path.join(PROJECT_ROOT, "vllm_cpu_venv", "bin", "python3"),
        os.path.join(PROJECT_ROOT, "vllm_test", "venv", "bin", "python3"),
        sys.executable,
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise RuntimeError(
        "No Python interpreter found. Set CPU_TEST_PYTHON or create "
        f"{PROJECT_ROOT}/vllm_cpu_venv"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM CPU batch-500 max_num_seqs sweep")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument(
        "--max-num-seqs-list",
        type=str,
        default="64,128,256",
        help="Comma-separated max_num_seqs values (default: 64,128,256)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="imu_thor",
        help="Suffix for output filenames: ..._seq{N}_{tag}.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for JSON files (default: this script's directory)",
    )
    args = parser.parse_args()

    seq_values = []
    for part in args.max_num_seqs_list.split(","):
        part = part.strip()
        if part:
            seq_values.append(int(part))
    if not seq_values:
        raise SystemExit("No values in --max-num-seqs-list")

    python_bin = resolve_cpu_python()
    worker = os.path.join(SCRIPT_DIR, "test_qwen3_vllm_cpu.py")
    out_dir = os.path.abspath(args.output_dir.strip() or SCRIPT_DIR)

    print(f"Using Python: {python_bin}")
    print(f"Model: {args.model} | batch_size={args.batch_size} | max_new_tokens={args.max_new_tokens}")
    print(f"max_num_seqs sweep: {seq_values}")
    print(f"Output directory: {out_dir}")
    print()

    env = os.environ.copy()
    env.pop("VLLM_CPU_PYTHON", None)

    for max_num_seqs in seq_values:
        effective = max(1, min(args.batch_size, max_num_seqs))
        fname = f"qwen3_1_7b_cpu_results_batch{args.batch_size}_seq{effective}_{args.tag}.json"
        out_path = os.path.join(out_dir, fname)

        started = datetime.now(timezone.utc).isoformat()
        system_info = collect_system_info()

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

        print(f"=== max_num_seqs={max_num_seqs} (effective {effective}) -> {fname} ===", flush=True)
        proc = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            env=env,
            text=True,
            timeout=7200,
        )

        partial = out_path + ".partial"
        if proc.returncode != 0:
            if os.path.isfile(partial):
                os.remove(partial)
            raise RuntimeError(f"vLLM worker failed with exit code {proc.returncode}")

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
                "experiment": "vLLM CPU only (batch 500, max_num_seqs sweep)",
                "system_info": system_info,
            },
            "qwen3-1.7B with vLLM": inner,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)

        print(f"Saved: {out_path}", flush=True)
        print()


if __name__ == "__main__":
    main()
