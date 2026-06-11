"""
CPU-only comparison for Qwen/Qwen3-1.7B:
1) qwen3-1.7B without vLLM (Hugging Face Transformers)
2) qwen3-1.7B with vLLM

Batch-size sweep: default 1, 500, 1000, 2000 concurrent prompts per experiment.
See README.md and QWEN3_1_7B_CPU_EVALUATION.md for prior methodology.
"""

from __future__ import annotations

import argparse
from typing import Any
import gc
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-1.7B"


def build_prompts(batch_size: int) -> list[str]:
    return [f"Question {i}: Explain CPU inference trade-offs in one paragraph." for i in range(batch_size)]


def _read_lscpu_key_values() -> dict:
    try:
        proc = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10, check=True)
    except Exception:
        return {}
    data: dict = {}
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip()
    return data


def collect_system_info() -> dict:
    lscpu = _read_lscpu_key_values()
    vm = psutil.virtual_memory()
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "cpu": {
            "vendor": lscpu.get("Vendor ID"),
            "model_name": lscpu.get("Model name"),
            "architecture": lscpu.get("Architecture"),
            "sockets": lscpu.get("Socket(s)"),
            "cores_per_socket": lscpu.get("Core(s) per socket"),
            "threads_per_core": lscpu.get("Thread(s) per core"),
            "logical_cpus": lscpu.get("CPU(s)", str(psutil.cpu_count(logical=True))),
            "physical_cores": str(psutil.cpu_count(logical=False)),
            "max_mhz": lscpu.get("CPU max MHz"),
            "min_mhz": lscpu.get("CPU min MHz"),
        },
        "cache": {
            "l1d": lscpu.get("L1d cache"),
            "l1i": lscpu.get("L1i cache"),
            "l2": lscpu.get("L2 cache"),
            "l3": lscpu.get("L3 cache"),
        },
        "memory": {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
        },
    }


def load_transformers_model(model: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_obj = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
    model_obj = model_obj.to("cpu")
    model_obj.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model_obj


def run_transformers_cpu_loaded(
    tokenizer: Any,
    model_obj: Any,
    model_name: str,
    batch_size: int,
    max_new_tokens: int,
) -> dict:
    prompts = build_prompts(batch_size)
    t0 = time.time()
    micro_batch_size = 1 if batch_size == 1 else min(25, batch_size)
    for i in range(0, batch_size, micro_batch_size):
        chunk = prompts[i : i + micro_batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cpu")
        with torch.no_grad():
            _ = model_obj.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    elapsed = time.time() - t0

    return {
        "engine": "qwen3-1.7B without vLLM",
        "model": model_name,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "micro_batch_size": micro_batch_size,
        "time_s": round(elapsed, 2),
        "per_prompt_ms": round((elapsed / batch_size) * 1000, 2),
        "throughput_prompts_per_s": round(batch_size / elapsed, 2) if elapsed > 0 else 0.0,
    }


def resolve_vllm_python(script_dir: str) -> str:
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    candidates = [
        os.environ.get("CPU_TEST_PYTHON", ""),
        os.environ.get("VLLM_CPU_PYTHON", ""),
        os.path.join(project_root, "vllm_cpu_venv", "bin", "python3"),
        os.path.join(project_root, "vllm_test", "venv", "bin", "python3"),
        sys.executable,
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise RuntimeError("No usable Python interpreter found for vLLM CPU test")


def run_vllm_cpu(
    model: str,
    batch_size: int,
    max_new_tokens: int,
    max_num_seqs: int,
    script_dir: str,
) -> dict:
    python_bin = resolve_vllm_python(script_dir)
    worker_script = os.path.join(script_dir, "test_qwen3_vllm_cpu.py")

    proc = subprocess.run(
        [
            python_bin,
            worker_script,
            "--model",
            model,
            "--batch-size",
            str(batch_size),
            "--max-new-tokens",
            str(max_new_tokens),
            "--max-num-seqs",
            str(max_num_seqs),
        ],
        cwd=script_dir,
        text=True,
        capture_output=True,
        timeout=7200,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"vLLM subprocess failed with exit code {proc.returncode}: {proc.stderr[-2000:] if proc.stderr else ''}"
        )

    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT_JSON__:"):
            return json.loads(line.split(":", 1)[1])
    raise RuntimeError("vLLM subprocess did not return JSON result")


def parse_batch_sizes(arg: str | None, legacy_single: int | None, legacy_large: int | None) -> list[int]:
    if legacy_single is not None and legacy_large is not None:
        return [legacy_single, legacy_large]
    if not arg or not arg.strip():
        return [1, 500, 1000, 2000]
    out = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        return [1, 500, 1000, 2000]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-1.7B CPU-only comparison (batch sweep)")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,500,1000,2000",
        help="Comma-separated batch sizes (default: 1,500,1000,2000)",
    )
    parser.add_argument(
        "--single-batch-size",
        type=int,
        default=None,
        help="Legacy: if set with --large-batch-size, only those two sizes run (ignores --batch-sizes)",
    )
    parser.add_argument("--large-batch-size", type=int, default=None, help="Legacy: pair with --single-batch-size")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=2048,
        help="vLLM max_num_seqs; should be >= largest batch size (default 2048 for 2000-batch runs)",
    )
    parser.add_argument(
        "--transformers-only",
        action="store_true",
        help="Run only Hugging Face Transformers on CPU (skip vLLM subprocess).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSON path (default: qwen3_1_7b_cpu_results.json or _transformers_only.json)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    started_at = datetime.now(timezone.utc).isoformat()

    batch_sizes = parse_batch_sizes(args.batch_sizes, args.single_batch_size, args.large_batch_size)
    max_bs = max(batch_sizes)
    if args.vllm_max_num_seqs < max_bs:
        print(
            f"WARNING: --vllm-max-num-seqs ({args.vllm_max_num_seqs}) < largest batch ({max_bs}). "
            f"vLLM may serialize work; consider --vllm-max-num-seqs {max_bs}",
            file=sys.stderr,
        )

    print("=" * 80)
    print("Qwen3-1.7B CPU-only comparison")
    print("=" * 80)
    print(f"Model: {args.model}")
    system_info = collect_system_info()

    print(f"Batch sizes: {batch_sizes}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"vLLM max_num_seqs: {args.vllm_max_num_seqs}")
    print(
        f"CPU: {system_info['cpu']['model_name']} | "
        f"{system_info['cpu']['physical_cores']} physical / {system_info['cpu']['logical_cpus']} logical"
    )
    print(
        f"Memory: total {system_info['memory']['total_gb']} GB, "
        f"available {system_info['memory']['available_gb']} GB"
    )
    print()

    results_without: dict[str, dict] = {}
    tokenizer = None
    model_obj = None

    try:
        print("Loading Transformers model once for all batch sizes...")
        tokenizer, model_obj = load_transformers_model(args.model)
        for bs in batch_sizes:
            print(f"  [without vLLM] batch_size={bs} ...", flush=True)
            results_without[str(bs)] = run_transformers_cpu_loaded(
                tokenizer, model_obj, args.model, bs, args.max_new_tokens
            )
            print(
                f"    time {results_without[str(bs)]['time_s']}s, "
                f"{results_without[str(bs)]['per_prompt_ms']} ms/prompt"
            )
    finally:
        del model_obj
        del tokenizer
        gc.collect()

    results_with: dict[str, dict] | None = None
    ratios: dict[str, dict] | None = None

    if not args.transformers_only:
        results_with = {}
        ratios = {}
        for bs in batch_sizes:
            print(f"  [with vLLM] batch_size={bs} ...", flush=True)
            results_with[str(bs)] = run_vllm_cpu(
                args.model, bs, args.max_new_tokens, args.vllm_max_num_seqs, script_dir
            )
            r = results_with[str(bs)]
            print(f"    time {r['time_s']}s (init {r['init_time_s']}s), {r['per_prompt_ms']} ms/prompt")
            tw = results_without[str(bs)]["time_s"]
            tv = r["time_s"]
            ratios[str(bs)] = {
                "time_ratio_vllm_over_transformers": round(tv / tw, 4) if tw else None,
                "throughput_ratio_vllm_over_transformers": round(
                    r["throughput_prompts_per_s"] / results_without[str(bs)]["throughput_prompts_per_s"],
                    4,
                )
                if results_without[str(bs)]["throughput_prompts_per_s"]
                else None,
            }

    summary: dict = {
        "meta": {
            "started_at_utc": started_at,
            "model": args.model,
            "batch_sizes": batch_sizes,
            "max_new_tokens": args.max_new_tokens,
            "vllm_max_num_seqs": args.vllm_max_num_seqs,
            "transformers_only": args.transformers_only,
            "experiment_names": (
                ["qwen3-1.7B without vLLM"]
                if args.transformers_only
                else ["qwen3-1.7B without vLLM", "qwen3-1.7B with vLLM"]
            ),
            "system_info": system_info,
        },
        "by_batch_size": {},
    }

    for bs in batch_sizes:
        key = str(bs)
        entry: dict = {"qwen3-1.7B without vLLM": results_without[key]}
        if results_with is not None:
            entry["qwen3-1.7B with vLLM"] = results_with[key]
        summary["by_batch_size"][key] = entry

    if ratios:
        summary["ratios"] = ratios

    out_name = args.output.strip()
    if not out_name:
        out_name = (
            "qwen3_1_7b_cpu_results_transformers_only.json"
            if args.transformers_only
            else "qwen3_1_7b_cpu_results.json"
        )
    out_path = os.path.join(script_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 80)
    print(f"Saved results: {out_path}")
    if ratios:
        print(json.dumps(ratios, indent=2))


if __name__ == "__main__":
    main()
