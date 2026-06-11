#!/usr/bin/env python3
"""
Offline batch inference with SGLang Engine — CPU and GPU.

Default workload: ``data/queries_512.jsonl`` (512 prompts). Use ``--batch-size`` for how
many prompts each ``generate()`` call processes (8, 16, 32, …). Use ``--batch-size 0`` for
one ``generate()`` over all prompts. For tiny smoke tests, add ``--synthetic``.

Requires: pip install -r requirements.txt (torch is pulled in by sglang on most platforms).

CPU path uses ``device="cpu"`` and ``attention_backend="torch_native"`` when supported.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _maybe_apply_torch_rope_cuda_patch(device: str) -> None:
    """Same RoPE fallback as venv ``sitecustomize`` (for runs without worker spawn)."""
    if device != "cuda":
        return

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import rope_torch_fallback

    rope_torch_fallback.apply()


def build_prompts(batch_size: int) -> list[str]:
    return [
        f"Question {i}: Summarize how offline batch inference differs from HTTP servers, in one short paragraph."
        for i in range(batch_size)
    ]


def load_prompts_from_file(path: str) -> list[str]:
    """
    Load prompts from JSONL (one object per line with a ``prompt`` field) or plain text
    (one non-empty line per prompt).
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Prompts file not found: {p}")
    raw = p.read_text(encoding="utf-8").splitlines()
    prompts: list[str] = []
    for line in raw:
        s = line.strip()
        if not s:
            continue
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                prompts.append(s)
                continue
            pr = obj.get("prompt")
            if isinstance(pr, str) and pr.strip():
                prompts.append(pr.strip())
            else:
                raise ValueError(f"JSONL line missing non-empty 'prompt' string: {s[:120]}...")
        else:
            prompts.append(s)
    if not prompts:
        raise ValueError(f"No prompts loaded from {p}")
    return prompts


def resolve_device(arg: str) -> str:
    if arg in ("cpu", "cuda"):
        return arg
    if arg == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    raise ValueError(f"Unknown device mode: {arg}")


def _maybe_reexec_for_cpu(device: str) -> None:
    """
    Re-exec once with CPU-engine env for SGLang.

    SGLang's CPU backend detection requires SGLANG_USE_CPU_ENGINE=1, and many ops
    dispatch by checking CUDA availability at import time. Hiding CUDA devices in the
    child process avoids accidentally taking CUDA-only kernels on CPU tensors.
    """
    if device != "cpu":
        return
    if os.environ.get("SGLANG_CPU_REEXEC_DONE") == "1":
        return

    env = os.environ.copy()
    env["SGLANG_CPU_REEXEC_DONE"] = "1"
    env["SGLANG_USE_CPU_ENGINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    # Disable project startup hook in CPU-only re-exec; it is for CUDA RoPE fallback.
    env.pop("SGLANG_EXPLORATION_ROOT", None)

    python_bin = shutil.which("python") or sys.executable
    os.execvpe(python_bin, [python_bin, *sys.argv], env)


def engine_kwargs_for_device(
    device: str,
    model: str,
    trust_remote_code: bool,
    mem_fraction_static: float | None,
    enable_cuda_graph: bool,
) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "model_path": model,
        "trust_remote_code": trust_remote_code,
    }
    if device == "cpu":
        kw["device"] = "cpu"
        kw["attention_backend"] = "torch_native"
    else:
        if mem_fraction_static is not None:
            kw["mem_fraction_static"] = mem_fraction_static
        # CUDA graph capture uses Triton/ptxas; Blackwell-class GPUs (e.g. sm_110a) may hit
        # "gpu-name is not defined" with the bundled ptxas unless graphs are disabled.
        if not enable_cuda_graph:
            kw["disable_cuda_graph"] = True
        # On aarch64 + very new SMs, FlashInfer/Triton kernels may call ptxas without sm_110a support.
        # Pure PyTorch attention avoids that path (slower but portable on Jetson Thor until toolchains catch up).
        if platform.machine() == "aarch64" and os.environ.get("SGLANG_FORCE_FLASHINFER_AARCH64") != "1":
            kw.setdefault("attention_backend", "torch_native")
    return kw


def _default_queries_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "queries_512.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGLang offline Engine batch experiment (CPU/GPU). "
        "Default: 512 prompts from data/queries_512.jsonl; --batch-size = prompts per generate().",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HF model id or local path")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prompts per generate() call (try 8, 16, 32, …). 0 = one generate() over the full prompt list.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Ignore default prompt file; use short built-in prompts (good for smoke tests).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=8,
        dest="num_prompts",
        help="With --synthetic: how many prompts to build (default 8).",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="",
        help="Override prompt file (JSONL with 'prompt' or plain lines). Not used with --synthetic.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help="Optional GPU memory fraction (e.g. 0.85 on tight VRAM). CPU ignores this.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=False to Engine (default is True)",
    )
    parser.add_argument("--output", type=str, default="", help="Optional path to write result JSON")
    parser.add_argument("--print-sample", action="store_true", help="Print first generated sample")
    parser.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        help="Allow CUDA graph capture on GPU (default: off — avoids Triton/ptxas issues on some Jetson Thor builds).",
    )
    args = parser.parse_args()

    trust_remote_code = not args.no_trust_remote_code

    device = resolve_device(args.device)
    _maybe_reexec_for_cpu(device)

    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                print("CUDA requested but torch.cuda.is_available() is False.", file=sys.stderr)
                sys.exit(2)
        except ImportError as e:
            print("CUDA requested but PyTorch is not installed.", file=sys.stderr)
            raise SystemExit(2) from e

    try:
        import sglang as sgl
    except ImportError as e:
        print(
            "sglang is not installed. Create a venv and run:\n"
            "  pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    _maybe_apply_torch_rope_cuda_patch(device)

    default_q = _default_queries_path()
    prompts_path_used: str | None = None
    if args.synthetic:
        n = max(1, args.num_prompts)
        prompts = build_prompts(n)
    elif args.prompts_file.strip():
        prompts_path_used = os.path.abspath(args.prompts_file.strip())
        prompts = load_prompts_from_file(prompts_path_used)
    elif default_q.is_file():
        prompts_path_used = str(default_q.resolve())
        prompts = load_prompts_from_file(prompts_path_used)
    else:
        print(
            "No prompts found: expected "
            f"{default_q} next to this script, or pass --prompts-file PATH, or --synthetic.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    num_prompts = len(prompts)
    ekw = engine_kwargs_for_device(
        device,
        args.model,
        trust_remote_code=trust_remote_code,
        mem_fraction_static=args.mem_fraction_static,
        enable_cuda_graph=args.enable_cuda_graph,
    )

    sampling_params: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
    }

    init_t0 = time.time()
    llm = sgl.Engine(**ekw)
    init_time = time.time() - init_t0

    gen_bs = 0 if args.batch_size <= 0 else args.batch_size
    try:
        t0 = time.time()
        outputs: list[Any] = []
        if gen_bs > 0:
            for off in range(0, num_prompts, gen_bs):
                outputs.extend(llm.generate(prompts[off : off + gen_bs], sampling_params))
        else:
            outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - t0
    finally:
        llm.shutdown()

    if args.print_sample and outputs:
        first = outputs[0]
        text = first.get("text", first) if isinstance(first, dict) else str(first)
        print("--- sample[0] ---", flush=True)
        print(text[:800], flush=True)

    result = {
        "engine": "sgl.Engine offline batch",
        "model": args.model,
        "device": device,
        "num_prompts": num_prompts,
        "batch_size": gen_bs,
        "synthetic": bool(args.synthetic),
        "prompts_file": prompts_path_used,
        "max_new_tokens": args.max_new_tokens,
        "engine_kwargs": {k: v for k, v in ekw.items() if k != "model_path"},
        "cuda_graph_enabled": bool(args.enable_cuda_graph) if device == "cuda" else None,
        "init_time_s": round(init_time, 3),
        "time_s": round(elapsed, 3),
        "per_prompt_ms": round((elapsed / num_prompts) * 1000, 3) if num_prompts else None,
        "throughput_prompts_per_s": round(num_prompts / elapsed, 3) if elapsed > 0 and num_prompts else None,
    }
    line = f"__RESULT_JSON__:{json.dumps(result)}"
    print(line, flush=True)
    if args.output.strip():
        out_path = os.path.abspath(args.output.strip())
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
