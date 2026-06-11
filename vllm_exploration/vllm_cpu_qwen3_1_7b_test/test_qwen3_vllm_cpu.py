"""
Standalone vLLM CPU test for Qwen/Qwen3-1.7B.
This script is intended to be called from a file-based entrypoint.

Must set ``VLLM_TARGET_DEVICE=cpu`` before importing vLLM (vLLM reads env at import time).
"""

import argparse
import json
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Must be set before any `import vllm`. Note: CUDA wheels still select GPU on
# machines with a GPU; CPU inference requires a vLLM build whose version
# string contains "cpu" (see vllm.platforms).
os.environ["VLLM_TARGET_DEVICE"] = "cpu"


def build_prompts(batch_size: int) -> list[str]:
    return [f"Question {i}: Explain CPU inference trade-offs in one paragraph." for i in range(batch_size)]


def load_prompts_from_file(path: str) -> list[str]:
    """Load prompts from JSONL (field 'prompt') or plain text (one prompt per line)."""
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("{") and s.endswith("}"):
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    out.append(s)
                    continue
                prompt = obj.get("prompt")
                if isinstance(prompt, str) and prompt.strip():
                    out.append(prompt.strip())
                else:
                    raise ValueError(f"JSONL line missing non-empty 'prompt': {s[:120]}...")
            else:
                out.append(s)
    if not out:
        raise ValueError(f"No prompts loaded from {path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM CPU test for Qwen3-1.7B")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="",
        help="Optional JSONL/text prompts file. If set, prompts are loaded from file.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=0,
        help="When using --prompts-file, optional cap on how many prompts to use (0 = all).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write the result JSON (same object as __RESULT_JSON__).",
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    if args.prompts_file.strip():
        prompts = load_prompts_from_file(os.path.abspath(args.prompts_file.strip()))
        if args.num_prompts > 0:
            prompts = prompts[: args.num_prompts]
        if not prompts:
            raise ValueError("No prompts selected after applying --num-prompts")
    else:
        prompts = build_prompts(args.batch_size)
    num_prompts = len(prompts)

    init_t0 = time.time()
    llm = LLM(
        model=args.model,
        max_num_seqs=max(1, min(args.batch_size, args.max_num_seqs)),
        enforce_eager=True,
    )
    init_time = time.time() - init_t0

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    t0 = time.time()
    if args.batch_size > 0:
        for off in range(0, num_prompts, args.batch_size):
            _ = llm.generate(prompts[off : off + args.batch_size], sampling_params)
    else:
        _ = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    result = {
        "engine": "qwen3-1.7B with vLLM",
        "model": args.model,
        "batch_size": args.batch_size,
        "num_prompts": num_prompts,
        "prompts_file": os.path.abspath(args.prompts_file.strip()) if args.prompts_file.strip() else None,
        "max_new_tokens": args.max_new_tokens,
        "max_num_seqs": max(1, min(args.batch_size, args.max_num_seqs)),
        "init_time_s": round(init_time, 2),
        "time_s": round(elapsed, 2),
        "per_prompt_ms": round((elapsed / num_prompts) * 1000, 2),
        "throughput_prompts_per_s": round(num_prompts / elapsed, 2),
    }
    print(f"__RESULT_JSON__:{json.dumps(result)}", flush=True)
    if args.output.strip():
        out_path = os.path.abspath(args.output.strip())
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
