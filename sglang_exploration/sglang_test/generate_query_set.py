#!/usr/bin/env python3
"""Write a JSONL query set for offline batch experiments (one JSON object per line)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _prompt_for_index(i: int) -> str:
    """Deterministic, varied short prompts suitable for throughput / batching tests."""
    verbs = (
        "Summarize",
        "Explain",
        "List three facts about",
        "Give a one-sentence definition of",
        "What is a common misconception about",
        "Name one application of",
        "Compare briefly:",
        "Translate this idea to a beginner:",
    )
    domains = (
        "offline LLM batching",
        "continuous batching in inference servers",
        "prefix caching and KV reuse",
        "radix trees for attention",
        "GPU memory bandwidth vs compute",
        "CPU vs GPU inference tradeoffs",
        "quantization effects on latency",
        "speculative decoding",
        "long-context retrieval",
        "RAG pipeline bottlenecks",
        "tokenizer edge cases",
        "temperature vs greedy decoding",
        "throughput vs latency tuning",
        "Jetson-class edge deployment",
        "power limits and thermal throttling",
    )
    tails = (
        "Answer in at most two sentences.",
        "Keep the answer under 60 words.",
        "Be concise.",
        "Use plain language.",
        "No bullet points.",
    )
    v = verbs[i % len(verbs)]
    d = domains[(i // len(verbs)) % len(domains)]
    t = tails[(i // (len(verbs) * len(domains))) % len(tails)]
    return f"{i + 1}. {v} {d}. {t}"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate JSONL prompts for SGLang batch experiments")
    p.add_argument("--count", type=int, default=512, help="Number of queries to emit")
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output path (default: stdout)",
    )
    args = p.parse_args()
    if args.count < 1:
        print("--count must be >= 1", file=sys.stderr)
        raise SystemExit(2)

    lines = []
    for i in range(args.count):
        rec = {"id": i, "prompt": _prompt_for_index(i)}
        lines.append(json.dumps(rec, ensure_ascii=False) + "\n")

    text = "".join(lines)
    if args.output.strip():
        out = Path(args.output.strip()).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.count} lines to {out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
