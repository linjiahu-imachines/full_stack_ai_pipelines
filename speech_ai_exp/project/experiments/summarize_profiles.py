#!/usr/bin/env python3
"""Aggregate staged-voice JSONL profiles (one JSON object per line)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", type=Path, help="profiles/runs.jsonl")
    args = p.parse_args()
    lines = args.jsonl.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    if not rows:
        print("No rows", file=sys.stderr)
        return 1

    keys = [
        "asr_s",
        "asr_rtf",
        "llm_ttft_s",
        "llm_generation_s",
        "llm_tokens_per_s_est",
        "tts_s",
        "audio_duration_s",
    ]
    print(f"n={len(rows)}")
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r]
        if not vals:
            continue
        print(
            f"{k:26s}  mean={statistics.mean(vals):.4f}  "
            f"p50={statistics.median(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
