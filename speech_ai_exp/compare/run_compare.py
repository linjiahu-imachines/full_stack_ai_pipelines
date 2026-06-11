#!/usr/bin/env python3
"""Compare Project 1 (staged) vs Project 2 (omni) profile JSON files — no package imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _staged_e2e(p: dict) -> float:
    if p.get("e2e_wall_s") is not None:
        return float(p["e2e_wall_s"])
    return float(p.get("asr_s", 0)) + float(p.get("llm_generation_s", 0)) + float(p.get("tts_s", 0))


def _staged_ttfa(p: dict) -> float:
    if p.get("ttfa_s") is not None:
        return float(p["ttfa_s"])
    return float(p.get("asr_s", 0)) + float(p.get("llm_ttft_s", 0))


def _normalize_staged(p: dict) -> dict:
    return {
        "stack": "staged",
        "backend": ((p.get("meta") or {}).get("llm") or {}).get("backend", "staged"),
        "audio_path": p.get("audio_path", ""),
        "audio_duration_s": p.get("audio_duration_s"),
        "e2e_wall_s": _staged_e2e(p),
        "ttfa_s": _staged_ttfa(p),
        "output_wav_path": p.get("output_wav_path"),
        "transcript": p.get("transcript", ""),
        "reply_text": p.get("reply_text", ""),
        "inner_text": "",
        "asr_s": p.get("asr_s"),
        "llm_ttft_s": p.get("llm_ttft_s"),
        "llm_generation_s": p.get("llm_generation_s"),
        "tts_s": p.get("tts_s"),
    }


def _normalize_omni(p: dict) -> dict:
    return {
        "stack": p.get("stack", "voice_to_voice"),
        "backend": p.get("backend", "omni"),
        "audio_path": p.get("audio_path", ""),
        "audio_duration_s": p.get("audio_duration_s"),
        "e2e_wall_s": float(p.get("e2e_wall_s", 0)),
        "ttfa_s": float(p.get("ttfa_s", 0)),
        "output_wav_path": p.get("output_wav_path"),
        "transcript": p.get("transcript", ""),
        "reply_text": p.get("reply_text", ""),
        "inner_text": p.get("inner_text", ""),
        "meta": p.get("meta", {}),
    }


def _md_table(staged: dict, omni: dict) -> str:
    rows = [
        ("Stack", staged["stack"], omni["stack"]),
        ("Backend", str(staged.get("backend", "")), str(omni.get("backend", ""))),
        ("Input audio", staged["audio_path"], omni["audio_path"]),
        ("Audio duration (s)", _fmt(staged.get("audio_duration_s")), _fmt(omni.get("audio_duration_s"))),
        ("E2E wall (s)", _fmt(staged["e2e_wall_s"]), _fmt(omni["e2e_wall_s"])),
        ("TTFA (s)", _fmt(staged["ttfa_s"]), _fmt(omni["ttfa_s"])),
        ("Output WAV", str(staged.get("output_wav_path", "")), str(omni.get("output_wav_path", ""))),
        ("User transcript", _short(staged.get("transcript", "")), _short(omni.get("transcript", ""))),
        ("Assistant text", _short(staged.get("reply_text", "")), _short(omni.get("reply_text", ""))),
    ]
    if omni.get("inner_text"):
        rows.append(("Omni inner text", "—", _short(omni.get("inner_text", ""))))

    if staged.get("asr_s") is not None:
        rows.extend(
            [
                ("ASR (s)", _fmt(staged.get("asr_s")), "—"),
                ("LLM TTFT (s)", _fmt(staged.get("llm_ttft_s")), "—"),
                ("LLM total (s)", _fmt(staged.get("llm_generation_s")), "—"),
                ("TTS (s)", _fmt(staged.get("tts_s")), "—"),
            ]
        )

    lines = [
        "# Staged (Project 1) vs Voice-to-voice (Project 2)",
        "",
        "| Metric | Project 1 (staged) | Project 2 (omni) |",
        "|--------|-------------------|------------------|",
    ]
    for label, a, b in rows:
        lines.append(f"| {label} | {a} | {b} |")
    lines.append("")
    delta_e2e = omni["e2e_wall_s"] - staged["e2e_wall_s"]
    lines.append(f"**Δ E2E (P2 − P1):** {delta_e2e:+.3f} s")
    lines.append("")
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _short(s: str, n: int = 80) -> str:
    s = (s or "").strip().replace("|", "\\|").replace("\n", " ")
    if len(s) <= n:
        return s or "—"
    return s[: n - 3] + "..."


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--staged", type=Path, required=True, help="Project 1 profile JSON")
    p.add_argument("--omni", type=Path, required=True, help="Project 2 profile JSON")
    p.add_argument("--out", type=Path, required=True, help="Markdown report path")
    args = p.parse_args(argv)

    staged_raw = _load(args.staged.expanduser().resolve())
    omni_raw = _load(args.omni.expanduser().resolve())
    staged = _normalize_staged(staged_raw)
    omni = _normalize_omni(omni_raw)

    report = _md_table(staged, omni)
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"  P1 E2E: {staged['e2e_wall_s']:.3f}s  P2 E2E: {omni['e2e_wall_s']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
