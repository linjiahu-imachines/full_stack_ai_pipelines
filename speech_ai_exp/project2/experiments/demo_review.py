#!/usr/bin/env python3
"""Display voice-to-voice profile and play WAVs (terminal + optional browser)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _play_wav(path: Path) -> bool:
    players = [
        ["pw-play", str(path)],
        ["paplay", str(path)],
        ["aplay", "-q", str(path)],
        ["ffplay", "-nodisp", "-autoexit", str(path)],
    ]
    for cmd in players:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  Played via: {cmd[0]}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return False


def _moshi_qa_warning(data: dict) -> str | None:
    if data.get("backend") != "moshi":
        return None
    t = (data.get("transcript") or "").lower()
    if not any(k in t for k in ("what", "how", "plus", "+", "equals", "?")):
        return None
    inner = (data.get("inner_text") or "").lower()
    if any(w in inner for w in ("how", "day", "hey", "hello", "going")):
        return (
            "Note: Moshi is duplex chitchat, not Q&A. For “2+3 → 5” use Project 1 "
            "or: omni-voice-run --backend mini_omni (see docs/PROJECT2_SEMANTIC_RELEVANCE.md)."
        )
    return None


def _print_turn(data: dict) -> None:
    w = 72
    print("=" * w)
    print("VOICE-TO-VOICE DEMO — Project 2")
    print("=" * w)
    warn = _moshi_qa_warning(data)
    if warn:
        print("!" * w)
        print(warn)
        print("!" * w)
    print(f"Backend:      {data.get('backend', '?')} ({data.get('stack', '?')})")
    print(f"Input audio:  {data.get('audio_path', '?')}")
    print(f"Output audio: {data.get('output_wav_path', '?')}")
    print("-" * w)
    transcript = (data.get("transcript") or "").strip()
    print("USER (input transcript — sidecar Whisper):")
    print(f"  {transcript or '(missing — re-run with: omni-voice-run --sidecar-whisper)'}")
    print("-" * w)
    meta = data.get("meta") or {}
    backend = data.get("backend", "")
    speech_src = meta.get("speech_source", "")
    if backend == "moshi":
        listen = (meta.get("listen_inner_text") or "").strip()
        if listen:
            print("MOSHI (during your speech — duplex):")
            print(f"  {listen[:500]}")
            print("-" * w)
    full = (meta.get("mini_omni_answer_text_full") or "").strip()
    if full and full != (data.get("inner_text") or "").strip():
        print("MODEL (full text answer):")
        print(f"  {full[:500]}")
        print("-" * w)
    inner = (data.get("inner_text") or "").strip()
    if inner:
        label = "REPLY (spoken script"
        if speech_src == "espeak_aligned":
            label += " — matches audio via eSpeak"
        elif speech_src == "mini_omni_t1_a2":
            label += " — Mini-Omni T1_A2"
        label += "):"
        print(label)
        print(f"  {inner[:500]}")
        print("-" * w)
    reply = (data.get("reply_text") or "").strip()
    if reply and reply != inner:
        print("REPLY (sidecar Whisper on WAV):")
        print(f"  {reply[:500]}")
        print("-" * w)
    print("-" * w)
    print(
        f"Timings (s): E2E wall={data.get('e2e_wall_s', 0):.3f}  "
        f"TTFA={data.get('ttfa_s', 0):.3f}"
    )
    if meta.get("inference_s") is not None:
        print(f"             inference={meta.get('inference_s', 0):.3f}")
    print("=" * w)


def _write_html(profile: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    in_name = "input.wav"
    out_name = "reply.wav"
    shutil.copy2(Path(profile["audio_path"]), out_dir / in_name)
    out_path = profile.get("output_wav_path")
    if out_path:
        shutil.copy2(Path(out_path), out_dir / out_name)

    transcript = _escape_html((profile.get("transcript") or "").strip())
    meta = profile.get("meta") or {}
    listen = _escape_html((meta.get("listen_inner_text") or "").strip())
    full = _escape_html((meta.get("mini_omni_answer_text_full") or "").strip())
    inner = _escape_html((profile.get("inner_text") or "").strip())
    reply = inner or _escape_html((profile.get("reply_text") or "").strip())
    backend = _escape_html(profile.get("backend", ""))
    speech_src = _escape_html(meta.get("speech_source", ""))
    align_note = ""
    if speech_src == "espeak_aligned":
        align_note = "<p><em>Speech is synthesized from the reply script (eSpeak) so audio matches text.</em></p>"
    elif backend == "mini_omni":
        align_note = "<p><em>Mini-Omni: text from audio understanding; speech from aligned TTS when configured.</em></p>"
    warn = _moshi_qa_warning(profile)
    warn_html = ""
    if warn:
        warn_html = f'<p style="background:#fef3c7;padding:0.75rem;border-radius:6px;">{_escape_html(warn)}</p>'
    if not transcript:
        transcript = "(Re-run omni-voice-run with --sidecar-whisper)"
    if not reply:
        reply = "(Sidecar ASR on output WAV — enable --sidecar-whisper)"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Omni voice demo</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 52rem; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ font-size: 1.25rem; }}
    section {{ margin: 1.5rem 0; padding: 1rem; background: #f4f4f5; border-radius: 8px; }}
    .label {{ font-weight: 600; color: #444; margin-bottom: 0.5rem; }}
    .text {{ font-size: 1.1rem; line-height: 1.5; white-space: pre-wrap; }}
    audio {{ width: 100%; margin-top: 0.5rem; }}
    table {{ border-collapse: collapse; font-size: 0.9rem; }}
    td, th {{ padding: 0.35rem 0.75rem; text-align: left; border-bottom: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>Project 2 — Voice-to-voice ({backend})</h1>
  {warn_html}
  {align_note}
  <section>
    <div class="label">User speech (input)</div>
    <div class="text">{transcript}</div>
    <audio controls src="{in_name}"></audio>
  </section>
  <section>
    <div class="label">Model full answer (optional)</div>
    <div class="text">{full or "—"}</div>
  </section>
  <section>
    <div class="label">Reply script (spoken; matches audio when eSpeak-aligned)</div>
    <div class="text">{inner or "—"}</div>
    <audio controls src="{out_name}"></audio>
  </section>
  <section>
    <div class="label">Timings (seconds)</div>
    <table>
      <tr><th>E2E wall</th><td>{profile.get("e2e_wall_s", 0):.3f}</td></tr>
      <tr><th>TTFA</th><td>{profile.get("ttfa_s", 0):.3f}</td></tr>
    </table>
  </section>
</body>
</html>
"""
    index = out_dir / "index.html"
    index.write_text(html, encoding="utf-8")
    return index


def _serve_dir(directory: Path, port: int) -> None:
    directory = directory.resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Open in browser: {url}")
    print("(Ctrl+C to stop server)")
    Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile",
        type=Path,
        default=root / "profiles" / "latest.json",
        help="Profile JSON from omni-voice-run",
    )
    p.add_argument("--play-input", action="store_true")
    p.add_argument("--play-output", action="store_true")
    p.add_argument("--play-all", action="store_true")
    p.add_argument("--html-dir", type=Path, default=root / "demo" / "latest")
    p.add_argument("--serve", type=int, metavar="PORT", nargs="?", const=8766)
    args = p.parse_args(argv)

    profile_path = args.profile if args.profile.is_absolute() else (Path.cwd() / args.profile)
    if not profile_path.is_file():
        print(f"Profile not found: {profile_path}", file=sys.stderr)
        print("Run omni-voice-run first, then pass --profile profiles/your_run.json", file=sys.stderr)
        return 2

    data = json.loads(profile_path.read_text(encoding="utf-8"))
    _print_turn(data)

    if args.play_input or args.play_all:
        print("\n▶ User input audio:")
        if not _play_wav(Path(data["audio_path"])):
            print("  Could not play. Try: pw-play", data["audio_path"])
    if args.play_output or args.play_all:
        out = data.get("output_wav_path")
        if out:
            print("\n▶ Model reply audio:")
            if not _play_wav(Path(out)):
                print("  Could not play. Try: pw-play", out)

    html_dir = args.html_dir if args.html_dir.is_absolute() else (Path.cwd() / args.html_dir)
    index = _write_html(data, html_dir)
    print(f"\nBrowser demo page: {index}")

    if args.serve is not None:
        _serve_dir(html_dir, int(args.serve))
    else:
        print("\nFor browser playback, run:")
        print(f"  python3 experiments/demo_review.py --profile {profile_path} --serve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
