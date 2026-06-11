"""CLI entrypoint: omni-voice-run"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("PyYAML required: pip install -e '.[experiment]'") from e
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def main(argv: list[str] | None = None) -> int:
    from omni_voice.backends.backend_factory import make_backend
    from omni_voice.config import RunConfig, overlay_from_yaml_dict
    from omni_voice.pipeline import OmniVoicePipeline
    from omni_voice.profiling import write_profile_json

    p = argparse.ArgumentParser(description="Run one voice-to-voice turn (Project 2).")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--profile-json", type=Path, default=Path("profiles/latest.json"))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--backend", choices=["moshi", "mini_omni"], default=None)
    p.add_argument("--moshi-hf-repo", default=None)
    p.add_argument("--moshi-device", default=None, choices=["auto", "cuda", "cpu"])
    p.add_argument("--moshi-max-frames", type=int, default=None)
    p.add_argument("--audio-out-dir", type=Path, default=None)
    p.add_argument(
        "--sidecar-whisper",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run Whisper on input/output for transcript in profile (default: on)",
    )
    args = p.parse_args(argv)

    cfg = RunConfig()
    if args.config:
        overlay_from_yaml_dict(cfg, _load_yaml(args.config.expanduser().resolve()))
    if args.backend:
        cfg.omni_backend = args.backend
    if args.moshi_hf_repo:
        cfg.moshi_hf_repo = args.moshi_hf_repo
    if args.moshi_device:
        cfg.moshi_device = args.moshi_device
    if args.moshi_max_frames is not None:
        cfg.moshi_max_frames = args.moshi_max_frames
    if args.audio_out_dir is not None:
        cfg.audio_out_dir = args.audio_out_dir
    if args.sidecar_whisper is not None:
        cfg.sidecar_whisper = args.sidecar_whisper

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        print(f"Audio not found: {audio_path}", file=sys.stderr)
        return 2

    try:
        backend = make_backend(cfg)
    except (ImportError, ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 3

    pipe = OmniVoicePipeline(cfg, backend)
    try:
        profile = pipe.run_turn(audio_path)
    except Exception as e:
        print(f"Omni run failed: {e}", file=sys.stderr)
        return 4

    profile_path = args.profile_json
    if not profile_path.is_absolute():
        profile_path = (Path.cwd() / profile_path).resolve()
    write_profile_json(profile, profile_path)
    print(f"Wrote profile: {profile_path}")
    print(f"Reply WAV:       {profile.output_wav_path}")
    if profile.transcript:
        print(f"Input transcript (Whisper): {profile.transcript}")
    if profile.inner_text:
        print(f"Model reply text:         {profile.inner_text[:300]}")
    if profile.reply_text:
        print(f"Output transcript (Whisper): {profile.reply_text[:300]}")
    return 0
