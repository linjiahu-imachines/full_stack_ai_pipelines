"""Command-line entrypoint for `staged-voice-run` and dev launchers."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(
            "PyYAML required for --config. Install with: pip install 'staged-voice[experiment]'"
        ) from e
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def main(argv: list[str] | None = None) -> int:
    from staged_voice.backends import FasterWhisperASR, OllamaLLM
    from staged_voice.backends.tts_factory import make_tts
    from staged_voice.config import RunConfig, overlay_from_yaml_dict
    from staged_voice.pipeline import StagedVoicePipeline
    from staged_voice.profiling import write_profile_json

    desc = "Run one ASR → LLM → TTS turn on a WAV file and emit a JSON profile."
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--audio", type=Path, required=True, help="Input WAV path")
    p.add_argument("--profile-json", type=Path, default=Path("profiles/latest.json"))
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML (see configs/example_ollama.yaml)",
    )

    p.add_argument("--asr-backend", choices=["faster_whisper"], default="faster_whisper")
    p.add_argument("--asr-model-size", default=None, dest="whisper_model_size")
    p.add_argument("--whisper-device", default=None)
    p.add_argument("--whisper-compute-type", default=None)
    p.add_argument("--whisper-language", default=None, help="e.g. en, or omit for auto")

    p.add_argument("--llm-backend", choices=["ollama", "hf"], default=None)
    p.add_argument("--ollama-host", default=None)
    p.add_argument("--ollama-model", default=None)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--max-tokens", type=int, default=None, help="Ollama num_predict cap")
    p.add_argument("--temperature", type=float, default=None)

    p.add_argument("--hf-model", default=None, dest="hf_model_name")
    p.add_argument("--hf-device", default=None)
    p.add_argument("--hf-max-new-tokens", type=int, default=None)
    p.add_argument("--hf-torch-dtype", default=None, choices=["auto", "bf16", "fp16", "fp32"])

    p.add_argument(
        "--tts-backend",
        choices=["espeak", "kokoro"],
        default=None,
        help="TTS engine: espeak (default) or kokoro (Kokoro-82M neural)",
    )
    p.add_argument("--tts-voice", default=None, help="espeak voice, or ignored when kokoro")
    p.add_argument("--tts-rate-wpm", type=int, default=None, help="espeak words/min only")
    p.add_argument(
        "--kokoro-lang",
        default=None,
        dest="kokoro_lang_code",
        help="Kokoro lang_code (a=American English, b=British, ...)",
    )
    p.add_argument("--kokoro-voice", default=None, help="Kokoro voice id, e.g. af_heart")
    p.add_argument("--kokoro-speed", type=float, default=None)
    p.add_argument("--audio-out-dir", type=Path, default=None)

    args = p.parse_args(argv)

    cfg = RunConfig()
    audio_path = args.audio.expanduser().resolve()

    if args.config is not None:
        overlay_from_yaml_dict(cfg, _load_yaml(args.config.expanduser().resolve()))

    if args.whisper_model_size:
        cfg.whisper_model_size = args.whisper_model_size
    if args.whisper_device:
        cfg.whisper_device = args.whisper_device
    if args.whisper_compute_type:
        cfg.whisper_compute_type = args.whisper_compute_type
    if args.whisper_language:
        cfg.whisper_language = args.whisper_language

    if args.llm_backend:
        cfg.llm_backend = args.llm_backend
    if args.ollama_host:
        cfg.ollama_host = args.ollama_host
    if args.ollama_model:
        cfg.ollama_model = args.ollama_model
    if args.system_prompt:
        cfg.system_prompt = args.system_prompt
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.hf_model_name:
        cfg.hf_model_name = args.hf_model_name
        if args.llm_backend is None:
            cfg.llm_backend = "hf"
    if args.hf_device:
        cfg.hf_device = args.hf_device
    if args.hf_max_new_tokens is not None:
        cfg.hf_max_new_tokens = args.hf_max_new_tokens
    if args.hf_torch_dtype:
        cfg.hf_torch_dtype = args.hf_torch_dtype
    if args.tts_backend:
        cfg.tts_backend = args.tts_backend
    if args.tts_voice:
        cfg.tts_voice = args.tts_voice
    if args.tts_rate_wpm is not None:
        cfg.tts_rate_wpm = args.tts_rate_wpm
    if args.kokoro_lang_code:
        cfg.kokoro_lang_code = args.kokoro_lang_code
    if args.kokoro_voice:
        cfg.kokoro_voice = args.kokoro_voice
    if args.kokoro_speed is not None:
        cfg.kokoro_speed = args.kokoro_speed
    if args.audio_out_dir is not None:
        cfg.audio_out_dir = args.audio_out_dir

    cwd = Path.cwd()
    cfg.audio_out_dir = Path(cfg.audio_out_dir)
    if not cfg.audio_out_dir.is_absolute():
        cfg.audio_out_dir = (cwd / cfg.audio_out_dir).resolve()

    if not audio_path.is_file():
        print(f"Audio not found: {audio_path}", file=sys.stderr)
        return 2

    if cfg.llm_backend == "ollama":
        llm = OllamaLLM(cfg.ollama_host, cfg.ollama_model)
    elif cfg.llm_backend == "hf":
        from staged_voice.backends.llm_hf_causal import HFCausalLM

        llm = HFCausalLM(
            cfg.hf_model_name,
            device=cfg.hf_device,
            torch_dtype=cfg.hf_torch_dtype,
        )
    else:
        print(f"Unknown llm backend: {cfg.llm_backend}", file=sys.stderr)
        return 2

    try:
        asr = FasterWhisperASR(
            model_size=cfg.whisper_model_size,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
            language=cfg.whisper_language,
        )
        tts = make_tts(cfg)
    except (EnvironmentError, ImportError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 3

    pipe = StagedVoicePipeline(cfg, asr, llm, tts)
    try:
        profile = pipe.run_turn(audio_path)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip()
        print(f"TTS subprocess failed (is espeak-ng installed?): {err}", file=sys.stderr)
        return 4
    except Exception as e:
        print(f"TTS failed: {e}", file=sys.stderr)
        return 4

    profile_path = args.profile_json
    if not profile_path.is_absolute():
        profile_path = (cwd / profile_path).resolve()
    write_profile_json(profile, profile_path)

    print(f"Wrote profile: {profile_path}")
    print(f"Reply WAV:       {profile.output_wav_path}")
    return 0
