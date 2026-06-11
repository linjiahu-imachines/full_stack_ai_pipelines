from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    omni_backend: str = "moshi"  # moshi | mini_omni

    moshi_hf_repo: str = "kyutai/moshiko-pytorch-bf16"
    moshi_device: str = "auto"
    moshi_temp: float = 0.8
    moshi_temp_text: float = 0.7
    moshi_max_frames: int | None = None  # cap input frames only; None = all input
    moshi_input_silence_s: float = 2.0  # silence appended to input (turn end cue)
    moshi_response_s: float = 3.0  # how long to generate reply after user stops

    sidecar_whisper: bool = True
    whisper_model: str = "small"
    whisper_device: str = "cpu"

    mini_omni_ckpt_dir: Path | None = None
    mini_omni_mode: str = "a1t2_t1a2"  # a1t2_t1a2 | a1_a2
    mini_omni_speech_align: str = "espeak"  # espeak (WAV matches script) | model (T1_A2)
    mini_omni_tts_voice: str = "en-us"
    mini_omni_tts_rate_wpm: int = 165

    audio_out_dir: Path = Path("audio/out")


def overlay_from_yaml_dict(cfg: RunConfig, data: dict[str, Any]) -> None:
    if not data:
        return
    if "omni" in data:
        o = data["omni"]
        cfg.omni_backend = o.get("backend", cfg.omni_backend)
    if "moshi" in data:
        m = data["moshi"]
        cfg.moshi_hf_repo = m.get("hf_repo", cfg.moshi_hf_repo)
        cfg.moshi_device = m.get("device", cfg.moshi_device)
        cfg.moshi_temp = float(m.get("temp", cfg.moshi_temp))
        cfg.moshi_temp_text = float(m.get("temp_text", cfg.moshi_temp_text))
        if m.get("max_frames") is not None:
            cfg.moshi_max_frames = int(m["max_frames"])
        if m.get("input_silence_s") is not None:
            cfg.moshi_input_silence_s = float(m["input_silence_s"])
        if m.get("response_s") is not None:
            cfg.moshi_response_s = float(m["response_s"])
    if "sidecar" in data:
        s = data["sidecar"]
        cfg.sidecar_whisper = bool(s.get("whisper", cfg.sidecar_whisper))
        cfg.whisper_model = s.get("model", cfg.whisper_model)
        cfg.whisper_device = s.get("device", cfg.whisper_device)
    if "mini_omni" in data:
        m = data["mini_omni"]
        if m.get("ckpt_dir"):
            cfg.mini_omni_ckpt_dir = Path(m["ckpt_dir"])
        cfg.mini_omni_mode = m.get("mode", cfg.mini_omni_mode)
        cfg.mini_omni_speech_align = m.get("speech_align", cfg.mini_omni_speech_align)
        cfg.mini_omni_tts_voice = m.get("tts_voice", cfg.mini_omni_tts_voice)
        if m.get("tts_rate_wpm") is not None:
            cfg.mini_omni_tts_rate_wpm = int(m["tts_rate_wpm"])
