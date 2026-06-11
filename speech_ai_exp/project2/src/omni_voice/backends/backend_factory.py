from __future__ import annotations

from omni_voice.backends.base_types import OmniBackend
from omni_voice.backends.mini_omni_backend import MiniOmniBackend
from omni_voice.backends.moshi_backend import MoshiBackend
from omni_voice.config import RunConfig


def make_backend(cfg: RunConfig) -> OmniBackend:
    name = cfg.omni_backend.lower()
    if name == "moshi":
        return MoshiBackend(
            hf_repo=cfg.moshi_hf_repo,
            device=cfg.moshi_device,
            temp=cfg.moshi_temp,
            temp_text=cfg.moshi_temp_text,
            max_frames=cfg.moshi_max_frames,
            input_silence_s=cfg.moshi_input_silence_s,
            response_s=cfg.moshi_response_s,
        )
    if name == "mini_omni":
        return MiniOmniBackend(
            ckpt_dir=cfg.mini_omni_ckpt_dir,
            device=cfg.moshi_device,
            mode=cfg.mini_omni_mode,
            speech_align=cfg.mini_omni_speech_align,
            tts_voice=cfg.mini_omni_tts_voice,
            tts_rate_wpm=cfg.mini_omni_tts_rate_wpm,
        )
    raise ValueError(f"Unknown omni backend: {cfg.omni_backend!r} (use moshi or mini_omni)")
