from __future__ import annotations

from staged_voice.backends.base_types import TTSBackend
from staged_voice.backends.tts_espeak import EspeakNgTTS
from staged_voice.config import RunConfig


def make_tts(cfg: RunConfig) -> TTSBackend:
    backend = cfg.tts_backend.lower()
    if backend == "espeak":
        return EspeakNgTTS(voice=cfg.tts_voice, rate_wpm=cfg.tts_rate_wpm)
    if backend == "kokoro":
        from staged_voice.backends.tts_kokoro import KokoroTTS

        return KokoroTTS(
            lang_code=cfg.kokoro_lang_code,
            voice=cfg.kokoro_voice,
            speed=cfg.kokoro_speed,
            repo_id=cfg.kokoro_repo_id,
        )
    raise ValueError(f"Unknown TTS backend: {cfg.tts_backend!r} (use espeak or kokoro)")
