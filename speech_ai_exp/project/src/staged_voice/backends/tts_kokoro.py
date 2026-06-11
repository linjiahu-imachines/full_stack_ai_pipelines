from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# Kokoro-82M outputs 24 kHz mono float audio.
KOKORO_SAMPLE_RATE = 24_000


class KokoroTTS:
    """Neural TTS via hexgrad/Kokoro-82M (Apache 2.0). Requires `espeak-ng` on PATH."""

    def __init__(
        self,
        *,
        lang_code: str = "a",
        voice: str = "af_heart",
        speed: float = 1.0,
        repo_id: str = "hexgrad/Kokoro-82M",
    ) -> None:
        if not shutil.which("espeak-ng") and not shutil.which("espeak"):
            raise EnvironmentError(
                "Kokoro uses espeak-ng for phonemization. Install: sudo apt install espeak-ng"
            )
        try:
            from kokoro import KPipeline  # noqa: WPS433
        except ImportError as e:
            raise ImportError(
                "Install Kokoro TTS: pip install 'staged-voice[tts-kokoro]' or pip install 'kokoro>=0.9.4'"
            ) from e

        self._lang_code = lang_code
        self._voice = voice
        self._speed = float(speed)
        self._repo_id = repo_id
        self._pipeline = KPipeline(lang_code=lang_code, repo_id=repo_id)

    def synthesize_file(self, text: str, out_wav_path: Path) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("Kokoro TTS received empty text")

        out_wav_path.parent.mkdir(parents=True, exist_ok=True)
        chunks: list[np.ndarray] = []
        for result in self._pipeline(text, voice=self._voice, speed=self._speed):
            audio = result.audio
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))

        if not chunks:
            raise RuntimeError("Kokoro produced no audio segments")

        waveform = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        sf.write(str(out_wav_path), waveform, KOKORO_SAMPLE_RATE)

        return {
            "tts_engine": "kokoro",
            "tts_model": self._repo_id,
            "kokoro_lang_code": self._lang_code,
            "kokoro_voice": self._voice,
            "kokoro_speed": self._speed,
            "sample_rate": KOKORO_SAMPLE_RATE,
            "segments": len(chunks),
        }
