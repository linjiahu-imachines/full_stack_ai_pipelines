from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


class FasterWhisperASR:
    def __init__(
        self,
        *,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # noqa: WPS433
        except ImportError as e:
            raise ImportError(
                "Install faster-whisper: pip install 'staged-voice[asr-whisper]' "
                "or pip install faster-whisper"
            ) from e

        resolved_device, resolved_compute = self._resolve_device_compute(device, compute_type)
        self._model = WhisperModel(model_size, device=resolved_device, compute_type=resolved_compute)
        self._language = language

    @staticmethod
    def _ctranslate2_has_cuda() -> bool:
        """faster-whisper uses CTranslate2; PyTorch CUDA does not imply CT2 CUDA."""
        try:
            import ctranslate2  # noqa: WPS433

            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    @staticmethod
    def _resolve_device_compute(device: str, compute_type: str) -> tuple[str, str]:
        d_raw = device.lower()
        ct2_cuda = FasterWhisperASR._ctranslate2_has_cuda()

        if d_raw == "auto":
            d = "cuda" if ct2_cuda else "cpu"
        elif d_raw == "cuda" and not ct2_cuda:
            warnings.warn(
                "Whisper `--whisper-device cuda` was resolved, but this CTranslate2 "
                "build has no CUDA (common with CPU-only wheels). Falling back to CPU. "
                "Install CUDA-enabled `ctranslate2`/`faster-whisper` for GPU ASR or pass "
                "`--whisper-device cpu`.",
                UserWarning,
                stacklevel=2,
            )
            d = "cpu"
        else:
            d = d_raw

        ct = compute_type
        if ct == "auto":
            ct = "int8_float16" if d == "cuda" else "int8"
        return d, ct

    def transcribe_file(self, wav_path: Path) -> tuple[str, dict[str, Any]]:
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        audio_duration_s = float(len(data) / float(sr))

        segments, info = self._model.transcribe(
            data,
            language=self._language,
            task="transcribe",
            vad_filter=True,
        )
        parts: list[str] = []
        for seg in segments:
            parts.append(seg.text.strip())
        text = " ".join(p for p in parts if p).strip()
        meta: dict[str, Any] = {
            "asr_language": info.language,
            "asr_probability": getattr(info, "language_probability", None),
            "sample_rate": sr,
        }
        self._last_audio_duration_s = audio_duration_s
        return text, meta | {"audio_duration_s": audio_duration_s}
