from __future__ import annotations

import warnings
from pathlib import Path


def transcribe_wav(
    wav_path: Path,
    *,
    model_size: str = "small",
    device: str = "auto",
    language: str | None = "en",
) -> tuple[str, dict]:
    """Post-hoc ASR for audit/demo (not in Moshi hot path)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError(
            "Sidecar Whisper requires: pip install faster-whisper"
        ) from e

    resolved_device = device
    compute_type = "int8"
    if device == "auto":
        try:
            import ctranslate2

            resolved_device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            resolved_device = "cpu"
    if resolved_device == "cuda":
        compute_type = "float16"

    model = WhisperModel(model_size, device=resolved_device, compute_type=compute_type)
    segments, info = model.transcribe(str(wav_path), language=language, beam_size=1)
    text = " ".join(s.text.strip() for s in segments).strip()
    meta = {
        "whisper_model": model_size,
        "whisper_device": resolved_device,
        "whisper_language": info.language,
        "whisper_duration_s": info.duration,
    }
    return text, meta
