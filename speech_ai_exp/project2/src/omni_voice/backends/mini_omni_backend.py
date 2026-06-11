from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from omni_voice.backends.base_types import OmniRunResult
from omni_voice.tts_espeak import synthesize_espeak

OUTPUT_SAMPLE_RATE = 24_000


def _mini_omni_root() -> Path:
    return Path(__file__).resolve().parents[3] / "third_party" / "mini-omni"


def _load_audio_whisper_mel(path: Path) -> tuple[Any, int, float]:
    import torch
    import whisper

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    duration_s = float(len(audio) / float(sr))
    if sr != 16_000:
        import torchaudio.functional as F

        audio = (
            F.resample(torch.from_numpy(audio).unsqueeze(0), sr, 16_000)
            .squeeze()
            .numpy()
        )
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    leng = int((len(audio) / 16_000) * 1000 / 20) + 1
    return mel, leng, duration_s


def _short_reply_for_speech(full_text: str) -> str:
    """Prefer a concise line for TTS so duration matches what users read."""
    text = full_text.strip()
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for s in reversed(sentences):
        s = s.strip()
        if not s:
            continue
        low = s.lower()
        if "equal" in low or re.search(r"\b\d+\b", s):
            return s if s.endswith((".", "!", "?")) else s + "."
    if len(text) > 160:
        return sentences[0].strip() if sentences else text[:160]
    return text


class MiniOmniBackend:
    """Mini-Omni A1_T2 (understand + answer text) + aligned speech (espeak or T1_A2)."""

    def __init__(
        self,
        *,
        ckpt_dir: str | Path | None = None,
        device: str = "auto",
        mode: str = "a1t2_t1a2",
        speech_align: str = "espeak",
        tts_voice: str = "en-us",
        tts_rate_wpm: int = 165,
    ) -> None:
        import torch

        root = _mini_omni_root()
        if not root.is_dir():
            raise ImportError(
                f"Mini-Omni not found at {root}. See project2/third_party/README.md"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        try:
            import inference as mo  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Mini-Omni deps missing: pip install litgpt==0.4.3 snac openai-whisper lightning"
            ) from e

        self._mo = mo
        self._device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if self._device == "auto":
            self._device = "cpu"
        ckpt = Path(ckpt_dir) if ckpt_dir else root / "checkpoint"
        if not ckpt.is_dir():
            mo.download_model(str(ckpt))
        self._fabric, self._model, self._tok, self._snac, self._whisper = mo.load_model(
            str(ckpt), self._device
        )
        self._mode = mode.lower()
        self._speech_align = speech_align.lower()
        self._tts_voice = tts_voice
        self._tts_rate_wpm = tts_rate_wpm

    def _latest_tmp_wav(self, tmp_dir: Path) -> Path:
        wavs = sorted(tmp_dir.glob("**/*.wav"), key=lambda p: p.stat().st_mtime)
        if not wavs:
            raise RuntimeError("Mini-Omni did not write an output WAV.")
        return wavs[-1]

    def _resample_to_24k(self, src: Path, dest: Path) -> None:
        import torch
        import torchaudio.functional as F

        data, sr = sf.read(str(src), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != OUTPUT_SAMPLE_RATE:
            t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
            data = F.resample(t, sr, OUTPUT_SAMPLE_RATE).squeeze().numpy()
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), data, OUTPUT_SAMPLE_RATE)

    def run_turn(self, wav_path: Path, out_wav_path: Path) -> OmniRunResult:
        mo = self._mo
        t0 = time.perf_counter()
        mel, leng, duration_s = _load_audio_whisper_mel(wav_path)
        tmp_dir = out_wav_path.parent / "_mo_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        audio_feature, input_ids = mo.get_input_ids_whisper(
            mel, leng, self._whisper, self._device
        )

        extra_meta: dict[str, Any] = {}
        speech_source = self._speech_align

        if self._mode == "a1_a2":
            answer_text = mo.A1_A2(
                self._fabric,
                audio_feature,
                input_ids,
                leng,
                self._model,
                self._tok,
                0,
                self._snac,
                out_dir=str(tmp_dir),
            )
            self._resample_to_24k(self._latest_tmp_wav(tmp_dir), out_wav_path)
            display_text = answer_text
            speech_source = "mini_omni_a1_a2"
        else:
            audio_feature, input_ids_at = mo.get_input_ids_whisper(
                mel,
                leng,
                self._whisper,
                self._device,
                special_token_a=mo._pad_a,
                special_token_t=mo._answer_t,
            )
            answer_text = mo.A1_T2(
                self._fabric,
                audio_feature,
                input_ids_at,
                leng,
                self._model,
                self._tok,
                0,
            )
            extra_meta["mini_omni_answer_text_full"] = answer_text
            speak_text = _short_reply_for_speech(answer_text)
            extra_meta["mini_omni_speak_text"] = speak_text

            if self._speech_align == "model":
                ta_ids = mo.get_input_ids_TA(speak_text, self._tok)
                spoken = mo.T1_A2(
                    self._fabric,
                    ta_ids,
                    self._model,
                    self._tok,
                    0,
                    self._snac,
                    out_dir=str(tmp_dir),
                )
                self._resample_to_24k(self._latest_tmp_wav(tmp_dir), out_wav_path)
                display_text = spoken.strip() or speak_text
                speech_source = "mini_omni_t1_a2"
            else:
                synthesize_espeak(
                    speak_text,
                    out_wav_path,
                    voice=self._tts_voice,
                    rate_wpm=self._tts_rate_wpm,
                )
                display_text = speak_text
                speech_source = "espeak_aligned"

        inference_s = time.perf_counter() - t0
        meta: dict[str, Any] = {
            "mini_omni_mode": self._mode,
            "mini_omni_device": self._device,
            "speech_source": speech_source,
            **extra_meta,
        }
        return OmniRunResult(
            output_wav_path=out_wav_path,
            audio_duration_s=duration_s,
            inference_s=inference_s,
            ttfa_s=inference_s,
            inner_text=display_text or "",
            meta=meta,
        )
