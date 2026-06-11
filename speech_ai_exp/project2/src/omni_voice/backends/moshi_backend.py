from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from omni_voice.backends.base_types import OmniRunResult

MOSHI_SAMPLE_RATE = 24_000
TEXT_PAD_IDS = {0, 3}


def _resolve_device(requested: str) -> str:
    d = requested.lower()
    if d != "auto":
        return d
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_wav_24k_mono(wav_path: Path) -> tuple[np.ndarray, float]:
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    duration = float(len(data) / float(sr))
    if sr != MOSHI_SAMPLE_RATE:
        try:
            import torch
            import torchaudio.functional as F

            t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
            t = F.resample(t, sr, MOSHI_SAMPLE_RATE)
            data = t.squeeze().numpy().astype(np.float32)
        except Exception as e:
            raise RuntimeError(
                f"Input must be {MOSHI_SAMPLE_RATE} Hz or install torchaudio for resampling. ({e})"
            ) from e
    return data, duration


def _decode_text_ids(tokenizer: Any, ids: list[int]) -> str:
    pieces: list[str] = []
    for tid in ids:
        if tid in TEXT_PAD_IDS:
            continue
        piece = tokenizer.id_to_piece(int(tid))
        pieces.append(piece.replace("▁", " "))
    return "".join(pieces).strip()


class MoshiBackend:
    """Batch turn: user WAV + trailing silence → Moshi stream; reply = audio after user segment."""

    def __init__(
        self,
        *,
        hf_repo: str = "kyutai/moshiko-pytorch-bf16",
        device: str = "auto",
        temp: float = 0.8,
        temp_text: float = 0.7,
        max_frames: int | None = None,
        input_silence_s: float = 2.0,
        response_s: float = 3.0,
    ) -> None:
        try:
            import torch
            from moshi.models import loaders
            from moshi.models.lm import LMGen
        except ImportError as e:
            raise ImportError(
                "Install Moshi: pip install moshi torch huggingface-hub (see project2/README.md)."
            ) from e

        self._torch = torch
        self._device = _resolve_device(device)
        self._hf_repo = hf_repo
        self._max_frames = max_frames
        self._input_silence_s = max(0.5, float(input_silence_s))
        self._response_s = max(0.5, float(response_s))

        info = loaders.CheckpointInfo.from_hf_repo(hf_repo)
        self._text_tokenizer = info.get_text_tokenizer()

        mimi = info.get_mimi(device=self._device)
        mimi.set_num_codebooks(8)
        self._mimi = mimi

        moshi_lm = info.get_moshi(device=self._device)
        gen_kw = dict(info.lm_gen_config)
        gen_kw.update(temp=temp, temp_text=temp_text)
        self._lm_gen = LMGen(moshi_lm, **gen_kw)
        self._frame_size = int(self._mimi.frame_size)
        self._samples_per_frame = self._frame_size

    def run_turn(self, wav_path: Path, out_wav_path: Path) -> OmniRunResult:
        torch = self._torch
        data, user_duration = _load_wav_24k_mono(wav_path)

        user_samples = len(data)
        silence_samples = int(self._input_silence_s * MOSHI_SAMPLE_RATE)
        data = np.concatenate([data, np.zeros(silence_samples, dtype=np.float32)])

        wav = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(self._device)
        n = wav.shape[-1]
        pad = (self._frame_size - (n % self._frame_size)) % self._frame_size
        if pad:
            wav = torch.nn.functional.pad(wav, (0, pad))

        all_codes: list[Any] = []
        with torch.no_grad(), self._mimi.streaming(1):
            for offset in range(0, wav.shape[-1], self._frame_size):
                frame = wav[:, :, offset : offset + self._frame_size]
                all_codes.append(self._mimi.encode(frame))

        if self._max_frames is not None:
            all_codes = all_codes[: self._max_frames]

        user_frames = int(np.ceil(user_samples / self._samples_per_frame))
        user_frames = min(user_frames, len(all_codes))

        out_chunks: list[Any] = []
        chunk_text_ids: list[int] = []
        ttfa_s = 0.0
        t0 = time.perf_counter()
        first_frame = True
        first_audio = True

        with torch.no_grad(), self._lm_gen.streaming(1), self._mimi.streaming(1):
            for code in all_codes:
                if first_frame:
                    _ = self._lm_gen.step(code)
                    first_frame = False
                tokens_out = self._lm_gen.step(code)
                if tokens_out is None:
                    continue
                tid = int(tokens_out[0, 0, 0].item())
                if tid not in TEXT_PAD_IDS:
                    chunk_text_ids.append(tid)
                audio_part = tokens_out[:, 1:]
                if audio_part.numel() > 0:
                    wav_chunk = self._mimi.decode(audio_part)
                    if first_audio:
                        ttfa_s = time.perf_counter() - t0
                        first_audio = False
                    out_chunks.append(wav_chunk.cpu())

        inference_s = time.perf_counter() - t0
        if not out_chunks:
            raise RuntimeError("Moshi produced no audio output.")

        full_wav = torch.cat(out_chunks, dim=-1).squeeze().numpy().astype(np.float32)

        # Reply = audio after user speech (skip duplex overlap / greeting during question).
        start_sample = user_frames * self._samples_per_frame
        start_sample = min(start_sample, max(0, full_wav.shape[0] - 1))
        reply_wav = full_wav[start_sample:]
        max_reply_samples = int(self._response_s * MOSHI_SAMPLE_RATE)
        if reply_wav.shape[0] > max_reply_samples:
            reply_wav = reply_wav[:max_reply_samples]

        if reply_wav.shape[0] < int(0.05 * MOSHI_SAMPLE_RATE):
            raise RuntimeError(
                "Moshi reply segment is too short. Increase moshi.input_silence_s or response_s."
            )

        out_wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav_path), reply_wav, MOSHI_SAMPLE_RATE)

        reply_text_ids = chunk_text_ids[user_frames:] if user_frames < len(chunk_text_ids) else chunk_text_ids
        listen_text = _decode_text_ids(self._text_tokenizer, chunk_text_ids[:user_frames])
        reply_inner = _decode_text_ids(self._text_tokenizer, reply_text_ids)

        meta: dict[str, Any] = {
            "moshi_hf_repo": self._hf_repo,
            "moshi_device": self._device,
            "frames_in": len(all_codes),
            "user_frames": user_frames,
            "frames_out": len(out_chunks),
            "sample_rate": MOSHI_SAMPLE_RATE,
            "input_silence_s": self._input_silence_s,
            "response_s": self._response_s,
            "listen_inner_text": listen_text,
            "reply_trim_from_sample": start_sample,
        }
        return OmniRunResult(
            output_wav_path=out_wav_path,
            audio_duration_s=user_duration,
            inference_s=inference_s,
            ttfa_s=ttfa_s,
            inner_text=reply_inner,
            meta=meta,
        )
