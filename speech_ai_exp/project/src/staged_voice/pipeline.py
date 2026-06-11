from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from staged_voice.backends.base_types import ASRBackend, ChatMessage, LLMBackend, TTSBackend
from staged_voice.config import RunConfig
from staged_voice.profiling import StageProfile


def _rough_tokens_from_text(text: str) -> float:
    return max(len(text.strip()), 1) / 4.0


class StagedVoicePipeline:
    def __init__(
        self,
        cfg: RunConfig,
        asr: ASRBackend,
        llm: LLMBackend,
        tts: TTSBackend,
    ) -> None:
        self._cfg = cfg
        self._asr = asr
        self._llm = llm
        self._tts = tts

    def run_turn(
        self,
        wav_path: Path,
        *,
        history: list[ChatMessage] | None = None,
        out_wav_path: Path | None = None,
    ) -> StageProfile:
        wav_path = wav_path.expanduser().resolve()
        history = list(history or [])

        meta: dict[str, Any] = {"chat_turns_prior": len(history)}

        t_asr0 = time.perf_counter()
        transcript, asr_meta = self._asr.transcribe_file(wav_path)
        t_asr1 = time.perf_counter()
        asr_s = t_asr1 - t_asr0
        duration = float(asr_meta.get("audio_duration_s") or 0.0)
        asr_rtf = (asr_s / duration) if duration > 0 else 0.0
        meta["asr"] = asr_meta

        llm_messages: list[ChatMessage] = list(history) + [
            {"role": "user", "content": transcript},
        ]

        t_llm0 = time.perf_counter()
        first_token = True
        llm_ttft_s = 0.0
        chunks: list[str] = []
        stream = self._llm.iter_chat_messages(
            messages=llm_messages,
            system_prompt=self._cfg.system_prompt,
            max_tokens=self._cfg.effective_max_tokens(),
            temperature=self._cfg.temperature,
        )
        for piece in stream:
            if first_token:
                llm_ttft_s = time.perf_counter() - t_llm0
                first_token = False
            chunks.append(piece)
        t_llm1 = time.perf_counter()
        llm_generation_s = t_llm1 - t_llm0
        if first_token:
            llm_ttft_s = llm_generation_s
        reply_text = "".join(chunks).strip()
        llm_chars = len(reply_text)
        tok_est = _rough_tokens_from_text(reply_text)
        llm_tps = tok_est / llm_generation_s if llm_generation_s > 0 else 0.0
        meta["llm"] = {"backend": self._cfg.llm_backend}

        if out_wav_path is not None:
            out_wav = out_wav_path.expanduser().resolve()
            out_wav.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path(self._cfg.audio_out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_wav = out_dir / f"{wav_path.stem}_reply.wav"

        t_tts0 = time.perf_counter()
        tts_meta = self._tts.synthesize_file(reply_text, out_wav)
        t_tts1 = time.perf_counter()
        tts_s = t_tts1 - t_tts0
        meta["tts"] = tts_meta

        return StageProfile(
            audio_path=str(wav_path),
            audio_duration_s=duration,
            asr_s=asr_s,
            asr_rtf=asr_rtf,
            llm_ttft_s=llm_ttft_s,
            llm_generation_s=llm_generation_s,
            llm_chars=llm_chars,
            llm_tokens_per_s_est=llm_tps,
            tts_s=tts_s,
            transcript=transcript,
            reply_text=reply_text,
            output_wav_path=str(out_wav),
            meta=meta,
        )
