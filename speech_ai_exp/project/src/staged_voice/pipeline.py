from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from staged_voice.backends.base_types import ASRBackend, ChatMessage, LLMBackend, TTSBackend
from staged_voice.config import RunConfig
from staged_voice.llm_call_log import log_llm_call, log_llm_turn_summary
from staged_voice.profiling import StageProfile
from staged_voice.prompt_stats import measure_prompt, measure_text, summarize_llm_calls

AgentRunner = Callable[
    [str, list[ChatMessage], LLMBackend, RunConfig],
    tuple[str, dict[str, Any]],
]


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
        agent_runner: AgentRunner | None = None,
        session_memory: dict[str, str] | None = None,
        llm: LLMBackend | None = None,
        llm_cfg: RunConfig | None = None,
        llm_model_id: str | None = None,
        llm_model_label: str | None = None,
        tools_enabled: bool | None = None,
    ) -> StageProfile:
        wav_path = wav_path.expanduser().resolve()
        history = list(history or [])
        active_llm = llm if llm is not None else self._llm
        active_cfg = llm_cfg if llm_cfg is not None else self._cfg
        if active_llm is None:
            raise RuntimeError("No LLM backend configured for this turn")

        meta: dict[str, Any] = {"chat_turns_prior": len(history)}
        if llm_model_id:
            meta["llm_model"] = llm_model_id

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
        agent_meta: dict[str, Any] = {}

        if agent_runner is not None:
            reply_text, agent_meta = agent_runner(
                transcript,
                list(history or []),
                active_llm,
                active_cfg,
            )
            llm_generation_s = time.perf_counter() - t_llm0
            llm_ttft_s = float(agent_meta.get("agent_ttft_s", llm_generation_s))
            if session_memory is not None and "session_memory" in agent_meta:
                session_memory.update(agent_meta["session_memory"])
        else:
            prompt_stats = measure_prompt(active_cfg.system_prompt, llm_messages)
            chunks: list[str] = []
            stream = active_llm.iter_chat_messages(
                messages=llm_messages,
                system_prompt=active_cfg.system_prompt,
                max_tokens=active_cfg.effective_max_tokens(),
                temperature=active_cfg.temperature,
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
            output_stats = measure_text("".join(chunks))
            llm_usage = summarize_llm_calls(
                [
                    {
                        "call_index": 1,
                        **prompt_stats,
                        "output_chars": output_stats["chars"],
                        "output_tokens_est": output_stats["tokens_est"],
                    }
                ]
            )
            agent_meta["llm_usage"] = llm_usage
            log_llm_call(
                model_id=llm_model_id,
                model_label=llm_model_label,
                cfg=active_cfg,
                call_index=1,
                prompt_chars=int(prompt_stats["prompt_chars"]),
                prompt_tokens_est=float(prompt_stats["prompt_tokens_est"]),
                output_chars=int(output_stats["chars"]),
                output_tokens_est=float(output_stats["tokens_est"]),
                duration_s=llm_generation_s,
                tools_enabled=tools_enabled,
            )

        llm_chars = len(reply_text)
        usage = agent_meta.get("llm_usage") or {}
        log_llm_turn_summary(
            model_id=llm_model_id,
            model_label=llm_model_label,
            cfg=active_cfg,
            usage=usage,
            llm_generation_s=llm_generation_s,
            tools_enabled=tools_enabled,
        )
        tok_est = float(usage.get("output_tokens_est") or measure_text(reply_text)["tokens_est"])
        llm_tps = tok_est / llm_generation_s if llm_generation_s > 0 else 0.0
        meta["llm"] = {
            "backend": active_cfg.llm_backend,
            "model_id": llm_model_id,
        }
        if usage:
            meta["llm"]["usage"] = usage
        if agent_meta:
            meta["agent"] = agent_meta

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
