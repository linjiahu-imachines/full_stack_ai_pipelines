from __future__ import annotations

import time
from pathlib import Path

from omni_voice.backends.base_types import OmniBackend
from omni_voice.config import RunConfig
from omni_voice.profiling import OmniTurnProfile


class OmniVoicePipeline:
    def __init__(self, cfg: RunConfig, backend: OmniBackend) -> None:
        self._cfg = cfg
        self._backend = backend

    def run_turn(self, wav_path: Path) -> OmniTurnProfile:
        wav_path = wav_path.expanduser().resolve()
        out_dir = Path(self._cfg.audio_out_dir)
        if not out_dir.is_absolute():
            out_dir = (Path.cwd() / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / f"{wav_path.stem}_omni_reply.wav"

        t0 = time.perf_counter()
        result = self._backend.run_turn(wav_path, out_wav)
        e2e = time.perf_counter() - t0

        transcript = ""
        reply_text = ""
        meta = dict(result.meta)
        meta["inference_s"] = result.inference_s

        if self._cfg.sidecar_whisper:
            from omni_voice.sidecar_whisper import transcribe_wav

            try:
                transcript, in_meta = transcribe_wav(
                    wav_path,
                    model_size=self._cfg.whisper_model,
                    device=self._cfg.whisper_device,
                )
                meta["sidecar_input"] = in_meta
                skip_out_asr = meta.get("speech_source") == "espeak_aligned"
                if (
                    not skip_out_asr
                    and result.output_wav_path
                    and Path(result.output_wav_path).is_file()
                ):
                    reply_text, out_meta = transcribe_wav(
                        Path(result.output_wav_path),
                        model_size=self._cfg.whisper_model,
                        device=self._cfg.whisper_device,
                    )
                    meta["sidecar_output"] = out_meta
            except ImportError as e:
                meta["sidecar_error"] = str(e)

        if self._cfg.omni_backend == "mini_omni" and result.inner_text.strip():
            reply_text = result.inner_text.strip()

        return OmniTurnProfile(
            stack="voice_to_voice",
            backend=self._cfg.omni_backend,
            audio_path=str(wav_path),
            audio_duration_s=result.audio_duration_s,
            e2e_wall_s=e2e,
            ttfa_s=result.ttfa_s,
            output_wav_path=str(result.output_wav_path),
            transcript=transcript,
            reply_text=reply_text,
            inner_text=result.inner_text,
            meta=meta,
        )
