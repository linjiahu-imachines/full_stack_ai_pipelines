from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from staged_voice.backends import FasterWhisperASR, OllamaLLM
from staged_voice.backends.base_types import ChatMessage
from staged_voice.backends.tts_factory import make_tts
from staged_voice.config import RunConfig, overlay_from_yaml_dict
from staged_voice.pipeline import StagedVoicePipeline
from staged_voice.profiling import StageProfile

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-not-found]

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


class PipelineService:
    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._cfg = RunConfig()
        overlay_from_yaml_dict(self._cfg, _load_yaml(config_path))
        self._pipe: StagedVoicePipeline | None = None
        self._lock = threading.Lock()
        self._ready = False
        self._error: str | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    def load(self) -> None:
        try:
            logger.info("Loading staged pipeline from %s", self._config_path)
            cfg = RunConfig()
            overlay_from_yaml_dict(cfg, _load_yaml(self._config_path))

            if cfg.llm_backend == "ollama":
                llm = OllamaLLM(cfg.ollama_host, cfg.ollama_model)
            elif cfg.llm_backend == "hf":
                from staged_voice.backends.llm_hf_causal import HFCausalLM

                llm = HFCausalLM(
                    cfg.hf_model_name,
                    device=cfg.hf_device,
                    torch_dtype=cfg.hf_torch_dtype,
                )
            else:
                raise ValueError(f"Unknown llm backend: {cfg.llm_backend}")

            asr = FasterWhisperASR(
                model_size=cfg.whisper_model_size,
                device=cfg.whisper_device,
                compute_type=cfg.whisper_compute_type,
                language=cfg.whisper_language,
            )
            tts = make_tts(cfg)
            self._cfg = cfg
            self._pipe = StagedVoicePipeline(cfg, asr, llm, tts)
            self._ready = True
            logger.info(
                "Pipeline ready (llm=%s, tts=%s, whisper=%s)",
                cfg.llm_backend,
                cfg.tts_backend,
                cfg.whisper_device,
            )
        except Exception as e:
            self._error = str(e)
            self._ready = False
            logger.exception("Failed to load pipeline")
            raise

    def run_turn(
        self,
        wav_path: Path,
        history: list[ChatMessage],
        out_wav_path: Path,
    ) -> StageProfile:
        if not self._ready or self._pipe is None:
            raise RuntimeError(self._error or "Pipeline not loaded")
        with self._lock:
            return self._pipe.run_turn(
                wav_path,
                history=history,
                out_wav_path=out_wav_path,
            )
