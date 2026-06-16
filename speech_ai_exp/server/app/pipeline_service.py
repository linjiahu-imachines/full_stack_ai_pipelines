from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from staged_voice.backends import FasterWhisperASR
from staged_voice.backends.base_types import ChatMessage
from staged_voice.backends.tts_factory import make_tts
from staged_voice.config import RunConfig, overlay_from_yaml_dict
from staged_voice.pipeline import StagedVoicePipeline
from staged_voice.profiling import StageProfile

from app.agent.rag import KnowledgeBase
from app.agent.service import AgentConfig, AgentService
from app.llm_registry import LlmRegistry, parse_llm_registry

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-not-found]

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _parse_agent_config(yaml_data: dict[str, Any], server_root: Path) -> AgentConfig:
    raw = yaml_data.get("agent") or {}
    if not isinstance(raw, dict):
        raw = {}
    enabled = str(os.environ.get("AGENT_ENABLED", raw.get("enabled", True))).lower() not in (
        "0",
        "false",
        "no",
    )
    kb_dir = os.environ.get("AGENT_KNOWLEDGE_DIR", raw.get("knowledge_dir", "data/knowledge"))
    kb_path = Path(kb_dir)
    if not kb_path.is_absolute():
        kb_path = (server_root / kb_path).resolve()
    index_raw = os.environ.get(
        "AGENT_KNOWLEDGE_INDEX", raw.get("knowledge_index", "data/knowledge_index.json")
    )
    index_path = Path(index_raw)
    if not index_path.is_absolute():
        index_path = (server_root / index_path).resolve()
    return AgentConfig(
        enabled=enabled,
        knowledge_dir=str(kb_path),
        knowledge_index=str(index_path),
        rag_top_k=int(raw.get("rag_top_k", 3)),
        max_tool_steps=int(raw.get("max_tool_steps", 3)),
        inject_rag=bool(raw.get("inject_rag", True)),
    )


class PipelineService:
    def __init__(self, config_path: Path, *, server_root: Path) -> None:
        self._config_path = config_path
        self._server_root = server_root
        self._cfg = RunConfig()
        overlay_from_yaml_dict(self._cfg, _load_yaml(config_path))
        self._yaml = _load_yaml(config_path)
        self._pipe: StagedVoicePipeline | None = None
        self._llm_registry: LlmRegistry | None = None
        self._agent: AgentService | None = None
        self._agent_cfg = _parse_agent_config(self._yaml, server_root)
        self._lock = threading.Lock()
        self._ready = False
        self._error: str | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def agent_enabled(self) -> bool:
        return self._agent is not None

    @property
    def llm_registry(self) -> LlmRegistry | None:
        return self._llm_registry

    @property
    def agent_info(self) -> dict[str, Any]:
        if self._agent is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "knowledge_dir": self._agent_cfg.knowledge_dir,
            "knowledge_index": self._agent_cfg.knowledge_index,
            "index_loaded": self._agent.loaded_from_index,
            "knowledge_chunks": self._agent.knowledge_chunks,
            "max_tool_steps": self._agent_cfg.max_tool_steps,
            "rag_top_k": self._agent_cfg.rag_top_k,
        }

    def list_llm_models(self) -> list[dict[str, Any]]:
        if self._llm_registry is None:
            return []
        return [
            {
                "id": m.id,
                "label": m.label,
                "backend": m.backend,
                "loaded": m.loaded,
            }
            for m in self._llm_registry.list_models()
        ]

    def load(self) -> None:
        try:
            logger.info("Loading staged pipeline from %s", self._config_path)
            cfg = RunConfig()
            overlay_from_yaml_dict(cfg, self._yaml)

            self._llm_registry = parse_llm_registry(self._yaml, cfg)
            default_llm, default_cfg, _ = self._llm_registry.resolve(None)
            preload = os.environ.get("LLM_PRELOAD", "").strip()
            if preload:
                self._llm_registry.preload(preload)
            else:
                # Eager-load remote models; lazy-load local HF until first use.
                for entry in self._llm_registry.list_models():
                    if entry.backend == "remote":
                        self._llm_registry.preload(entry.id)

            asr = FasterWhisperASR(
                model_size=cfg.whisper_model_size,
                device=cfg.whisper_device,
                compute_type=cfg.whisper_compute_type,
                language=cfg.whisper_language,
            )
            tts = make_tts(cfg)
            self._cfg = cfg
            self._pipe = StagedVoicePipeline(cfg, asr, default_llm, tts)

            if self._agent_cfg.enabled:
                kb = KnowledgeBase(
                    Path(self._agent_cfg.knowledge_dir),
                    index_path=Path(self._agent_cfg.knowledge_index),
                )
                n = kb.load()
                self._agent = AgentService(self._agent_cfg, kb)
                logger.info(
                    "Agent enabled (knowledge_chunks=%s, dir=%s, index=%s, from_index=%s)",
                    n,
                    self._agent_cfg.knowledge_dir,
                    kb.index_path,
                    kb.loaded_from_index,
                )
            else:
                self._agent = None
                logger.info("Agent disabled")

            self._ready = True
            logger.info(
                "Pipeline ready (llm_models=%s, default=%s, tts=%s, whisper=%s, agent=%s)",
                [m.id for m in self._llm_registry.list_models()],
                self._llm_registry.default_id,
                cfg.tts_backend,
                cfg.whisper_device,
                self.agent_enabled,
            )
        except Exception as e:
            self._error = str(e)
            self._ready = False
            logger.exception("Failed to load pipeline")
            raise

    def _agent_runner(self, session_memory: dict[str, str]):
        agent = self._agent
        if agent is None:
            return None

        def runner(
            transcript: str,
            history: list[ChatMessage],
            llm: Any,
            cfg: RunConfig,
        ) -> tuple[str, dict[str, Any]]:
            result = agent.run(
                llm=llm,
                transcript=transcript,
                history=history,
                session_memory=session_memory,
                system_prompt=cfg.system_prompt,
                max_tokens=cfg.effective_max_tokens(),
                temperature=cfg.temperature,
            )
            meta = {
                "rag_sources": result.rag_sources,
                "tool_calls": result.tool_calls,
                "agent_steps": result.agent_steps,
                "agent_s": result.agent_s,
                "session_memory": dict(session_memory),
            }
            return result.reply_text, meta

        return runner

    def run_turn(
        self,
        wav_path: Path,
        history: list[ChatMessage],
        out_wav_path: Path,
        *,
        session_memory: dict[str, str] | None = None,
        llm_model: str | None = None,
    ) -> StageProfile:
        if not self._ready or self._pipe is None or self._llm_registry is None:
            raise RuntimeError(self._error or "Pipeline not loaded")
        mem = session_memory if session_memory is not None else {}
        agent_runner = self._agent_runner(mem) if self._agent else None
        llm, llm_cfg, model_info = self._llm_registry.resolve(llm_model)
        with self._lock:
            return self._pipe.run_turn(
                wav_path,
                history=history,
                out_wav_path=out_wav_path,
                agent_runner=agent_runner,
                session_memory=mem,
                llm=llm,
                llm_cfg=llm_cfg,
                llm_model_id=model_info.id,
            )
