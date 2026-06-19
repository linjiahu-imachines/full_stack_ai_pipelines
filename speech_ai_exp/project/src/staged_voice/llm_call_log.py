from __future__ import annotations

import logging
import socket
from typing import Any

from staged_voice.config import RunConfig

logger = logging.getLogger("staged_voice.llm")


def local_hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown-host"


def describe_llm_backend(cfg: RunConfig) -> str:
    if cfg.log_backend:
        return cfg.log_backend
    return cfg.llm_backend


def describe_llm_host(cfg: RunConfig) -> str:
    if cfg.log_runs_on:
        return cfg.log_runs_on
    backend = cfg.llm_backend
    if backend == "hf":
        return f"in-process@{local_hostname()}"
    if backend == "remote":
        return str(cfg.remote_base_url or "remote")
    if backend == "ollama":
        return str(cfg.ollama_host or "ollama")
    return backend


def describe_model_id(cfg: RunConfig, model_id: str | None) -> str:
    if cfg.log_id:
        return cfg.log_id
    return model_id or "-"


def describe_model_name(cfg: RunConfig) -> str:
    if cfg.log_model:
        return cfg.log_model
    if cfg.llm_backend == "hf":
        return cfg.hf_model_name
    if cfg.llm_backend == "remote":
        return cfg.remote_model
    if cfg.llm_backend == "ollama":
        return cfg.ollama_model
    return cfg.llm_backend


def log_llm_call(
    *,
    model_id: str | None,
    model_label: str | None,
    cfg: RunConfig,
    call_index: int,
    prompt_chars: int,
    prompt_tokens_est: float,
    output_chars: int,
    output_tokens_est: float,
    duration_s: float | None = None,
    tools_enabled: bool | None = None,
) -> None:
    """Structured system log for one LLM completion."""
    tools_note = ""
    if tools_enabled is not None:
        tools_note = f" | agent_tools={'on' if tools_enabled else 'off'}"
    duration_note = f" | duration_s={duration_s:.2f}" if duration_s is not None else ""
    logger.info(
        "LLM call #%s | id=%s | label=%s | backend=%s | model=%s | runs_on=%s | "
        "input_chars=%s input_tokens_est=%.0f | output_chars=%s output_tokens_est=%.0f%s%s",
        call_index,
        describe_model_id(cfg, model_id),
        model_label or "-",
        describe_llm_backend(cfg),
        describe_model_name(cfg),
        describe_llm_host(cfg),
        prompt_chars,
        prompt_tokens_est,
        output_chars,
        output_tokens_est,
        duration_note,
        tools_note,
    )


def log_llm_turn_summary(
    *,
    model_id: str | None,
    model_label: str | None,
    cfg: RunConfig,
    usage: dict[str, Any],
    llm_generation_s: float,
    tools_enabled: bool | None = None,
) -> None:
    """End-of-turn rollup when a voice turn used one or more LLM calls."""
    if not usage:
        return
    tools_note = ""
    if tools_enabled is not None:
        tools_note = f" | agent_tools={'on' if tools_enabled else 'off'}"
    logger.info(
        "LLM turn complete | id=%s | label=%s | backend=%s | model=%s | runs_on=%s | "
        "calls=%s | total_input_tokens_est=%.0f | total_output_tokens_est=%.0f | "
        "llm_wall_s=%.2f%s",
        describe_model_id(cfg, model_id),
        model_label or "-",
        describe_llm_backend(cfg),
        describe_model_name(cfg),
        describe_llm_host(cfg),
        int(usage.get("llm_calls", 1)),
        float(usage.get("prompt_tokens_est", 0)),
        float(usage.get("output_tokens_est", 0)),
        llm_generation_s,
        tools_note,
    )
