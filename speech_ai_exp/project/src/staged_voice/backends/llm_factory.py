from __future__ import annotations

import os
from typing import Any

from staged_voice.backends.llm_ollama import OllamaLLM
from staged_voice.backends.llm_remote_chat import RemoteChatLLM
from staged_voice.config import RunConfig


def make_llm(cfg: RunConfig, *, env: dict[str, str] | None = None) -> Any:
    env = env if env is not None else dict(os.environ)

    if cfg.llm_backend == "ollama":
        return OllamaLLM(cfg.ollama_host, cfg.ollama_model)
    if cfg.llm_backend == "hf":
        from staged_voice.backends.llm_hf_causal import HFCausalLM

        return HFCausalLM(
            cfg.hf_model_name,
            device=cfg.hf_device,
            torch_dtype=cfg.hf_torch_dtype,
        )
    if cfg.llm_backend == "remote":
        base_url = env.get("REMOTE_LLM_BASE_URL", cfg.remote_base_url)
        model = env.get("REMOTE_LLM_MODEL", cfg.remote_model)
        api_key = env.get("REMOTE_LLM_API_KEY", cfg.remote_api_key)
        timeout_raw = env.get("REMOTE_LLM_TIMEOUT_SEC", "").strip()
        timeout_sec = int(timeout_raw) if timeout_raw.isdigit() else None
        return RemoteChatLLM(base_url, model, api_key=api_key, timeout_sec=timeout_sec)

    raise ValueError(f"Unknown llm backend: {cfg.llm_backend}")
