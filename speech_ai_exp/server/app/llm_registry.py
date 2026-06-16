from __future__ import annotations

import copy
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from staged_voice.backends.llm_factory import make_llm
from staged_voice.config import RunConfig, overlay_from_yaml_dict

logger = logging.getLogger(__name__)


@dataclass
class LlmModelInfo:
    id: str
    label: str
    backend: str
    loaded: bool = False


@dataclass
class _LlmModelEntry:
    info: LlmModelInfo
    cfg: RunConfig
    _llm: Any = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class LlmRegistry:
    """Runtime registry of selectable LLM backends (lazy-load local HF)."""

    def __init__(
        self,
        *,
        base_cfg: RunConfig,
        models: dict[str, _LlmModelEntry],
        default_id: str,
    ) -> None:
        if default_id not in models:
            raise ValueError(f"Default llm model {default_id!r} not in registry")
        self._base_cfg = base_cfg
        self._models = models
        self._default_id = default_id

    @property
    def default_id(self) -> str:
        return self._default_id

    def list_models(self) -> list[LlmModelInfo]:
        return [copy.copy(entry.info) for entry in self._models.values()]

    def resolve(self, model_id: str | None) -> tuple[Any, RunConfig, LlmModelInfo]:
        mid = (model_id or "").strip() or self._default_id
        entry = self._models.get(mid)
        if entry is None:
            known = ", ".join(sorted(self._models))
            raise ValueError(f"Unknown llm model {mid!r}. Available: {known}")

        with entry._lock:
            if entry._llm is None:
                logger.info("Loading LLM model %s (backend=%s)", mid, entry.info.backend)
                entry._llm = make_llm(entry.cfg)
                entry.info.loaded = True
                logger.info("LLM model %s ready", mid)

        return entry._llm, entry.cfg, copy.copy(entry.info)

    def preload(self, model_id: str) -> None:
        self.resolve(model_id)


def parse_llm_registry(yaml_data: dict[str, Any], base_cfg: RunConfig) -> LlmRegistry:
    """Build registry from `llm_models` block or legacy single `llm` section."""
    block = yaml_data.get("llm_models")
    models_raw: dict[str, Any]
    default_id: str

    if isinstance(block, dict) and block.get("models"):
        models_raw = dict(block["models"])
        default_id = str(block.get("default") or next(iter(models_raw)))
    else:
        legacy = dict(yaml_data.get("llm") or {})
        default_id = "local_model_on_thor_machine"
        if legacy.get("backend") == "remote":
            default_id = "remote_model_on_remote_sim_im_cpu"
        models_raw = {
            "local_model_on_thor_machine": {
                "label": "Local model on imu-thor",
                "backend": "hf",
                "hf_model": base_cfg.hf_model_name,
                "hf_device": base_cfg.hf_device,
                "hf_max_new_tokens": base_cfg.hf_max_new_tokens,
                "temperature": base_cfg.temperature,
                "system_prompt": base_cfg.system_prompt,
            },
            "remote_model_on_remote_sim_im_cpu": {
                "label": "Remote model on sim IM CPU",
                "backend": "remote",
                "remote_base_url": base_cfg.remote_base_url,
                "remote_model": base_cfg.remote_model,
                "remote_api_key": base_cfg.remote_api_key,
                "max_tokens": base_cfg.max_tokens,
                "temperature": base_cfg.temperature,
                "system_prompt": base_cfg.system_prompt,
            },
        }
        if legacy:
            models_raw[default_id] = {**models_raw.get(default_id, {}), **legacy}

    entries: dict[str, _LlmModelEntry] = {}
    for mid, spec in models_raw.items():
        if not isinstance(spec, dict):
            continue
        cfg = copy.deepcopy(base_cfg)
        overlay_from_yaml_dict(cfg, {"llm": spec})
        label = str(spec.get("label") or mid)
        backend = str(spec.get("backend") or cfg.llm_backend)
        entries[mid] = _LlmModelEntry(
            info=LlmModelInfo(id=mid, label=label, backend=backend),
            cfg=cfg,
        )

    if not entries:
        raise ValueError("No llm models configured")

    env_default = os.environ.get("LLM_MODEL_DEFAULT", "").strip()
    if env_default and env_default in entries:
        default_id = env_default

    return LlmRegistry(base_cfg=base_cfg, models=entries, default_id=default_id)
