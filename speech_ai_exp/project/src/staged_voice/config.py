from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    """Runtime wiring for backends (CLI + optional YAML overlay)."""

    # ASR (faster-whisper)
    asr_backend: str = "faster_whisper"
    whisper_model_size: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_language: str | None = None  # ISO code or None for auto

    # LLM
    llm_backend: str = "ollama"  # ollama | hf | remote
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    remote_base_url: str = "http://127.0.0.1:8080"
    remote_model: str = "local-model"
    remote_api_key: str = ""
    system_prompt: str = (
        "You are a helpful voice assistant. Keep answers concise for spoken delivery."
    )
    max_tokens: int = 256
    temperature: float = 0.3

    hf_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_device: str = "auto"
    hf_max_new_tokens: int = 256
    hf_torch_dtype: str = "auto"  # auto | bf16 | fp16 | fp32

    # TTS: espeak | kokoro
    tts_backend: str = "espeak"  # espeak | kokoro
    tts_voice: str = "en-us"  # espeak voice
    tts_rate_wpm: int = 175  # espeak only
    kokoro_lang_code: str = "a"  # Kokoro: a=American English, b=British, ...
    kokoro_voice: str = "af_heart"
    kokoro_speed: float = 1.0
    kokoro_repo_id: str = "hexgrad/Kokoro-82M"
    audio_out_dir: Path = Path("audio/out")

    def effective_max_tokens(self) -> int:
        return self.hf_max_new_tokens if self.llm_backend == "hf" else self.max_tokens


def overlay_from_yaml_dict(cfg: RunConfig, data: dict[str, Any]) -> None:
    if not data:
        return
    if "asr" in data:
        a = data["asr"]
        cfg.whisper_model_size = a.get("model_size", cfg.whisper_model_size)
        cfg.whisper_device = a.get("device", cfg.whisper_device)
        cfg.whisper_compute_type = a.get("compute_type", cfg.whisper_compute_type)
        cfg.whisper_language = a.get("language", cfg.whisper_language)
    if "llm" in data:
        l = data["llm"]
        cfg.llm_backend = l.get("backend", cfg.llm_backend)
        cfg.ollama_host = l.get("ollama_host", cfg.ollama_host)
        cfg.ollama_model = l.get("ollama_model", cfg.ollama_model)
        cfg.remote_base_url = l.get("remote_base_url", cfg.remote_base_url)
        cfg.remote_model = l.get("remote_model", cfg.remote_model)
        cfg.remote_api_key = l.get("remote_api_key", cfg.remote_api_key)
        cfg.system_prompt = l.get("system_prompt", cfg.system_prompt)
        cfg.max_tokens = int(l.get("max_tokens", cfg.max_tokens))
        cfg.temperature = float(l.get("temperature", cfg.temperature))
        cfg.hf_model_name = l.get("hf_model", cfg.hf_model_name)
        cfg.hf_device = l.get("hf_device", cfg.hf_device)
        cfg.hf_max_new_tokens = int(l.get("hf_max_new_tokens", cfg.hf_max_new_tokens))
        cfg.hf_torch_dtype = l.get("hf_torch_dtype", cfg.hf_torch_dtype)
    if "tts" in data:
        t = data["tts"]
        cfg.tts_backend = t.get("backend", cfg.tts_backend)
        cfg.tts_voice = t.get("voice", cfg.tts_voice)
        cfg.tts_rate_wpm = int(t.get("rate_wpm", cfg.tts_rate_wpm))
        cfg.kokoro_lang_code = t.get("kokoro_lang_code", cfg.kokoro_lang_code)
        cfg.kokoro_voice = t.get("kokoro_voice", cfg.kokoro_voice)
        cfg.kokoro_speed = float(t.get("kokoro_speed", cfg.kokoro_speed))
        cfg.kokoro_repo_id = t.get("kokoro_repo_id", cfg.kokoro_repo_id)
