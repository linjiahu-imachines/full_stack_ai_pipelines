"""Backend implementations for ASR, LLM, and TTS.

`HFCausalLM` imports `torch` — import lazily from `staged_voice.backends.llm_hf_causal` when needed.
"""

from staged_voice.backends.asr_faster_whisper import FasterWhisperASR
from staged_voice.backends.llm_ollama import OllamaLLM
from staged_voice.backends.tts_espeak import EspeakNgTTS

__all__ = [
    "EspeakNgTTS",
    "FasterWhisperASR",
    "OllamaLLM",
]
