from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class ASRBackend(Protocol):
    def transcribe_file(self, wav_path: Path) -> tuple[str, dict[str, Any]]: ...


class LLMBackend(Protocol):
    def iter_chat_messages(
        self,
        *,
        messages: list[ChatMessage],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        """Yield streamed assistant text for a full chat history (multi-turn)."""


class TTSBackend(Protocol):
    def synthesize_file(self, text: str, out_wav_path: Path) -> dict[str, Any]: ...
