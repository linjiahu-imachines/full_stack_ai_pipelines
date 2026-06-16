from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from staged_voice.backends.base_types import ChatMessage


def remote_chat_completions_url(base_url: str) -> str:
    """Build the chat-completions endpoint URL from a server base URL."""
    url = base_url.rstrip("/")
    if url.endswith("/v1/chat/completions"):
        return url
    return f"{url}/v1/chat/completions"


class RemoteChatLLM:
    """Streaming chat via a remote server's /v1/chat/completions API (SSE)."""

    def __init__(self, base_url: str, model: str, *, api_key: str = "") -> None:
        self._url = remote_chat_completions_url(base_url)
        self._model = model
        self._api_key = api_key

    @property
    def api_url(self) -> str:
        return self._url

    def iter_chat_messages(
        self,
        *,
        messages: list[ChatMessage],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        api_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for m in messages:
            role = m["role"]
            if role in ("user", "assistant"):
                api_messages.append({"role": role, "content": m["content"]})

        payload: dict[str, Any] = {
            "model": self._model,
            "stream": True,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(req, timeout=600) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content") or ""
                    if text:
                        yield text
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Remote LLM HTTP {e.code}: {body}") from e
        except URLError as e:
            raise RuntimeError(f"Remote LLM unreachable at {self._url}: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Remote LLM returned invalid JSON: {e}") from e
