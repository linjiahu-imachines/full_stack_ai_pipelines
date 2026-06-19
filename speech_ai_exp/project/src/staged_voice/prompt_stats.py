from __future__ import annotations

from typing import Any

from staged_voice.backends.base_types import ChatMessage


def estimate_tokens(chars: int) -> float:
    return max(chars, 1) / 4.0


def measure_text(text: str) -> dict[str, float | int]:
    chars = len(text)
    return {"chars": chars, "tokens_est": estimate_tokens(chars)}


def measure_prompt(system_prompt: str, messages: list[ChatMessage]) -> dict[str, Any]:
    system_chars = len(system_prompt)
    messages_chars = sum(len(str(m.get("content", ""))) for m in messages)
    prompt_chars = system_chars + messages_chars
    return {
        "system_chars": system_chars,
        "messages_chars": messages_chars,
        "prompt_chars": prompt_chars,
        "prompt_tokens_est": estimate_tokens(prompt_chars),
    }


def summarize_llm_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_chars = sum(int(c.get("prompt_chars", 0)) for c in calls)
    output_chars = sum(int(c.get("output_chars", 0)) for c in calls)
    return {
        "llm_calls": len(calls),
        "prompt_chars": prompt_chars,
        "prompt_tokens_est": estimate_tokens(prompt_chars),
        "system_chars": sum(int(c.get("system_chars", 0)) for c in calls),
        "messages_chars": sum(int(c.get("messages_chars", 0)) for c in calls),
        "output_chars": output_chars,
        "output_tokens_est": estimate_tokens(output_chars),
        "calls": calls,
    }
