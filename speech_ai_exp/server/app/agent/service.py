from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from staged_voice.backends.base_types import ChatMessage, LLMBackend

from app.agent.rag import KnowledgeBase, format_chunks_for_prompt
from app.agent.tools import ToolRegistry

logger = logging.getLogger(__name__)

_TOOL_RE = re.compile(
    r"TOOL_CALL:\s*(\{.*?\})\s*$",
    re.MULTILINE | re.DOTALL,
)
_FINAL_RE = re.compile(r"^FINAL:\s*(.+)$", re.MULTILINE | re.DOTALL)


@dataclass
class AgentResult:
    reply_text: str
    rag_sources: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    agent_steps: int = 0
    agent_s: float = 0.0


@dataclass
class AgentConfig:
    enabled: bool = True
    knowledge_dir: str = "data/knowledge"
    knowledge_index: str = "data/knowledge_index.json"
    rag_top_k: int = 3
    max_tool_steps: int = 3
    inject_rag: bool = True


class AgentService:
    """RAG + tool loop without LangChain."""

    def __init__(self, cfg: AgentConfig, kb: KnowledgeBase) -> None:
        self._cfg = cfg
        self._kb = kb
        self._tools = ToolRegistry(kb, rag_top_k=cfg.rag_top_k)

    @property
    def knowledge_chunks(self) -> int:
        return self._kb.chunk_count

    @property
    def loaded_from_index(self) -> bool:
        return self._kb.loaded_from_index

    @property
    def index_path(self) -> Path:
        return self._kb.index_path

    def run(
        self,
        *,
        llm: LLMBackend,
        transcript: str,
        history: list[ChatMessage],
        session_memory: dict[str, str],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> AgentResult:
        t0 = time.perf_counter()
        self._tools.bind_session_memory(session_memory)

        rag_hits = self._kb.search(transcript, top_k=self._cfg.rag_top_k)
        rag_block = format_chunks_for_prompt(rag_hits)
        rag_sources = list({h.chunk.source for h in rag_hits})

        agent_system = _build_agent_system_prompt(
            base=system_prompt,
            tools=self._tools.describe_for_prompt(),
            rag_block=rag_block if self._cfg.inject_rag else "",
        )

        messages: list[ChatMessage] = list(history) + [
            {"role": "user", "content": transcript},
        ]
        tool_calls: list[dict[str, Any]] = []
        reply_text = ""

        for step in range(self._cfg.max_tool_steps + 1):
            raw = _llm_complete(
                llm,
                messages=messages,
                system_prompt=agent_system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            final = _parse_final(raw)
            if final is not None:
                reply_text = final.strip()
                return AgentResult(
                    reply_text=reply_text,
                    rag_sources=rag_sources,
                    tool_calls=tool_calls,
                    agent_steps=step + 1,
                    agent_s=time.perf_counter() - t0,
                )

            tool_call = _parse_tool_call(raw)
            if tool_call is None:
                reply_text = raw.strip() or "I could not produce an answer."
                break

            name = str(tool_call.get("name", ""))
            arguments = tool_call.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            result = self._tools.run(name, arguments)
            tool_calls.append({"name": name, "arguments": arguments, "result": result[:500]})
            messages.append({"role": "assistant", "content": raw.strip()})
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool result for {name}:\n{result}\n\n"
                    "Continue. Use another TOOL_CALL or reply with FINAL: <short spoken answer>.",
                }
            )
            logger.info("Agent tool step %s: %s", step + 1, name)

        if not reply_text:
            reply_text = "I could not find an answer in time. Please try again."

        return AgentResult(
            reply_text=reply_text,
            rag_sources=rag_sources,
            tool_calls=tool_calls,
            agent_steps=len(tool_calls) + 1,
            agent_s=time.perf_counter() - t0,
        )


def _llm_complete(
    llm: LLMBackend,
    *,
    messages: list[ChatMessage],
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    chunks: list[str] = []
    for piece in llm.iter_chat_messages(
        messages=messages,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    ):
        chunks.append(piece)
    return "".join(chunks)


def _build_agent_system_prompt(*, base: str, tools: str, rag_block: str) -> str:
    parts = [
        base.strip(),
        "",
        "You are an agent with tools and a knowledge base.",
        "For voice output, FINAL answers must be one or two short spoken sentences.",
        "",
        "Available tools:",
        tools,
        "",
        "To call a tool, output exactly one line:",
        'TOOL_CALL: {"name": "<tool_name>", "arguments": {<json>}}',
        "",
        "When you can answer the user, output exactly one line:",
        "FINAL: <your short spoken answer>",
        "",
        "Do not invent facts. Use tool results and the knowledge passages below.",
    ]
    if rag_block:
        parts.extend(["", "Knowledge passages (may be incomplete):", rag_block])
    return "\n".join(parts)


def _parse_final(text: str) -> str | None:
    m = _FINAL_RE.search(text.strip())
    if m:
        return m.group(1).strip()
    return None


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    m = _TOOL_RE.search(text.strip())
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "name" not in payload:
        return None
    return payload
