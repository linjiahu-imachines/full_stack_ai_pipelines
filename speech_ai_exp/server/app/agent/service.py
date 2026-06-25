from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from staged_voice.backends.base_types import ChatMessage, LLMBackend
from staged_voice.config import RunConfig
from staged_voice.llm_call_log import log_llm_call
from staged_voice.prompt_stats import measure_prompt, measure_text, summarize_llm_calls

from app.agent.rag import KnowledgeBase, format_chunks_for_prompt
from app.agent.tools import ToolRegistry
from app.agent.web_search import WebSearchConfig, make_web_search, should_auto_web_search
from app.db.config import DatabaseConfig
from app.db.repositories import CommerceRepository

logger = logging.getLogger(__name__)

_TOOL_RE = re.compile(
    r"TOOL_CALL:\s*(\{.*?\})\s*$",
    re.MULTILINE | re.DOTALL,
)
_FINAL_RE = re.compile(
    r"^\s*(?:FINAL|Final):\s*(.+)$",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
_FINAL_INLINE_RE = re.compile(r"^\s*(?:FINAL|Final):\s*(.+)$", re.DOTALL | re.IGNORECASE)


@dataclass
class AgentResult:
    reply_text: str
    rag_sources: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    agent_steps: int = 0
    agent_s: float = 0.0
    llm_usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    enabled: bool = True
    knowledge_dir: str = "data/knowledge"
    knowledge_index: str = "data/knowledge_index.json"
    rag_top_k: int = 3
    max_tool_steps: int = 3
    inject_rag: bool = True
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)


class AgentService:
    """RAG + tool loop without LangChain."""

    def __init__(
        self,
        cfg: AgentConfig,
        kb: KnowledgeBase,
        *,
        commerce: CommerceRepository | None = None,
    ) -> None:
        self._cfg = cfg
        self._kb = kb
        self._commerce = commerce
        web = make_web_search(cfg.web_search)
        self._tools = ToolRegistry(
            kb,
            rag_top_k=cfg.rag_top_k,
            web_search=web,
            commerce=commerce,
        )

    @property
    def web_search_enabled(self) -> bool:
        return self._tools.web_search_enabled

    @property
    def knowledge_chunks(self) -> int:
        return self._kb.chunk_count

    @property
    def loaded_from_index(self) -> bool:
        return self._kb.loaded_from_index

    @property
    def rag_backend(self) -> str:
        return getattr(self._kb, "rag_backend", "bm25")

    @property
    def vector_doc_count(self) -> int:
        return getattr(self._kb, "vector_doc_count", 0)

    @property
    def commerce_enabled(self) -> bool:
        return self._tools.commerce_enabled

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
        llm_cfg: RunConfig,
        model_id: str | None = None,
        model_label: str | None = None,
    ) -> AgentResult:
        t0 = time.perf_counter()
        self._tools.bind_session_memory(session_memory)

        rag_hits: list = []
        rag_sources: list[str] = []
        auto_web = (
            self._tools.web_search_enabled
            and self._cfg.web_search.auto_search
            and should_auto_web_search(transcript)
        )
        if self._cfg.inject_rag and not auto_web:
            rag_hits = self._kb.search(transcript, top_k=self._cfg.rag_top_k)
            rag_sources = list({h.chunk.source for h in rag_hits})

        tool_calls: list[dict[str, Any]] = []
        web_block = ""
        if auto_web:
            web_query = transcript.strip()
            web_result = self._tools.run("search_web", {"query": web_query})
            web_block = web_result
            tool_calls.append(
                {
                    "name": "search_web",
                    "arguments": {"query": web_query},
                    "result": web_result[:500],
                    "auto": True,
                }
            )
            logger.info("Auto web search for query: %s", web_query[:80])

        rag_block = format_chunks_for_prompt(rag_hits) if rag_hits else ""

        agent_system = _build_agent_system_prompt(
            base=system_prompt,
            tools=self._tools.describe_for_prompt(),
            rag_block=rag_block if self._cfg.inject_rag else "",
            web_block=web_block,
            web_search_enabled=self._tools.web_search_enabled,
        )

        messages: list[ChatMessage] = list(history) + [
            {"role": "user", "content": transcript},
        ]
        reply_text = ""
        llm_call_records: list[dict[str, Any]] = []

        for step in range(self._cfg.max_tool_steps + 1):
            prompt_stats = measure_prompt(agent_system, messages)
            t_call = time.perf_counter()
            raw = _llm_complete(
                llm,
                messages=messages,
                system_prompt=agent_system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            call_s = time.perf_counter() - t_call
            output_stats = measure_text(raw)
            llm_call_records.append(
                {
                    "call_index": step + 1,
                    **prompt_stats,
                    "output_chars": output_stats["chars"],
                    "output_tokens_est": output_stats["tokens_est"],
                }
            )
            log_llm_call(
                model_id=model_id,
                model_label=model_label,
                cfg=llm_cfg,
                call_index=step + 1,
                prompt_chars=int(prompt_stats["prompt_chars"]),
                prompt_tokens_est=float(prompt_stats["prompt_tokens_est"]),
                output_chars=int(output_stats["chars"]),
                output_tokens_est=float(output_stats["tokens_est"]),
                duration_s=call_s,
                tools_enabled=True,
            )
            final = _parse_final(raw)
            if final is not None:
                reply_text = sanitize_agent_reply(final)
                return AgentResult(
                    reply_text=reply_text,
                    rag_sources=rag_sources,
                    tool_calls=tool_calls,
                    agent_steps=step + 1,
                    agent_s=time.perf_counter() - t0,
                    llm_usage=summarize_llm_calls(llm_call_records),
                )

            tool_call = _parse_tool_call(raw)
            if tool_call is None:
                reply_text = sanitize_agent_reply(raw.strip() or "I could not produce an answer.")
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

        reply_text = sanitize_agent_reply(reply_text)

        return AgentResult(
            reply_text=reply_text,
            rag_sources=rag_sources,
            tool_calls=tool_calls,
            agent_steps=len(tool_calls) + 1,
            agent_s=time.perf_counter() - t0,
            llm_usage=summarize_llm_calls(llm_call_records),
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


def _build_agent_system_prompt(
    *,
    base: str,
    tools: str,
    rag_block: str,
    web_block: str = "",
    web_search_enabled: bool = False,
) -> str:
    parts = [
        base.strip(),
        "",
        "You are an agent with tools and a customer-service knowledge base.",
        "For voice output, FINAL answers must be one or two short spoken sentences.",
        "",
        "Available tools:",
        tools,
        "",
        "To call a tool, output exactly one line (no other text on that line):",
        'TOOL_CALL: {"name": "<tool_name>", "arguments": {<json>}}',
        "",
        "When you can answer the user, output exactly one line:",
        "FINAL: <your short spoken answer>",
        "",
        "Routing rules:",
        "- Orders, shipping, billing, returns, licenses, subscriptions: use lookup_customer, "
        "lookup_order, lookup_invoice, lookup_payment_status, list_open_orders, "
        "list_delayed_orders, lookup_licenses, lookup_subscription, get_active_order, "
        "compare_support_plans, search_knowledge_base, and/or knowledge passages below.",
        "- If the caller gives only a name or vague account request, use lookup_customer or "
        "ask for customer ID / email, then remember it for follow-up turns.",
        "- Spoken order numbers (e.g. 784921) and customer IDs (e.g. C zero zero one two five "
        "seven) should be passed as-is to lookup tools — normalization is handled server-side.",
    ]
    if web_search_enabled:
        parts.extend(
            [
                "- Weather, news, stocks, exchange rates, and other live public facts: "
                "use search_web and/or the web search results below. "
                "Do not use the knowledge base for these.",
                "- If the user asks for live public data and no web results are below yet, "
                "your next line must be a search_web TOOL_CALL (do not guess).",
                "- Never say you cannot access the internet or real-time data when "
                "search_web is available or web results are provided below.",
                "",
                "Example (weather):",
                'TOOL_CALL: {"name": "search_web", "arguments": {"query": "Seattle weather forecast today"}}',
            ]
        )
    parts.append("Do not invent facts. Prefer tool results and the passages below.")
    if web_block and not web_block.startswith("Error:"):
        parts.extend(
            [
                "",
                "Web search results (live public data — use these for the user's question):",
                web_block,
            ]
        )
    elif web_block:
        parts.extend(["", "Web search note:", web_block])
    if rag_block:
        parts.extend(["", "Knowledge passages (Alex / Horizon Store — not for weather/news):", rag_block])
    return "\n".join(parts)


def _parse_final(text: str) -> str | None:
    m = _FINAL_RE.search(text.strip())
    if m:
        return m.group(1).strip()
    m = _FINAL_INLINE_RE.match(text.strip())
    if m:
        return m.group(1).strip()
    return None


def sanitize_agent_reply(text: str) -> str:
    """Strip agent protocol prefix (FINAL:/Final:) for display and TTS."""
    parsed = _parse_final(text)
    if parsed is not None:
        return parsed
    return text.strip()


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    for line in stripped.splitlines():
        candidate = line.strip()
        if not candidate.upper().startswith("TOOL_CALL:"):
            continue
        payload_str = candidate.split(":", 1)[1].strip()
        payload = _load_tool_call_payload(payload_str)
        if payload is not None:
            return payload

    m = _TOOL_RE.search(stripped)
    if not m:
        return None
    return _load_tool_call_payload(m.group(1))


def _load_tool_call_payload(payload_str: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "name" not in payload:
        return None
    return payload
