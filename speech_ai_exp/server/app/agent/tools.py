from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from app.agent.rag import KnowledgeBase, format_chunks_for_prompt
from app.agent.web_search import TavilyWebSearch, WebSearchConfig


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]


class ToolRegistry:
    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        rag_top_k: int = 3,
        web_search: TavilyWebSearch | None = None,
    ) -> None:
        self._kb = kb
        self._rag_top_k = rag_top_k
        self._web_search = web_search
        self._session_memory: dict[str, str] = {}
        self._tools: dict[str, ToolSpec] = {}
        self._register_defaults()

    @property
    def web_search_enabled(self) -> bool:
        return self._web_search is not None

    def bind_session_memory(self, memory: dict[str, str]) -> None:
        self._session_memory = memory

    def _register_defaults(self) -> None:
        self.register(
            ToolSpec(
                name="search_knowledge_base",
                description=(
                    "Search the customer-service knowledge base (.md/.txt files). "
                    "Use for Alex's orders, shipments, returns, refunds, warranties, "
                    "loyalty benefits, and store policies."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
                handler=self._search_kb,
            )
        )
        if self._web_search is not None:
            self.register(
                ToolSpec(
                    name="search_web",
                    description=(
                        "Search the public web for current events, general facts, "
                        "or information not in the internal knowledge base."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Web search query"},
                        },
                        "required": ["query"],
                    },
                    handler=self._search_web,
                )
            )
        self.register(
            ToolSpec(
                name="get_current_time",
                description="Return the current UTC date and time.",
                parameters={"type": "object", "properties": {}},
                handler=self._get_time,
            )
        )
        self.register(
            ToolSpec(
                name="remember",
                description="Store a short fact for this conversation session.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["key", "value"],
                },
                handler=self._remember,
            )
        )
        self.register(
            ToolSpec(
                name="recall",
                description="Recall a fact stored earlier in this session.",
                parameters={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
                handler=self._recall,
            )
        )

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def list_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def describe_for_prompt(self) -> str:
        lines: list[str] = []
        for spec in self._tools.values():
            lines.append(f"- {spec.name}: {spec.description}")
        return "\n".join(lines)

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        spec = self._tools.get(name)
        if spec is None:
            return f"Error: unknown tool {name!r}"
        try:
            return spec.handler(**arguments)
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:
            return f"Error running {name}: {e}"

    def _search_kb(self, query: str = "") -> str:
        hits = self._kb.search(query, top_k=self._rag_top_k)
        return format_chunks_for_prompt(hits)

    def _search_web(self, query: str = "") -> str:
        if self._web_search is None:
            return "Error: web search is not configured"
        return self._web_search.search(query)

    def _get_time(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _remember(self, key: str, value: str) -> str:
        self._session_memory[key.strip()] = value.strip()
        return f"Stored {key!r} for this session."

    def _recall(self, key: str) -> str:
        val = self._session_memory.get(key.strip())
        if val is None:
            return f"No value stored for key {key!r}."
        return val
