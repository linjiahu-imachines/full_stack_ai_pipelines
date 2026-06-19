from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class WebSearchConfig:
    enabled: bool = False
    provider: str = "tavily"
    api_key: str = ""
    max_results: int = 5
    search_depth: str = "basic"  # basic | advanced
    auto_search: bool = True  # run Tavily for weather/news/current queries before LLM

    @classmethod
    def from_yaml(cls, raw: dict[str, Any] | None) -> WebSearchConfig:
        raw = raw if isinstance(raw, dict) else {}
        ws = raw.get("web_search") if isinstance(raw.get("web_search"), dict) else {}
        if not ws and raw.get("web_search_enabled") is not None:
            ws = raw
        enabled = str(os.environ.get("WEB_SEARCH_ENABLED", ws.get("enabled", False))).lower() in (
            "1",
            "true",
            "yes",
        )
        api_key = os.environ.get("TAVILY_API_KEY", ws.get("tavily_api_key", ""))
        auto = ws.get("auto_search", True)
        return cls(
            enabled=enabled and bool(api_key.strip()),
            provider=str(ws.get("provider", "tavily")),
            api_key=api_key.strip(),
            max_results=int(ws.get("max_results", 5)),
            search_depth=str(ws.get("search_depth", "basic")),
            auto_search=bool(auto),
        )


_AUTO_WEB_PATTERNS = re.compile(
    r"\b("
    r"weather|forecast|temperature|rain|snow|humidity|"
    r"today|tonight|tomorrow|right now|currently|current|latest|live|real[- ]?time|"
    r"news|headline|stock price|exchange rate|"
    r"who is the (ceo|president|prime minister)"
    r")\b",
    re.I,
)


def should_auto_web_search(query: str) -> bool:
    """Heuristic: queries that need live public data (weather, news, etc.)."""
    return bool(_AUTO_WEB_PATTERNS.search(query))


def format_web_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "(no web results)"
    parts: list[str] = []
    for i, hit in enumerate(results, start=1):
        title = hit.get("title") or "(no title)"
        url = hit.get("url") or ""
        snippet = (hit.get("content") or hit.get("snippet") or "").strip()
        parts.append(f"[{i}] {title}\nurl={url}\n{snippet}")
    return "\n\n".join(parts)


class TavilyWebSearch:
    """Tavily Search API — https://docs.tavily.com/"""

    _URL = "https://api.tavily.com/search"

    def __init__(self, cfg: WebSearchConfig) -> None:
        if not cfg.api_key:
            raise ValueError("Tavily web search requires TAVILY_API_KEY or agent.web_search.tavily_api_key")
        self._cfg = cfg

    def search(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "Error: empty search query"

        payload = {
            "api_key": self._cfg.api_key,
            "query": query,
            "search_depth": self._cfg.search_depth,
            "max_results": self._cfg.max_results,
            "include_answer": False,
        }
        req = Request(
            self._URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return f"Error: Tavily HTTP {e.code}: {body[:500]}"
        except URLError as e:
            return f"Error: Tavily unreachable: {e}"
        except json.JSONDecodeError as e:
            return f"Error: invalid Tavily response: {e}"

        results = data.get("results") or []
        if not results and data.get("answer"):
            return f"Summary: {data['answer']}"
        return format_web_results(results)


def make_web_search(cfg: WebSearchConfig) -> TavilyWebSearch | None:
    if not cfg.enabled:
        return None
    if cfg.provider != "tavily":
        raise ValueError(f"Unsupported web search provider: {cfg.provider!r}")
    return TavilyWebSearch(cfg)
