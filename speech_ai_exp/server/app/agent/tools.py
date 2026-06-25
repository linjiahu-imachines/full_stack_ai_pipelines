from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from app.agent.rag import KnowledgeBase, format_chunks_for_prompt
from app.agent.web_search import TavilyWebSearch, WebSearchConfig
from app.db.repositories import CommerceRepository


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
        commerce: CommerceRepository | None = None,
    ) -> None:
        self._kb = kb
        self._rag_top_k = rag_top_k
        self._web_search = web_search
        self._commerce = commerce
        self._session_memory: dict[str, str] = {}
        self._tools: dict[str, ToolSpec] = {}
        self._register_defaults()

    @property
    def commerce_enabled(self) -> bool:
        return self._commerce is not None

    @property
    def web_search_enabled(self) -> bool:
        return self._web_search is not None

    def bind_session_memory(self, memory: dict[str, str]) -> None:
        self._session_memory = memory

    def _register_defaults(self) -> None:
        if self._commerce is not None:
            self.register(
                ToolSpec(
                    name="lookup_customer",
                    description=(
                        "Look up a customer profile by customer_id (e.g. C-001257, C-19382), "
                        "email, or name."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "email": {"type": "string"},
                            "name": {"type": "string"},
                        },
                    },
                    handler=self._lookup_customer,
                )
            )
            self.register(
                ToolSpec(
                    name="lookup_order",
                    description=(
                        "Look up order by order ID or spoken order number (e.g. 784921, 918273, "
                        "ORD-784921). Returns status, shipments, and ETA."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "Order ID"},
                        },
                        "required": ["order_id"],
                    },
                    handler=self._lookup_order,
                )
            )
            self.register(
                ToolSpec(
                    name="get_active_order",
                    description=(
                        "Return the customer's current active order. Use when caller asks about "
                        "'my order' without a number. Pass customer_id, email, or name from context."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "email": {"type": "string"},
                            "name": {"type": "string"},
                        },
                    },
                    handler=self._get_active_order,
                )
            )
            self.register(
                ToolSpec(
                    name="list_customer_orders",
                    description="List recent orders for a customer.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "email": {"type": "string"},
                            "name": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    },
                    handler=self._list_customer_orders,
                )
            )
            self.register(
                ToolSpec(
                    name="lookup_invoice",
                    description="Look up latest or specific invoice by customer ID, email, or invoice ID.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "email": {"type": "string"},
                            "invoice_id": {"type": "string"},
                            "latest": {"type": "boolean"},
                        },
                    },
                    handler=self._lookup_invoice,
                )
            )
            self.register(
                ToolSpec(
                    name="lookup_payment_status",
                    description="Check whether payment was captured for a customer email or order.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "customer_id": {"type": "string"},
                            "order_id": {"type": "string"},
                        },
                    },
                    handler=self._lookup_payment_status,
                )
            )
            self.register(
                ToolSpec(
                    name="list_open_orders",
                    description="List open/in-progress orders for a customer.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "email": {"type": "string"},
                            "name": {"type": "string"},
                        },
                    },
                    handler=self._list_open_orders,
                )
            )
            self.register(
                ToolSpec(
                    name="list_delayed_orders",
                    description="List delayed orders for a customer.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "email": {"type": "string"},
                            "name": {"type": "string"},
                        },
                    },
                    handler=self._list_delayed_orders,
                )
            )
            self.register(
                ToolSpec(
                    name="lookup_licenses",
                    description="List software licenses and count how many are active.",
                    parameters={
                        "type": "object",
                        "properties": {"customer_id": {"type": "string"}, "email": {"type": "string"}},
                    },
                    handler=self._lookup_licenses,
                )
            )
            self.register(
                ToolSpec(
                    name="lookup_subscription",
                    description="Subscription renewal date, amount, and downgrade impact.",
                    parameters={
                        "type": "object",
                        "properties": {"customer_id": {"type": "string"}, "email": {"type": "string"}},
                    },
                    handler=self._lookup_subscription,
                )
            )
            self.register(
                ToolSpec(
                    name="compare_support_plans",
                    description="Compare Premium vs Enterprise support plans.",
                    parameters={"type": "object", "properties": {}},
                    handler=self._compare_support_plans,
                )
            )
        self.register(
            ToolSpec(
                name="search_knowledge_base",
                description=(
                    "Semantic search over product docs, policies, troubleshooting, error codes, "
                    "compliance (GDPR), and support workflows. Use for product features, returns, "
                    "refunds, warranty, error codes, and plan comparisons."
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
                        "Search the public web for current weather, news, stock prices, "
                        "and other live facts not in the internal knowledge base. "
                        "Required for weather and news questions."
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

    def _lookup_customer(self, customer_id: str = "", email: str = "", name: str = "") -> str:
        if self._commerce is None:
            return "Error: customer database is not configured"
        return self._commerce.lookup_customer(customer_id=customer_id, email=email, name=name)

    def _lookup_order(self, order_id: str = "") -> str:
        if self._commerce is None:
            return "Error: order database is not configured"
        return self._commerce.lookup_order(order_id=order_id)

    def _lookup_invoice(
        self,
        customer_id: str = "",
        email: str = "",
        invoice_id: str = "",
        latest: bool = True,
    ) -> str:
        if self._commerce is None:
            return "Error: database is not configured"
        return self._commerce.lookup_invoice(
            customer_id=customer_id, email=email, invoice_id=invoice_id, latest=latest
        )

    def _lookup_payment_status(self, email: str = "", customer_id: str = "", order_id: str = "") -> str:
        if self._commerce is None:
            return "Error: database is not configured"
        return self._commerce.lookup_payment_status(email=email, customer_id=customer_id, order_id=order_id)

    def _list_open_orders(self, customer_id: str = "", email: str = "", name: str = "") -> str:
        if self._commerce is None:
            return "Error: database is not configured"
        return self._commerce.list_open_orders(customer_id=customer_id, email=email, name=name)

    def _list_delayed_orders(self, customer_id: str = "", email: str = "", name: str = "") -> str:
        if self._commerce is None:
            return "Error: database is not configured"
        return self._commerce.list_delayed_orders(customer_id=customer_id, email=email, name=name)

    def _lookup_licenses(self, customer_id: str = "", email: str = "") -> str:
        if self._commerce is None:
            return "Error: database is not configured"
        return self._commerce.lookup_licenses(customer_id=customer_id, email=email)

    def _lookup_subscription(self, customer_id: str = "", email: str = "") -> str:
        if self._commerce is None:
            return "Error: database is not configured"
        return self._commerce.lookup_subscription(customer_id=customer_id, email=email)

    def _compare_support_plans(self) -> str:
        if self._commerce is None:
            return "Error: database is not configured"
        return self._commerce.compare_support_plans()

    def _get_active_order(self, customer_id: str = "", email: str = "", name: str = "") -> str:
        if self._commerce is None:
            return "Error: order database is not configured"
        return self._commerce.get_active_order(customer_id=customer_id, email=email, name=name)

    def _list_customer_orders(
        self, customer_id: str = "", email: str = "", name: str = "", limit: int = 5
    ) -> str:
        if self._commerce is None:
            return "Error: order database is not configured"
        return self._commerce.list_customer_orders(
            customer_id=customer_id, email=email, name=name, limit=limit
        )

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
