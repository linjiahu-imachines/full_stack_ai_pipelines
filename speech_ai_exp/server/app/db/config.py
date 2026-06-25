from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_POSTGRES_URL = (
    "postgresql+psycopg://horizon:horizon@127.0.0.1:5432/horizon_store"
)


@dataclass
class DatabaseConfig:
    """PostgreSQL + Chroma vector store settings for the agent."""

    enabled: bool = True
    sql_url: str = DEFAULT_POSTGRES_URL
    vector_path: str = "data/chroma"
    vector_collection: str = "horizon_kb"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_backend: str = "hybrid"  # bm25 | vector | hybrid

    @classmethod
    def from_yaml(cls, raw: dict[str, Any] | None, *, server_root: Path) -> DatabaseConfig:
        raw = raw if isinstance(raw, dict) else {}
        db = raw.get("database") if isinstance(raw.get("database"), dict) else {}
        if not db and raw.get("sql_url"):
            db = raw

        enabled = str(os.environ.get("DATABASE_ENABLED", db.get("enabled", True))).lower() not in (
            "0",
            "false",
            "no",
        )
        sql_url = os.environ.get("DATABASE_SQL_URL", db.get("sql_url", DEFAULT_POSTGRES_URL))
        vector_path = os.environ.get("DATABASE_VECTOR_PATH", db.get("vector_path", "data/chroma"))
        vector_collection = str(
            os.environ.get("DATABASE_VECTOR_COLLECTION", db.get("vector_collection", "horizon_kb"))
        )
        embed_model = str(
            os.environ.get("DATABASE_EMBED_MODEL", db.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"))
        )
        rag_backend = str(os.environ.get("DATABASE_RAG_BACKEND", db.get("rag_backend", "hybrid")))

        vec = Path(vector_path)
        if not vec.is_absolute():
            vector_path = str((server_root / vec).resolve())

        return cls(
            enabled=enabled,
            sql_url=sql_url,
            vector_path=vector_path,
            vector_collection=vector_collection,
            embed_model=embed_model,
            rag_backend=rag_backend,
        )
