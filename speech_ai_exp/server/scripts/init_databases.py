#!/usr/bin/env python3
"""Initialize PostgreSQL commerce DB and Chroma vector index for agent RAG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from app.db.config import DatabaseConfig  # noqa: E402
from app.db.hybrid_rag import build_vector_documents  # noqa: E402
from app.db.repositories import CommerceRepository  # noqa: E402
from app.db.seed import seed_demo_data  # noqa: E402
from app.db.session import get_session_factory, init_sql_schema, reset_sql_schema  # noqa: E402
from app.db.vector_store import VectorKnowledgeStore  # noqa: E402
from app.env_file import load_server_env  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize Horizon Store PostgreSQL + vector databases")
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=SERVER_ROOT / "data" / "knowledge",
        help="Knowledge text files for vector indexing",
    )
    parser.add_argument("--force-vector", action="store_true", help="Rebuild Chroma collection")
    parser.add_argument("--sql-only", action="store_true", help="Skip vector index build")
    parser.add_argument(
        "--reset-sql",
        action="store_true",
        help="Drop and recreate PostgreSQL tables before seeding (destructive)",
    )
    args = parser.parse_args()

    load_server_env(SERVER_ROOT)
    db_cfg = DatabaseConfig.from_yaml({}, server_root=SERVER_ROOT)
    print(f"PostgreSQL: {db_cfg.sql_url}")

    try:
        if args.reset_sql:
            print("Resetting PostgreSQL schema...")
            reset_sql_schema(db_cfg.sql_url)
        else:
            init_sql_schema(db_cfg.sql_url)
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        print("Start Postgres first, e.g.: cd server && docker compose up -d postgres")
        return 1
    session_factory = get_session_factory(db_cfg.sql_url)
    with session_factory() as session:
        seed_demo_data(session)
    print("Seeded customers, orders, invoices, licenses, products, policies, and KB articles.")

    repo = CommerceRepository(session_factory)
    if args.sql_only:
        print("SQL-only mode; skipping vector index.")
        return 0

    vector = VectorKnowledgeStore(
        persist_path=Path(db_cfg.vector_path),
        collection_name=db_cfg.vector_collection,
        embed_model=db_cfg.embed_model,
    )
    if not vector.connect():
        print("Vector dependencies missing. Install with: pip install -e '.[vector]'")
        return 1
    if vector.doc_count > 0 and not args.force_vector:
        print(f"Vector index already has {vector.doc_count} documents at {db_cfg.vector_path}")
        print("Use --force-vector to rebuild.")
        return 0

    docs = build_vector_documents(repo.iter_vector_documents(), args.knowledge_dir.resolve())
    print(f"Building vector index with {len(docs)} documents...")
    n = vector.rebuild(docs)
    print(f"Vector index ready: {n} documents in {db_cfg.vector_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
