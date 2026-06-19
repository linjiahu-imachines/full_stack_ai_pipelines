#!/usr/bin/env python3
"""Build a persisted RAG chunk index from server/data/knowledge/.

No server or staged_voice install required — only imports app.agent.rag.
Optional: activate project venv if you use a different Python layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_ROOT))

from app.agent.rag import (  # noqa: E402
    build_and_save_index,
    collect_source_fingerprints,
    default_index_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk knowledge files and write data/knowledge_index.json",
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=SERVER_ROOT / "data" / "knowledge",
        help="Directory containing .md / .txt / .rst files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Index JSON path (default: sibling of knowledge dir)",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=600,
        help="Max characters per chunk (must match server config)",
    )
    args = parser.parse_args()

    knowledge_dir = args.knowledge_dir.expanduser().resolve()
    if not knowledge_dir.is_dir():
        print(f"Error: knowledge dir not found: {knowledge_dir}", file=sys.stderr)
        return 1

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_index_path(knowledge_dir)
    )

    sources = collect_source_fingerprints(knowledge_dir)
    if not sources:
        print(f"Warning: no .md/.txt/.rst files under {knowledge_dir}", file=sys.stderr)

    n = build_and_save_index(knowledge_dir, output, chunk_chars=args.chunk_chars)
    print(f"Wrote {n} chunks from {len(sources)} file(s)")
    print(f"  knowledge: {knowledge_dir}")
    print(f"  index:     {output}")
    for src in sources:
        print(f"    - {src['path']} ({src['size']} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
