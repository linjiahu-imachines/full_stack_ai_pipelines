from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DocumentChunk:
    doc_id: str
    text: str
    source: str


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


_WORD = re.compile(r"[a-z0-9]+", re.I)
SUPPORTED_SUFFIXES = {".md", ".txt", ".rst"}
INDEX_VERSION = 1


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def default_index_path(knowledge_root: Path) -> Path:
    """Default: sibling of the knowledge folder, e.g. data/knowledge_index.json."""
    return knowledge_root.parent / "knowledge_index.json"


def discover_knowledge_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            paths.append(path)
    return paths


def source_fingerprint(path: Path, *, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def collect_source_fingerprints(root: Path) -> list[dict[str, Any]]:
    return [source_fingerprint(p, root=root) for p in discover_knowledge_paths(root)]


def chunks_from_root(root: Path, *, chunk_chars: int = 600) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in discover_knowledge_paths(root):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        doc_id = path.relative_to(root).as_posix()
        for i, piece in enumerate(_split_text(text, chunk_chars)):
            chunks.append(
                DocumentChunk(doc_id=f"{doc_id}#{i}", text=piece, source=doc_id)
            )
    return chunks


def build_index_payload(root: Path, *, chunk_chars: int = 600) -> dict[str, Any]:
    root = root.resolve()
    chunks = chunks_from_root(root, chunk_chars=chunk_chars)
    return {
        "version": INDEX_VERSION,
        "chunk_chars": chunk_chars,
        "knowledge_dir": str(root),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sources": collect_source_fingerprints(root),
        "chunks": [asdict(c) for c in chunks],
    }


def index_is_fresh(payload: dict[str, Any], root: Path, *, chunk_chars: int) -> bool:
    if payload.get("version") != INDEX_VERSION:
        return False
    if payload.get("chunk_chars") != chunk_chars:
        return False
    indexed = {s["path"]: s for s in payload.get("sources") or []}
    current = {s["path"]: s for s in collect_source_fingerprints(root)}
    return indexed == current


def build_and_save_index(
    knowledge_root: Path,
    index_path: Path,
    *,
    chunk_chars: int = 600,
) -> int:
    payload = build_index_payload(knowledge_root, chunk_chars=chunk_chars)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(payload["chunks"])


class KnowledgeBase:
    """Lightweight local RAG index (no LangChain / no embeddings)."""

    def __init__(
        self,
        root: Path,
        *,
        chunk_chars: int = 600,
        index_path: Path | None = None,
    ) -> None:
        self._root = root.resolve()
        self._chunk_chars = chunk_chars
        self._index_path = (
            index_path.resolve() if index_path is not None else default_index_path(self._root)
        )
        self._chunks: list[DocumentChunk] = []
        self._loaded_from_index = False

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def index_path(self) -> Path:
        return self._index_path

    @property
    def loaded_from_index(self) -> bool:
        return self._loaded_from_index

    def load(self) -> int:
        self._chunks.clear()
        self._loaded_from_index = False
        if not self._root.is_dir():
            return 0

        if self._index_path.is_file():
            try:
                payload = json.loads(self._index_path.read_text(encoding="utf-8"))
                if index_is_fresh(payload, self._root, chunk_chars=self._chunk_chars):
                    self._chunks = [
                        DocumentChunk(
                            doc_id=str(c["doc_id"]),
                            text=str(c["text"]),
                            source=str(c["source"]),
                        )
                        for c in payload.get("chunks") or []
                    ]
                    self._loaded_from_index = True
                    return len(self._chunks)
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                pass

        self._chunks = chunks_from_root(self._root, chunk_chars=self._chunk_chars)
        return len(self._chunks)

    def search(self, query: str, *, top_k: int = 3) -> list[RetrievedChunk]:
        if not self._chunks or not query.strip():
            return []
        q_tf = _term_freq(_tokenize(query))
        if not q_tf:
            return []
        scored: list[RetrievedChunk] = []
        for chunk in self._chunks:
            c_tf = _term_freq(_tokenize(chunk.text))
            score = _bm25_score(q_tf, c_tf, len(_tokenize(chunk.text)))
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def _split_text(text: str, max_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if buf:
                out.append(buf.strip())
                buf = ""
            for i in range(0, len(para), max_chars):
                out.append(para[i : i + max_chars].strip())
            continue
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                out.append(buf.strip())
            buf = para
    if buf:
        out.append(buf.strip())
    return out


def _term_freq(tokens: list[str]) -> dict[str, int]:
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


def _bm25_score(
    q_tf: dict[str, int],
    doc_tf: dict[str, int],
    doc_len: int,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    avg_dl: float = 200.0,
) -> float:
    score = 0.0
    for term, qf in q_tf.items():
        tf = doc_tf.get(term, 0)
        if tf == 0:
            continue
        idf = math.log(1.0 + (1.0 / (tf + 0.5)))
        denom = tf + k1 * (1.0 - b + b * doc_len / avg_dl)
        score += idf * (tf * (k1 + 1.0)) / denom * qf
    return score


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no matching knowledge base passages)"
    parts: list[str] = []
    for i, hit in enumerate(chunks, start=1):
        parts.append(f"[{i}] source={hit.chunk.source} score={hit.score:.2f}\n{hit.chunk.text}")
    return "\n\n".join(parts)
