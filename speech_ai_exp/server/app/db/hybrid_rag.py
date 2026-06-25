from __future__ import annotations

from pathlib import Path

from app.agent.rag import KnowledgeBase, RetrievedChunk, chunks_from_root
from app.db.vector_store import VectorKnowledgeStore


class HybridKnowledgeBase:
    """BM25 file index + optional Chroma vector search."""

    def __init__(
        self,
        bm25: KnowledgeBase,
        vector: VectorKnowledgeStore | None,
        *,
        rag_backend: str = "hybrid",
    ) -> None:
        self._bm25 = bm25
        self._vector = vector
        self._requested_backend = rag_backend
        self._rag_backend = rag_backend
        self._chunks_loaded = 0

    @property
    def chunk_count(self) -> int:
        return self._chunks_loaded

    @property
    def index_path(self) -> Path:
        return self._bm25.index_path

    @property
    def loaded_from_index(self) -> bool:
        return self._bm25.loaded_from_index

    @property
    def rag_backend(self) -> str:
        return self._rag_backend

    @property
    def vector_doc_count(self) -> int:
        return self._vector.doc_count if self._vector else 0

    def load(self) -> int:
        n_bm25 = self._bm25.load()
        n_vec = 0
        if self._vector:
            self._vector.connect()
            n_vec = self._vector.doc_count
        if self._requested_backend in ("vector", "hybrid") and (not self._vector or not self._vector.available):
            self._rag_backend = "bm25"
        else:
            self._rag_backend = self._requested_backend
        self._chunks_loaded = n_bm25 + n_vec
        return self._chunks_loaded

    def search(self, query: str, *, top_k: int = 3) -> list[RetrievedChunk]:
        if self._rag_backend == "bm25":
            return self._bm25.search(query, top_k=top_k)
        if self._rag_backend == "vector" and self._vector:
            return self._vector.search(query, top_k=top_k)
        bm25_hits = self._bm25.search(query, top_k=top_k)
        if not self._vector or not self._vector.available:
            return bm25_hits
        vec_hits = self._vector.search(query, top_k=top_k)
        return _merge_hits(bm25_hits, vec_hits, top_k=top_k)


def build_vector_documents(
    commerce_docs: list[tuple[str, str, str]],
    knowledge_root: Path,
    *,
    chunk_chars: int = 600,
) -> list[tuple[str, str, str]]:
    docs = list(commerce_docs)
    for chunk in chunks_from_root(knowledge_root, chunk_chars=chunk_chars):
        docs.append((chunk.doc_id, chunk.source, chunk.text))
    return docs


def _merge_hits(
    bm25_hits: list[RetrievedChunk],
    vec_hits: list[RetrievedChunk],
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    ranked: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for rank, hit in enumerate(bm25_hits):
        key = hit.chunk.doc_id
        scores[key] = scores.get(key, 0.0) + hit.score + (top_k - rank) * 0.05
        ranked[key] = hit

    for rank, hit in enumerate(vec_hits):
        key = hit.chunk.doc_id
        scores[key] = scores.get(key, 0.0) + hit.score + (top_k - rank) * 0.05
        if key not in ranked or hit.score > ranked[key].score:
            ranked[key] = hit

    merged = list(ranked.values())
    merged.sort(key=lambda h: scores.get(h.chunk.doc_id, h.score), reverse=True)
    for hit in merged:
        hit.score = scores.get(hit.chunk.doc_id, hit.score)
    return merged[:top_k]
