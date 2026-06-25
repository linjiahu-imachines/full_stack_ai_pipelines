from __future__ import annotations

import logging
from pathlib import Path

from app.agent.rag import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorKnowledgeStore:
    """Chroma-backed semantic retrieval."""

    def __init__(
        self,
        *,
        persist_path: Path,
        collection_name: str,
        embed_model: str,
    ) -> None:
        self._persist_path = persist_path.resolve()
        self._collection_name = collection_name
        self._embed_model = embed_model
        self._client = None
        self._collection = None
        self._available = False
        self._doc_count = 0

    @property
    def available(self) -> bool:
        return self._available

    @property
    def doc_count(self) -> int:
        return self._doc_count

    @property
    def persist_path(self) -> Path:
        return self._persist_path

    def connect(self) -> bool:
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError:
            logger.warning("chromadb not installed; vector RAG disabled")
            return False

        self._persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_path))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self._embed_model)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._doc_count = int(self._collection.count())
        self._available = True
        logger.info(
            "Vector store ready | path=%s | collection=%s | docs=%s",
            self._persist_path,
            self._collection_name,
            self._doc_count,
        )
        return True

    def rebuild(self, documents: list[tuple[str, str, str]], *, batch_size: int = 32) -> int:
        """Replace collection contents with (doc_id, source, text) tuples."""
        if not self.connect():
            return 0
        assert self._collection is not None
        if self._client is not None:
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:
                pass
            from chromadb.utils import embedding_functions

            ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self._embed_model)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, str]] = []
        for doc_id, source, text in documents:
            text = text.strip()
            if not text:
                continue
            ids.append(doc_id)
            texts.append(text)
            metadatas.append({"source": source})

        for i in range(0, len(ids), batch_size):
            self._collection.add(
                ids=ids[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        self._doc_count = len(ids)
        return self._doc_count

    def search(self, query: str, *, top_k: int = 3) -> list[RetrievedChunk]:
        if not self._available or self._collection is None or not query.strip():
            return []
        if self._doc_count == 0:
            return []
        try:
            result = self._collection.query(query_texts=[query.strip()], n_results=min(top_k, self._doc_count))
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            return []

        hits: list[RetrievedChunk] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for doc_id, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
            if not text:
                continue
            source = str((meta or {}).get("source") or "vector")
            score = max(0.0, 1.0 - float(dist or 1.0))
            hits.append(
                RetrievedChunk(
                    chunk=DocumentChunk(doc_id=str(doc_id), text=str(text), source=source),
                    score=score,
                )
            )
        return hits
