"""Persistent local ChromaDB storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .chunker import TextChunk


@dataclass(frozen=True)
class SearchResult:
    text: str
    distance: float
    metadata: Dict[str, Any]


class ChromaVectorStore:
    def __init__(self, persist_directory: Path, collection_name: str = "lass_documents") -> None:
        self.persist_directory = persist_directory.expanduser().resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required for vector storage. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: Sequence[TextChunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self._collection.upsert(
            ids=[f"chunk-{chunk.index}" for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
            metadatas=[
                {"index": chunk.index, "start": chunk.start, "end": chunk.end}
                for chunk in chunks
            ],
        )

    def search(self, query_embedding: Sequence[float], limit: int = 6) -> List[SearchResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            SearchResult(text=document, metadata=metadata or {}, distance=float(distance))
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]
