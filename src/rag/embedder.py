"""Lazy-loaded local sentence embedding model."""

from __future__ import annotations

from typing import Any, List, Sequence


class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for embeddings. "
                    "Install dependencies with `pip install -r requirements.txt`."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("Query text must not be empty")
        return self.embed_documents([text])[0]
