"""Retrieval-augmented generation components."""

from .chunker import TextChunk, iter_text_chunks, split_text
from .embedder import LocalEmbedder
from .vector_store import SQLiteVectorStore, SearchResult
from .text_store import SQLiteTextStore, TextSearchResult

__all__ = [
    "LocalEmbedder",
    "SQLiteVectorStore",
    "SQLiteTextStore",
    "SearchResult",
    "TextSearchResult",
    "TextChunk",
    "iter_text_chunks",
    "split_text",
]
