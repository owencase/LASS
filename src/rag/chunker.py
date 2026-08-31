"""Dependency-free text chunking with overlap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start: int
    end: int


def _natural_break(text: str, start: int, limit: int, minimum: int) -> int:
    for separator in ("\n\n", "\n", ". ", "다. ", "요. ", " "):
        position = text.rfind(separator, minimum, limit)
        if position != -1:
            return position + len(separator)
    return limit


def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[TextChunk]:
    """Split text near natural boundaries while retaining a small overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    normalized = text.strip()
    if not normalized:
        return []

    chunks: List[TextChunk] = []
    start = 0
    text_length = len(normalized)
    while start < text_length:
        limit = min(start + chunk_size, text_length)
        end = limit
        if limit < text_length:
            minimum = start + max(chunk_size // 2, 1)
            end = _natural_break(normalized, start, limit, minimum)

        raw_chunk = normalized[start:end]
        content = raw_chunk.strip()
        if content:
            chunks.append(TextChunk(len(chunks), content, start, end))
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)

    return chunks
