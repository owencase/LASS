"""Dependency-free text chunking with overlap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List


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


def iter_text_chunks(
    text: str, chunk_size: int = 1200, overlap: int = 150
) -> Iterator[TextChunk]:
    """Yield natural text windows without retaining a document-sized chunk list."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    text_length = len(text)
    first = 0
    while first < text_length and text[first].isspace():
        first += 1
    last = text_length
    while last > first and text[last - 1].isspace():
        last -= 1
    if first >= last:
        return

    index = 0
    start = first
    while start < last:
        limit = min(start + chunk_size, last)
        end = limit
        if limit < last:
            minimum = start + max(chunk_size // 2, 1)
            end = _natural_break(text, start, limit, minimum)

        raw_chunk = text[start:end]
        content = raw_chunk.strip()
        if content:
            yield TextChunk(index, content, start, end)
            index += 1
        if end >= last:
            break
        start = max(end - overlap, start + 1)


def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[TextChunk]:
    """Compatibility wrapper for callers that explicitly need a materialized list."""
    return list(iter_text_chunks(text, chunk_size=chunk_size, overlap=overlap))
