"""Single-file SQLite vector storage for local lecture retrieval."""

from __future__ import annotations

import json
import heapq
from itertools import zip_longest
import math
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .chunker import TextChunk


@dataclass(frozen=True)
class SearchResult:
    text: str
    distance: float
    metadata: Dict[str, Any]


class SQLiteVectorStore:
    """Store all lecture collections in one portable SQLite database file."""

    def __init__(self, database_path: Path, collection_name: str = "lass_documents") -> None:
        self.database_path = database_path.expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    fingerprint TEXT NOT NULL DEFAULT ''
                )
                """
            )
            collection_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(collections)"
                ).fetchall()
            }
            if "fingerprint" not in collection_columns:
                connection.execute(
                    "ALTER TABLE collections ADD COLUMN fingerprint TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    collection_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (collection_name, chunk_id),
                    FOREIGN KEY (collection_name) REFERENCES collections(name)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_vectors_collection "
                "ON vectors(collection_name)"
            )

    @staticmethod
    def _encode_embedding(embedding: Sequence[float]) -> bytes:
        values = [float(value) for value in embedding]
        if not values:
            raise ValueError("embedding cannot be empty")
        return struct.pack(f"<{len(values)}f", *values)

    @staticmethod
    def _decode_embedding(blob: bytes, dimension: int) -> tuple[float, ...]:
        if dimension <= 0 or len(blob) != dimension * 4:
            raise ValueError("stored embedding has an invalid dimension")
        return struct.unpack(f"<{dimension}f", blob)

    @staticmethod
    def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("query and stored embeddings must have the same dimension")
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 1.0
        similarity = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
        return 1.0 - similarity

    def upsert(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        fingerprint: str = "",
    ) -> None:
        self.upsert_stream(chunks, embeddings, fingerprint=fingerprint)

    def upsert_stream(
        self,
        chunks: Iterable[TextChunk],
        embeddings: Iterable[Sequence[float]],
        *,
        fingerprint: str = "",
        batch_size: int = 16,
    ) -> None:
        """Replace a collection without materializing document vectors in RAM."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        sentinel = object()
        dimension = None
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO collections(name, created_at, fingerprint)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET fingerprint = excluded.fingerprint
                """,
                (self.collection_name, now, fingerprint),
            )
            connection.execute(
                "DELETE FROM vectors WHERE collection_name = ?",
                (self.collection_name,),
            )
            rows = []
            for chunk, embedding in zip_longest(chunks, embeddings, fillvalue=sentinel):
                if chunk is sentinel or embedding is sentinel:
                    raise ValueError("chunks and embeddings must have the same length")
                current_dimension = len(embedding)
                if current_dimension <= 0 or (
                    dimension is not None and current_dimension != dimension
                ):
                    raise ValueError("all embeddings must have the same non-zero dimension")
                dimension = current_dimension
                rows.append(
                    (
                        self.collection_name,
                        f"chunk-{chunk.index}",
                        chunk.text,
                        self._encode_embedding(embedding),
                        current_dimension,
                        json.dumps(
                            {"index": chunk.index, "start": chunk.start, "end": chunk.end},
                            ensure_ascii=False,
                        ),
                    )
                )
                if len(rows) >= batch_size:
                    self._insert_rows(connection, rows)
                    rows.clear()
            if rows:
                self._insert_rows(connection, rows)

    @staticmethod
    def _insert_rows(connection: sqlite3.Connection, rows: list[tuple]) -> None:
        connection.executemany(
            """
            INSERT INTO vectors (
                collection_name, chunk_id, document, embedding, dimension,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def get_fingerprint(self) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT fingerprint FROM collections WHERE name = ?",
                (self.collection_name,),
            ).fetchone()
        return str(row["fingerprint"]) if row is not None else ""

    def count(self) -> int:
        """Return the number of indexed chunks without loading any vector rows."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM vectors WHERE collection_name = ?",
                (self.collection_name,),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def search(self, query_embedding: Sequence[float], limit: int = 6) -> List[SearchResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        query = tuple(float(value) for value in query_embedding)
        if not query:
            raise ValueError("query_embedding cannot be empty")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document, embedding, dimension, metadata_json
                FROM vectors WHERE collection_name = ?
                """,
                (self.collection_name,),
            )
            best = []
            for index, row in enumerate(rows):
                result = SearchResult(
                    text=row["document"],
                    distance=self._cosine_distance(
                        query,
                        self._decode_embedding(row["embedding"], row["dimension"]),
                    ),
                    metadata=json.loads(row["metadata_json"]),
                )
                entry = (-result.distance, index, result)
                if len(best) < limit:
                    heapq.heappush(best, entry)
                elif result.distance < -best[0][0]:
                    heapq.heapreplace(best, entry)
        return sorted((entry[2] for entry in best), key=lambda result: result.distance)

    @staticmethod
    def delete_collection(database_path: Path, collection_name: str) -> bool:
        path = database_path.expanduser().resolve()
        if not collection_name or not path.is_file():
            return False
        with sqlite3.connect(str(path), timeout=10) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            cursor = connection.execute(
                "DELETE FROM collections WHERE name = ?", (collection_name,)
            )
        return cursor.rowcount > 0
