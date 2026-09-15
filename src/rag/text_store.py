"""Disk-backed sparse retrieval that does not load an embedding model."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .chunker import TextChunk


_TERM = re.compile(r"[0-9A-Za-z가-힣]{2,}")


@dataclass(frozen=True)
class TextSearchResult:
    text: str
    score: float
    metadata: Dict[str, Any]


class SQLiteTextStore:
    """Use SQLite FTS5/BM25 so retrieval size grows on disk, not in RAM."""

    def __init__(self, database_path: Path, collection_name: str) -> None:
        self.database_path = database_path.expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute("PRAGMA cache_size = -2048")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS text_collections (
                    name TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    fingerprint TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS text_chunks USING fts5(
                    collection_name UNINDEXED,
                    chunk_id UNINDEXED,
                    document,
                    metadata_json UNINDEXED,
                    tokenize='unicode61 remove_diacritics 2'
                )
                """
            )

    def upsert_stream(
        self,
        chunks: Iterable[TextChunk],
        *,
        fingerprint: str = "",
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO text_collections(name, created_at, fingerprint)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    created_at = excluded.created_at,
                    fingerprint = excluded.fingerprint
                """,
                (self.collection_name, now, fingerprint),
            )
            connection.execute(
                "DELETE FROM text_chunks WHERE collection_name = ?",
                (self.collection_name,),
            )
            rows = []
            for chunk in chunks:
                rows.append(
                    (
                        self.collection_name,
                        f"chunk-{chunk.index}",
                        chunk.text,
                        json.dumps(
                            {"index": chunk.index, "start": chunk.start, "end": chunk.end},
                            ensure_ascii=False,
                        ),
                    )
                )
                if len(rows) >= batch_size:
                    connection.executemany(
                        "INSERT INTO text_chunks VALUES (?, ?, ?, ?)", rows
                    )
                    rows.clear()
            if rows:
                connection.executemany(
                    "INSERT INTO text_chunks VALUES (?, ?, ?, ?)", rows
                )

    def get_fingerprint(self) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT fingerprint FROM text_collections WHERE name = ?",
                (self.collection_name,),
            ).fetchone()
        return str(row["fingerprint"]) if row is not None else ""

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM text_chunks WHERE collection_name = ?",
                (self.collection_name,),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    @staticmethod
    def _match_query(query: str) -> str:
        # OR is deliberate: Korean postpositions often make exact multi-term
        # intersection too strict, while quoted terms keep FTS syntax safe.
        terms = list(dict.fromkeys(_TERM.findall(query.casefold())))[:24]
        return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)

    def search(self, query: str, limit: int = 6) -> List[TextSearchResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        match_query = self._match_query(query)
        with self._connect() as connection:
            rows = []
            if match_query:
                rows = connection.execute(
                    """
                    SELECT document, metadata_json, bm25(text_chunks) AS score
                    FROM text_chunks
                    WHERE text_chunks MATCH ? AND collection_name = ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (match_query, self.collection_name, limit),
                ).fetchall()
            if not rows:
                rows = connection.execute(
                    """
                    SELECT document, metadata_json, 0.0 AS score
                    FROM text_chunks
                    WHERE collection_name = ?
                    ORDER BY CAST(substr(chunk_id, 7) AS INTEGER)
                    LIMIT ?
                    """,
                    (self.collection_name, limit),
                ).fetchall()
        return [
            TextSearchResult(
                text=str(row["document"]),
                score=float(row["score"]),
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    @staticmethod
    def delete_collection(database_path: Path, collection_name: str) -> bool:
        path = database_path.expanduser().resolve()
        if not collection_name or not path.is_file():
            return False
        with sqlite3.connect(str(path), timeout=10) as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'text_collections'"
            ).fetchone()
            if table is None:
                return False
            connection.execute(
                "DELETE FROM text_chunks WHERE collection_name = ?", (collection_name,)
            )
            cursor = connection.execute(
                "DELETE FROM text_collections WHERE name = ?", (collection_name,)
            )
        return cursor.rowcount > 0
