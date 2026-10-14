"""SQLite-backed lecture library and media storage."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence
from uuid import uuid4
from .note_sources import NoteSourcesMixin


@dataclass(frozen=True)
class LectureSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class FolderRecord:
    id: str
    name: str
    created_at: str
    updated_at: str



@dataclass(frozen=True)
class ChatMessageRecord:
    id: str
    folder_id: Optional[str]
    role: str
    content: str
    created_at: str


@dataclass(frozen=True)
class LectureRecord:
    id: str
    title: str
    source_name: str
    source_path: str
    folder_id: Optional[str]
    has_media: bool
    media_type: str
    file_size: int
    status: str
    prompt_key: str
    prompt_name: str
    language: Optional[str]
    duration: Optional[float]
    transcript: str
    transcript_chars: int
    segments: List[LectureSegment]
    segment_count: int
    summary: str
    notes: str
    vector_collection: Optional[str]
    error_message: Optional[str]
    created_at: str
    updated_at: str
    document_pages: Any = field(default=None, compare=False, repr=False)


class LectureRepository(NoteSourcesMixin):
    """Small CRUD repository with one short-lived connection per operation."""

    MEDIA_CHUNK_SIZE = 8 * 1024 * 1024

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS folders (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS lectures (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    folder_id TEXT REFERENCES folders(id) ON DELETE SET NULL,
                    media_blob BLOB,
                    media_type TEXT NOT NULL DEFAULT '',
                    file_size INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'processing',
                    prompt_key TEXT NOT NULL DEFAULT '',
                    prompt_name TEXT NOT NULL DEFAULT '',
                    language TEXT,
                    duration REAL,
                    transcript TEXT NOT NULL DEFAULT '',
                    segments_json TEXT NOT NULL DEFAULT '[]',
                    summary TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    vector_collection TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(lectures)").fetchall()
            }
            self._initialize_sources(connection)
            if "media_blob" not in columns:
                connection.execute("ALTER TABLE lectures ADD COLUMN media_blob BLOB")
            if "folder_id" not in columns:
                connection.execute(
                    """
                    ALTER TABLE lectures ADD COLUMN folder_id TEXT
                    REFERENCES folders(id) ON DELETE SET NULL
                    """
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_lectures_updated_at "
                "ON lectures(updated_at DESC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS lecture_media_chunks (
                    lecture_id TEXT NOT NULL REFERENCES lectures(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    data BLOB NOT NULL,
                    PRIMARY KEY (lecture_id, chunk_index)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS app_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS folder_chat_messages (
                    id TEXT PRIMARY KEY,
                    folder_id TEXT NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_folder_chat_messages_folder "
                "ON folder_chat_messages(folder_id, created_at)"
            )
            connection.execute("""CREATE TABLE IF NOT EXISTS global_chat_messages (
                id TEXT PRIMARY KEY,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL, created_at TEXT NOT NULL)""")
            from .rollback_compat import restore_document_text
            restore_document_text(connection)
            connection.execute("""CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY, title TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            chat_columns = {row['name'] for row in connection.execute('PRAGMA table_info(global_chat_messages)')}
            if 'session_id' not in chat_columns:
                connection.execute("ALTER TABLE global_chat_messages ADD COLUMN session_id TEXT NOT NULL DEFAULT 'legacy'")
            now = self._now()
            if not connection.execute("SELECT 1 FROM app_metadata WHERE key = 'chat_sessions_migrated'").fetchone():
                connection.execute("INSERT OR IGNORE INTO chat_sessions(id, title, created_at, updated_at) VALUES ('legacy', '기존 대화', ?, ?)", (now, now))
                connection.execute("INSERT INTO app_metadata(key, value) VALUES ('chat_sessions_migrated', 'true')")
            connection.execute('CREATE INDEX IF NOT EXISTS chat_session_messages ON global_chat_messages(session_id)')
            connection.execute("""CREATE TABLE IF NOT EXISTS live_transcription_events (
                note_id TEXT NOT NULL REFERENCES lectures(id) ON DELETE CASCADE,
                nonce TEXT NOT NULL, PRIMARY KEY(note_id, nonce))""")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")

    def notes_revision(self, folder_id: Optional[str] = None) -> str:
        """Content-free retrieval cache key; None includes unfiled notes too."""
        import hashlib
        with self._connect() as connection:
            where = " WHERE folder_id = ?" if folder_id is not None else ""
            rows = connection.execute("SELECT id, updated_at FROM lectures" + where + " ORDER BY id",
                                      (folder_id,) if folder_id is not None else ())
            digest = hashlib.sha256()
            for row in rows:
                digest.update(f"{row['id']}:{row['updated_at']}\n".encode())
        return digest.hexdigest()

    def folder_revision(self, folder_id: str) -> str:
        return self.notes_revision(folder_id)

    @classmethod
    def _store_media_chunks(
        cls,
        connection: sqlite3.Connection,
        lecture_id: str,
        media_data: bytes,
    ) -> None:
        connection.execute(
            "DELETE FROM lecture_media_chunks WHERE lecture_id = ?", (lecture_id,)
        )
        if not media_data:
            return
        connection.executemany(
            """
            INSERT INTO lecture_media_chunks(lecture_id, chunk_index, data)
            VALUES (?, ?, ?)
            """,
            (
                (
                    lecture_id,
                    index,
                    sqlite3.Binary(
                        media_data[offset : offset + cls.MEDIA_CHUNK_SIZE]
                    ),
                )
                for index, offset in enumerate(
                    range(0, len(media_data), cls.MEDIA_CHUNK_SIZE)
                )
            ),
        )

    def _to_record(self, row: sqlite3.Row) -> LectureRecord:
        try:
            raw_segments = json.loads(row["segments_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            raw_segments = []
        segments = [
            LectureSegment(
                start=float(item.get("start", 0)),
                end=float(item.get("end", 0)),
                text=str(item.get("text", "")),
            )
            for item in raw_segments
            if isinstance(item, dict)
        ]
        return LectureRecord(
            id=row["id"],
            title=row["title"],
            source_name=row["source_name"],
            source_path=row["source_path"],
            folder_id=row["folder_id"],
            has_media=bool(row["has_media"]),
            media_type=row["media_type"],
            file_size=int(row["file_size"]),
            status=row["status"],
            prompt_key=row["prompt_key"],
            prompt_name=row["prompt_name"],
            language=row["language"],
            duration=row["duration"],
            transcript=row["transcript"],
            transcript_chars=int(row["transcript_chars"]),
            segments=segments,
            segment_count=int(row["segment_count"]),
            summary=row["summary"],
            notes=row["notes"],
            vector_collection=row["vector_collection"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            document_pages=(lambda: self.iter_document_pages(row["id"])) if row["has_document_pages"] else None,
        )

    def create(
        self,
        *,
        title: str,
        source_name: str,
        media_data: bytes = b"",
        source_path: Optional[Path] = None,
        media_type: str = "",
        file_size: Optional[int] = None,
        prompt_key: str = "",
        folder_id: Optional[str] = None,
    ) -> LectureRecord:
        lecture_id = uuid4().hex
        now = self._now()
        clean_title = title.strip() or Path(source_name).stem or "새 노트"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO lectures (
                    id, title, source_name, source_path, media_blob, media_type,
                    file_size, status, prompt_key, folder_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lecture_id,
                    clean_title,
                    source_name,
                    str(source_path) if source_path is not None else "",
                    None,
                    media_type,
                    len(media_data) if file_size is None else max(0, int(file_size)),
                    "processing" if source_name or source_path or media_data else "draft",
                    prompt_key,
                    folder_id,
                    now,
                    now,
                ),
            )
            self._store_media_chunks(connection, lecture_id, media_data)
        record = self.get(lecture_id)
        assert record is not None
        return record

    def complete(
        self,
        lecture_id: str,
        *,
        prompt_name: str,
        language: Optional[str],
        duration: Optional[float],
        transcript: str,
        segments: Sequence[Any],
        summary: str,
        vector_collection: Optional[str] = None,
    ) -> LectureRecord:
        encoded_segments = json.dumps(
            [
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": str(segment.text),
                }
                for segment in segments
            ],
            ensure_ascii=False,
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE lectures
                SET status = 'ready', prompt_name = ?, language = ?, duration = ?,
                    transcript = ?, segments_json = ?, summary = ?,
                    vector_collection = ?, error_message = NULL, updated_at = ?
                WHERE id = ?
                """,
                (
                    prompt_name,
                    language,
                    duration,
                    transcript,
                    encoded_segments,
                    summary,
                    vector_collection,
                    self._now(),
                    lecture_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown lecture: {lecture_id}")
        record = self.get(lecture_id)
        assert record is not None
        return record

    def mark_failed(self, lecture_id: str, message: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE lectures
                SET status = 'failed', error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (message[:2000], self._now(), lecture_id),
            )

    def update(
        self,
        lecture_id: str,
        *,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> LectureRecord:
        assignments = []
        values: List[Any] = []
        if title is not None:
            clean_title = title.strip()
            if not clean_title:
                raise ValueError("title cannot be empty")
            assignments.append("title = ?")
            values.append(clean_title)
        if summary is not None:
            assignments.append("summary = ?")
            values.append(summary)
        if notes is not None:
            assignments.append("notes = ?")
            values.append(notes)
        if not assignments:
            record = self.get(lecture_id)
            if record is None:
                raise KeyError(f"Unknown lecture: {lecture_id}")
            return record

        assignments.append("updated_at = ?")
        values.append(self._now())
        values.append(lecture_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE lectures SET {', '.join(assignments)} WHERE id = ?", values
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown lecture: {lecture_id}")
        record = self.get(lecture_id)
        assert record is not None
        return record

    def replace_summary(
        self,
        lecture_id: str,
        *,
        summary: str,
        prompt_key: str,
        prompt_name: str,
        vector_collection: Optional[str],
    ) -> LectureRecord:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE lectures
                SET status = 'ready', error_message = NULL, summary = ?, prompt_key = ?, prompt_name = ?,
                    vector_collection = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    summary,
                    prompt_key,
                    prompt_name,
                    vector_collection,
                    self._now(),
                    lecture_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown lecture: {lecture_id}")
            connection.execute("DELETE FROM summary_source_links WHERE note_id = ?", (lecture_id,))
            connection.execute("""INSERT INTO summary_source_links(note_id, source_id, page_number)
                SELECT s.note_id, p.source_id, p.page_number FROM source_text_pages p
                JOIN note_sources s ON s.id = p.source_id WHERE s.note_id = ?""", (lecture_id,))
        record = self.get(lecture_id)
        assert record is not None
        return record

    def update_segments(
        self, lecture_id: str, segments: Sequence[Any]
    ) -> LectureRecord:
        encoded_segments = []
        for segment in segments:
            start = float(segment.start)
            end = float(segment.end)
            text = str(segment.text).strip()
            if start < 0 or end < start:
                raise ValueError("segment timestamps are invalid")
            if not text:
                raise ValueError("segment text cannot be empty")
            encoded_segments.append({"start": start, "end": end, "text": text})
        transcript = " ".join(item["text"] for item in encoded_segments)
        existing = self.get(lecture_id)
        if existing and self.list_sources(lecture_id):
            # Source headings/PDF text must survive editing an audio segment.
            if len(existing.segments) != len(encoded_segments):
                raise ValueError("첨부 자료가 있는 노트에서는 녹취 구간 수를 유지해 주세요.")
            transcript = existing.transcript
            cursor = 0
            for previous, updated in zip(existing.segments, encoded_segments):
                index = transcript.find(previous.text, cursor)
                if index >= 0:
                    transcript = transcript[:index] + updated["text"] + transcript[index + len(previous.text):]
                    cursor = index + len(updated["text"])
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE lectures
                SET transcript = ?, segments_json = ?, vector_collection = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    transcript,
                    json.dumps(encoded_segments, ensure_ascii=False),
                    self._now(),
                    lecture_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown lecture: {lecture_id}")
        record = self.get(lecture_id)
        assert record is not None
        return record

    def get(self, lecture_id: str) -> Optional[LectureRecord]:
        with self._connect() as connection:
            row = connection.execute(
                f"{self._record_query()} WHERE id = ?", (lecture_id,)
            ).fetchone()
        return self._to_record(row) if row is not None else None

    def get_workspace(self, lecture_id: str) -> Optional[LectureRecord]:
        """Load visible note fields while leaving transcript payloads on disk."""
        query = self._record_query()
        query = query.replace("duration, transcript,", "duration, '' AS transcript,")
        query = query.replace(
            "segments_json, json_array_length(segments_json) AS segment_count,",
            "'[]' AS segments_json, json_array_length(segments_json) AS segment_count,",
        )
        with self._connect() as connection:
            row = connection.execute(f"{query} WHERE id = ?", (lecture_id,)).fetchone()
        return self._to_record(row) if row is not None else None

    @staticmethod
    def _record_query(lightweight: bool = False) -> str:
        query = """
            SELECT id, title, source_name, source_path, folder_id, media_type, file_size,
                   status, prompt_key, prompt_name, language, duration, transcript,
                   length(transcript) + COALESCE((SELECT SUM(p.chars) FROM source_text_pages p
                       JOIN note_sources s ON p.source_id = s.id WHERE s.note_id = lectures.id), 0) AS transcript_chars,
                   EXISTS(SELECT 1 FROM source_text_pages p JOIN note_sources s
                       ON p.source_id = s.id WHERE s.note_id = lectures.id) AS has_document_pages,
                   segments_json, json_array_length(segments_json) AS segment_count,
                   summary, notes, vector_collection, error_message,
                   created_at, updated_at,
                   CASE WHEN media_blob IS NOT NULL AND length(media_blob) > 0
                             OR EXISTS (
                                SELECT 1 FROM lecture_media_chunks
                                WHERE lecture_id = lectures.id
                             )
                        THEN 1 ELSE 0 END AS has_media
            FROM lectures
        """
        if lightweight:
            query = query.replace("duration, transcript,", "duration, '' AS transcript,")
            query = query.replace(
                "segments_json, json_array_length(segments_json) AS segment_count,\n"
                "                   summary, notes,",
                "'[]' AS segments_json, json_array_length(segments_json) AS segment_count,\n"
                "                   '' AS summary, '' AS notes,",
            )
        return query

    def get_media(self, lecture_id: str) -> Optional[bytes]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT media_blob FROM lectures WHERE id = ?", (lecture_id,)
            ).fetchone()
            chunks = connection.execute(
                """
                SELECT data FROM lecture_media_chunks
                WHERE lecture_id = ? ORDER BY chunk_index
                """,
                (lecture_id,),
            ).fetchall()
        if row is None:
            return None
        if chunks:
            return b"".join(bytes(chunk["data"]) for chunk in chunks)
        if row["media_blob"] is not None:
            return bytes(row["media_blob"])
        return None

    def set_media(
        self,
        lecture_id: str,
        media_data: bytes,
        *,
        clear_source_path: bool = False,
    ) -> None:
        if not media_data:
            raise ValueError("media_data cannot be empty")
        source_assignment = ", source_path = ''" if clear_source_path else ""
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE lectures
                SET media_blob = NULL, file_size = ?, updated_at = ?{source_assignment}
                WHERE id = ?
                """,
                (
                    len(media_data),
                    self._now(),
                    lecture_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown lecture: {lecture_id}")
            self._store_media_chunks(connection, lecture_id, media_data)

    def clear_source_path(self, lecture_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE lectures SET source_path = '' WHERE id = ?", (lecture_id,)
            )

    def list(self, search: str = "", *, lightweight: bool = False) -> List[LectureRecord]:
        with self._connect() as connection:
            if search.strip():
                pattern = f"%{search.strip()}%"
                rows = connection.execute(
                    f"""
                    {self._record_query(lightweight)}
                    WHERE title LIKE ? COLLATE NOCASE
                       OR source_name LIKE ? COLLATE NOCASE
                       OR notes LIKE ? COLLATE NOCASE
                    ORDER BY updated_at DESC
                    """,
                    (pattern, pattern, pattern),
                ).fetchall()
            else:
                rows = connection.execute(
                    f"{self._record_query(lightweight)} ORDER BY updated_at DESC"
                ).fetchall()
        return [self._to_record(row) for row in rows]

    def list_by_folder(
        self, folder_id: str, *, lightweight: bool = False
    ) -> List[LectureRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                f"{self._record_query(lightweight)} WHERE folder_id = ? ORDER BY updated_at DESC",
                (folder_id,),
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def iter_by_folder(self, folder_id: str):
        """Yield full folder records one by one from SQLite for retrieval jobs."""
        yield from self.iter_notes(folder_id)

    def iter_notes(self, folder_id: Optional[str] = None):
        """Stream notes from the whole library or one folder without media blobs."""
        with self._connect() as connection:
            where = " WHERE folder_id = ?" if folder_id is not None else ""
            rows = connection.execute(
                self._record_query() + where + " ORDER BY updated_at DESC",
                (folder_id,) if folder_id is not None else (),
            )
            for row in rows:
                yield self._to_record(row)

    def folder_has_content(self, folder_id: str) -> bool:
        """Check chat readiness without selecting any full note content."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM lectures
                    WHERE folder_id = ? AND (
                        length(trim(transcript)) > 0 OR
                        length(trim(summary)) > 0 OR
                        length(trim(notes)) > 0 OR EXISTS (SELECT 1 FROM source_text_pages p
                            JOIN note_sources s ON p.source_id = s.id WHERE s.note_id = lectures.id)
                    )
                ) AS has_content
                """,
                (folder_id,),
            ).fetchone()
        return bool(row["has_content"]) if row is not None else False

    def delete(self, lecture_id: str) -> Optional[LectureRecord]:
        record = self.get(lecture_id)
        if record is None:
            return None
        with self._connect() as connection:
            connection.execute("DELETE FROM lectures WHERE id = ?", (lecture_id,))
        return record

    def move(self, lecture_id: str, folder_id: Optional[str]) -> LectureRecord:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE lectures SET folder_id = ?, updated_at = ? WHERE id = ?",
                (folder_id, self._now(), lecture_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown lecture: {lecture_id}")
        record = self.get(lecture_id)
        assert record is not None
        return record

    def create_folder(self, name: str) -> FolderRecord:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("folder name cannot be empty")
        folder_id = uuid4().hex
        now = self._now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO folders(id, name, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (folder_id, clean_name, now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("folder name already exists") from exc
        folder = self.get_folder(folder_id)
        assert folder is not None
        return folder

    def get_folder(self, folder_id: str) -> Optional[FolderRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name, created_at, updated_at FROM folders WHERE id = ?",
                (folder_id,),
            ).fetchone()
        if row is None:
            return None
        return FolderRecord(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_folders(self) -> List[FolderRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, created_at, updated_at
                FROM folders ORDER BY name COLLATE NOCASE
                """
            ).fetchall()
        return [
            FolderRecord(
                id=row["id"],
                name=row["name"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def rename_folder(self, folder_id: str, name: str) -> FolderRecord:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("folder name cannot be empty")
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE folders SET name = ?, updated_at = ? WHERE id = ?",
                    (clean_name, self._now(), folder_id),
                )
                if cursor.rowcount == 0:
                    raise KeyError(f"Unknown folder: {folder_id}")
        except sqlite3.IntegrityError as exc:
            raise ValueError("folder name already exists") from exc
        folder = self.get_folder(folder_id)
        assert folder is not None
        return folder

    def delete_folder(self, folder_id: str) -> Optional[FolderRecord]:
        folder = self.get_folder(folder_id)
        if folder is None:
            return None
        with self._connect() as connection:
            connection.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
        return folder

    def active_chat_session(self) -> str:
        selected = self.get_metadata('active_chat_session')
        sessions = self.list_chat_sessions()
        if any(row['id'] == selected for row in sessions):
            return selected
        if not sessions:
            return self.create_chat_session()
        selected = sessions[0]['id']
        self.set_metadata('active_chat_session', selected)
        return selected

    def rename_chat_session(self, session_id: str, title: str) -> None:
        title = title.strip()
        if not title or len(title) > 120:
            raise ValueError('대화 이름은 1~120자로 입력해 주세요.')
        with self._connect() as connection:
            if not connection.execute('UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?',
                                      (title, self._now(), session_id)).rowcount:
                raise KeyError(session_id)

    def delete_chat_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute('DELETE FROM global_chat_messages WHERE session_id = ?', (session_id,))
            connection.execute('DELETE FROM chat_sessions WHERE id = ?', (session_id,))
            connection.execute("DELETE FROM app_metadata WHERE key = 'active_chat_session' AND value = ?", (session_id,))

    def list_chat_sessions(self) -> list[dict]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute('SELECT * FROM chat_sessions ORDER BY updated_at DESC, rowid DESC')]

    def create_chat_session(self) -> str:
        session_id, now = uuid4().hex, self._now()
        with self._connect() as connection:
            connection.execute("INSERT INTO chat_sessions(id, title, created_at, updated_at) VALUES (?, '새 대화', ?, ?)", (session_id, now, now))
        self.set_metadata('active_chat_session', session_id)
        return session_id

    def add_chat_message(
        self, folder_id: Optional[str], role: str, content: str, *, session_id: Optional[str] = None
    ) -> ChatMessageRecord:
        if role not in {"user", "assistant"}:
            raise ValueError("chat role must be user or assistant")
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("chat content cannot be empty")
        message_id = uuid4().hex
        created_at = self._now()
        if folder_id is None:
            session_id = session_id or self.active_chat_session()
        with self._connect() as connection:
            if folder_id is None:
                if not connection.execute('SELECT 1 FROM chat_sessions WHERE id = ?', (session_id,)).fetchone():
                    raise KeyError(session_id)
                first = not connection.execute('SELECT 1 FROM global_chat_messages WHERE session_id = ?', (session_id,)).fetchone()
                connection.execute("INSERT INTO global_chat_messages(id, role, content, created_at, session_id) VALUES (?, ?, ?, ?, ?)",
                                   (message_id, role, clean_content, created_at, session_id))
                connection.execute('UPDATE chat_sessions SET updated_at = ? WHERE id = ?', (created_at, session_id))
                if first and role == 'user':
                    connection.execute("UPDATE chat_sessions SET title = ? WHERE id = ? AND title IN ('새 대화', '기존 대화')", (clean_content[:48], session_id))
            else:
                connection.execute(
                    """
                    INSERT INTO folder_chat_messages(id, folder_id, role, content, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (message_id, folder_id, role, clean_content, created_at),
                )
        return ChatMessageRecord(
            id=message_id,
            folder_id=folder_id,
            role=role,
            content=clean_content,
            created_at=created_at,
        )

    def list_chat_messages(
        self, folder_id: Optional[str], limit: Optional[int] = 100, *, session_id: Optional[str] = None
    ) -> List[ChatMessageRecord]:
        if limit is not None and limit <= 0:
            return []
        table = "global_chat_messages" if folder_id is None else "folder_chat_messages"
        folder_column = "NULL AS folder_id" if folder_id is None else "folder_id"
        where = "WHERE session_id = ?" if folder_id is None else "WHERE folder_id = ?"
        selected = (session_id or self.active_chat_session()) if folder_id is None else folder_id
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, folder_id, role, content, created_at
                FROM (
                    SELECT rowid AS sequence, id, {folder_column}, role, content, created_at
                    FROM {table}
                    {where}
                    ORDER BY rowid DESC LIMIT ?
                )
                ORDER BY sequence
                """,
                (selected, limit if limit is not None else -1),
            ).fetchall()
        return [
            ChatMessageRecord(
                id=row["id"],
                folder_id=row["folder_id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def clear_chat_messages(self, folder_id: Optional[str]) -> int:
        with self._connect() as connection:
            if folder_id is None:
                return connection.execute("DELETE FROM global_chat_messages WHERE session_id = ?", (self.active_chat_session(),)).rowcount
            cursor = connection.execute(
                "DELETE FROM folder_chat_messages WHERE folder_id = ?", (folder_id,)
            )
        return cursor.rowcount

    def get_metadata(self, key: str) -> Optional[str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM app_metadata WHERE key = ?", (key,)
            ).fetchone()
        return str(row["value"]) if row is not None else None

    def set_metadata(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO app_metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )
