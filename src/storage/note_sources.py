"""Multiple disk-backed attachments per note, with recoverable processing state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Any
from uuid import uuid4



class NoteSourcesMixin:
    def _initialize_sources(self, connection) -> None:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS note_sources (
                id TEXT PRIMARY KEY,
                note_id TEXT NOT NULL REFERENCES lectures(id) ON DELETE CASCADE,
                name TEXT NOT NULL, path TEXT NOT NULL, media_type TEXT NOT NULL,
                file_size INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'attached',
                error_message TEXT, created_at TEXT NOT NULL
            )
        """)
        connection.execute("CREATE INDEX IF NOT EXISTS source_note ON note_sources(note_id)")

    def create_empty(self, title: str = "새 노트", folder_id: Optional[str] = None):
        return self.create(title=title, source_name="", folder_id=folder_id)

    def add_source(self, note_id: str, path: Path, name: Optional[str] = None) -> dict:
        path = path.expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("내용이 있는 파일을 선택해 주세요.")
        source_id = uuid4().hex
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO note_sources(id, note_id, name, path, media_type, file_size, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (source_id, note_id, name or path.name, str(path), path.suffix.lower(),
                  path.stat().st_size, self._now()))
            connection.execute("UPDATE lectures SET updated_at = ? WHERE id = ?", (self._now(), note_id))
        return self.get_source(source_id)

    def list_sources(self, note_id: str) -> list[dict]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM note_sources WHERE note_id = ? ORDER BY created_at, rowid", (note_id,)
            )]

    def get_source(self, source_id: str) -> Optional[dict]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM note_sources WHERE id = ?", (source_id,)).fetchone()
        return dict(row) if row else None

    def fail_source(self, source_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE note_sources SET status = 'failed', error_message = ? WHERE id = ?",
                               (error[:2000], source_id))

    def complete_source(self, source_id: str, *, transcript: str, segments: Sequence[Any] = (),
                        duration: Optional[float] = None, language: Optional[str] = None):
        """Append once, atomically; keep user memo, edited text and existing summary."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            source = connection.execute("SELECT * FROM note_sources WHERE id = ?", (source_id,)).fetchone()
            if source is None:
                raise KeyError(source_id)
            note_id = source["note_id"]
            if source["status"] != "ready":
                note = connection.execute("SELECT * FROM lectures WHERE id = ?", (note_id,)).fetchone()
                offset = note["duration"] or 0.0
                combined_segments = json.loads(note["segments_json"])
                combined_segments.extend({"start": item.start + offset, "end": item.end + offset,
                                          "text": item.text} for item in segments)
                combined_text = (note["transcript"].rstrip() + "\n\n## " + source["name"] + "\n\n" + transcript).strip()
                connection.execute("""
                    UPDATE lectures SET transcript = ?, segments_json = ?, duration = ?, language = ?,
                        status = 'draft', error_message = NULL, vector_collection = NULL, updated_at = ?
                    WHERE id = ?
                """, (combined_text, json.dumps(combined_segments, ensure_ascii=False),
                      offset + duration if duration is not None else note["duration"],
                      language or note["language"], self._now(), note_id))
                connection.execute("UPDATE note_sources SET status = 'ready', error_message = NULL WHERE id = ?", (source_id,))
        return self.get(note_id)

    def append_live_transcription(
        self,
        note_id: str,
        *,
        transcript: str,
        segments: Sequence[Any] = (),
        duration: Optional[float] = None,
        language: Optional[str] = None,
        start_session: bool = False,
    ):
        """Append one short microphone chunk without retaining its audio in RAM.

        Live recorder blobs are written to a temporary file, transcribed, then
        discarded.  Only the compact text/timestamps are committed here.  The
        transaction also supplies the current timestamp offset, so repeated
        component events cannot overwrite an earlier chunk.
        """
        clean_text = str(transcript).strip()
        if not clean_text:
            record = self.get(note_id)
            if record is None:
                raise KeyError(note_id)
            return record
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            note = connection.execute(
                "SELECT transcript, segments_json, duration, language FROM lectures WHERE id = ?",
                (note_id,),
            ).fetchone()
            if note is None:
                raise KeyError(note_id)
            offset = float(note["duration"] or 0.0)
            combined_segments = json.loads(note["segments_json"] or "[]")
            combined_segments.extend(
                {
                    "start": float(item.start) + offset,
                    "end": float(item.end) + offset,
                    "text": str(item.text),
                }
                for item in segments
            )
            prefix = "\n\n## 실시간 녹음\n\n" if start_session else " "
            combined_text = (str(note["transcript"] or "").rstrip() + prefix + clean_text).strip()
            new_duration = offset + float(duration or 0.0)
            connection.execute(
                """
                UPDATE lectures SET transcript = ?, segments_json = ?, duration = ?, language = ?,
                    status = 'draft', error_message = NULL, vector_collection = NULL, updated_at = ?
                WHERE id = ?
                """,
                (
                    combined_text,
                    json.dumps(combined_segments, ensure_ascii=False),
                    new_duration,
                    language or note["language"],
                    self._now(),
                    note_id,
                ),
            )
        return self.get(note_id)

    def recent_segments(self, note_id: str, limit: int = 120) -> list[Any]:
        """Read only the newest timestamp rows instead of materializing a note."""
        from .lecture_repository import LectureSegment

        bounded_limit = min(500, max(1, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    CAST(json_extract(value, '$.start') AS REAL) AS start,
                    CAST(json_extract(value, '$.end') AS REAL) AS end,
                    CAST(json_extract(value, '$.text') AS TEXT) AS text,
                    CAST(key AS INTEGER) AS segment_index
                FROM json_each((SELECT segments_json FROM lectures WHERE id = ?))
                ORDER BY segment_index DESC
                LIMIT ?
                """,
                (note_id, bounded_limit),
            ).fetchall()
        return [
            LectureSegment(float(row["start"] or 0), float(row["end"] or 0), str(row["text"] or ""))
            for row in reversed(rows)
        ]

    def iter_transcript_pages(self, note_id: str, page_chars: int = 1800):
        """Yield bounded SQLite substrings; never select the complete transcript."""
        bounded_size = min(16_000, max(256, int(page_chars)))
        with self._connect() as connection:
            row = connection.execute(
                "SELECT length(transcript) AS chars FROM lectures WHERE id = ?",
                (note_id,),
            ).fetchone()
            if row is None:
                raise KeyError(note_id)
            total = int(row["chars"] or 0)
            for offset in range(1, total + 1, bounded_size):
                page_row = connection.execute(
                    "SELECT substr(transcript, ?, ?) AS page FROM lectures WHERE id = ?",
                    (offset, bounded_size, note_id),
                ).fetchone()
                page = str(page_row["page"] or "")
                if page.strip():
                    yield page

    def export_media(self, note_id: str, destination: Path) -> Path:
        """Export legacy DB media in bounded chunks, without joining the full file."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection, destination.open("wb") as output:
            count = 0
            for row in connection.execute("SELECT data FROM lecture_media_chunks WHERE lecture_id = ? ORDER BY chunk_index", (note_id,)):
                output.write(row["data"])
                count += 1
            if not count:
                size_row = connection.execute("SELECT length(media_blob) FROM lectures WHERE id = ?", (note_id,)).fetchone()
                size = size_row[0] if size_row else 0
                for offset in range(0, size or 0, self.MEDIA_CHUNK_SIZE):
                    row = connection.execute("SELECT substr(media_blob, ?, ?) FROM lectures WHERE id = ?",
                                             (offset + 1, self.MEDIA_CHUNK_SIZE, note_id)).fetchone()
                    output.write(row[0])
        return destination
