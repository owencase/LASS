"""Preserve the original document/section on every retrieval and exam excerpt."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceExcerpt:
    reference: str
    text: str
    note_id: str


def note_excerpts(notes, *, original_only=False):
    for note in notes() if callable(notes) else notes:
        title = str(getattr(note, "title", "제목 없음"))
        note_id = str(getattr(note, "id", title))
        document_pages = getattr(note, "document_pages", None)
        fields = (("transcript", "전체 녹취"), ("summary", "AI 요약"), ("notes", "사용자 메모"))
        if original_only:
            fields = (("transcript", "전체 녹취"),)
            if not str(getattr(note, "transcript", "") or "").strip() and not callable(document_pages):
                fields += (("notes", "사용자 메모"),)
        for field, label in fields:
            text = str(getattr(note, field, "") or "").strip()
            if not text:
                continue
            filename = str(getattr(note, "source_name", "") or title)
            section = label
            for part in re.split(r"(?m)^(#{1,6} .+)\s*$", text):
                if part.startswith("#"):
                    heading = part.lstrip("# ").strip()
                    if re.search(r"\.(?:pdf|pptx?|wav|mp3|mp4)$", heading, re.I):
                        filename, section = heading, label
                    else:
                        section = f"{label} / {heading}"
                elif part.strip():
                    yield SourceExcerpt(f"{title} · {filename} - {section}", part.strip(), note_id)
        if callable(document_pages):
            for page in document_pages():
                yield SourceExcerpt(f"{title} · {page['name']} - {page['page_number']}페이지/슬라이드", page["text"], note_id)
