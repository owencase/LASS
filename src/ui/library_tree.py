"""Streamlit custom component for the folder and note explorer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import streamlit.components.v1 as components

from src.storage import FolderRecord, LectureRecord


_FRONTEND_PATH = Path(__file__).parent / "library_component" / "frontend"
_library_tree = components.declare_component(
    "lass_library_tree", path=str(_FRONTEND_PATH.resolve())
)


def render_library_tree(
    folders: Iterable[FolderRecord],
    notes: Iterable[LectureRecord],
    selected_id: Optional[str],
    selected_folder_id: Optional[str] = None,
    *,
    key: str = "lass-library-tree",
    theme: str = "dark",
    text_scale: int = 100,
    sessions: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    folder_payload = [{"id": folder.id, "name": folder.name} for folder in folders]
    note_payload = [
        {
            "id": note.id,
            "title": note.title,
            "folder_id": note.folder_id,
            "status": note.status,
            "source_name": note.source_name,
        }
        for note in notes
    ]
    return _library_tree(
        folders=folder_payload,
        notes=note_payload,
        sessions=sessions or [],
        mode="sessions" if sessions is not None else "notes",
        selected_id=selected_id,
        selected_folder_id=selected_folder_id,
        theme_mode=theme,
        text_scale=text_scale,
        default=None,
        key=key,
        tab_index=0,
    )
