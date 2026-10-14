"""Double-click editors for summaries and timestamped transcript rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import streamlit.components.v1 as components

from src.storage import LectureSegment


_FRONTEND_PATH = Path(__file__).parent / "editable_component" / "frontend"
_inline_editor = components.declare_component(
    "lass_inline_editor", path=str(_FRONTEND_PATH.resolve())
)


def render_editable_summary(
    summary: str,
    *,
    key: str,
    theme: str = "dark",
    text_scale: int = 100,
    streaming: bool = False,
    progress: str = "",
    error: str = "",
) -> Optional[dict[str, Any]]:
    return _inline_editor(
        mode="summary",
        draft_key=key,
        text=summary,
        segments=[],
        theme_mode=theme,
        text_scale=text_scale,
        streaming=streaming,
        progress=progress,
        error=error,
        default=None,
        key=key,
        tab_index=0,
    )


def render_editable_transcript(
    segments: Iterable[LectureSegment],
    *,
    key: str,
    theme: str = "dark",
    text_scale: int = 100,
) -> Optional[dict[str, Any]]:
    return _inline_editor(
        mode="transcript",
        draft_key=key,
        text="",
        segments=[
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments
        ],
        theme_mode=theme,
        text_scale=text_scale,
        default=None,
        key=key,
        tab_index=0,
    )


def render_memo_editor(
    text: str,
    *,
    key: str,
    theme: str = "dark",
    text_scale: int = 100,
    height: int = 180,
) -> Optional[dict[str, Any]]:
    """Render the memo editor with formatting tools and debounced auto-save."""
    return _inline_editor(
        mode="memo",
        draft_key=key,
        text=text,
        segments=[],
        theme_mode=theme,
        text_scale=text_scale,
        height=max(120, height),
        default=None,
        key=key,
        tab_index=0,
    )
