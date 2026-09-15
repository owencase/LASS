"""Browser-side center-pane layout and live transcript behavior."""

from __future__ import annotations

from pathlib import Path
import streamlit.components.v1 as components


_FRONTEND_PATH = Path(__file__).parent / "layout_component" / "frontend"
_layout_resizer = components.declare_component(
    "lass_layout_resizer", path=str(_FRONTEND_PATH.resolve())
)


def render_layout_resizer(
    content_layout: str,
    *,
    transcript_version: int = 0,
    transcript_active: bool = False,
    key: str = "lass-layout-resizer",
) -> None:
    """Mount the PDF/memo splitter without resizing the outer sidebars."""
    _layout_resizer(
        content_layout="side" if content_layout == "side" else "stack",
        transcript_version=max(0, int(transcript_version)),
        transcript_active=bool(transcript_active),
        default=None,
        key=key,
        tab_index=-1,
    )
