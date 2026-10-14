"""Persistent, double-buffered document images; navigation never blanks the page."""
from pathlib import Path
import streamlit.components.v1 as components

_viewer = components.declare_component("lass_document_viewer", path=str(Path(__file__).parent / "document_component/frontend"))


def render_document_viewer(url, title, page_number, *, zoom, fit, layout, key):
    _viewer(url=url, title=title, page_number=page_number, zoom=zoom, fit=fit,
            layout=layout, key=key, default=None, tab_index=-1)
