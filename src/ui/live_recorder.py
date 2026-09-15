"""Small chunked microphone component for incremental local STT."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import streamlit.components.v1 as components


_FRONTEND_PATH = Path(__file__).parent / "live_recorder_component" / "frontend"
_live_recorder = components.declare_component(
    "lass_live_recorder", path=str(_FRONTEND_PATH.resolve())
)


def render_live_recorder(
    *,
    ack_nonce: str = "",
    theme: str = "dark",
    key: str = "lass-live-recorder",
) -> Optional[dict[str, Any]]:
    """Return one bounded audio-chunk event at a time.

    The browser waits for ``ack_nonce`` before releasing its next chunk, which
    prevents several large base64 values from being retained by Streamlit.
    """
    return _live_recorder(
        ack_nonce=ack_nonce,
        theme_mode=theme,
        default=None,
        key=key,
        tab_index=0,
    )
