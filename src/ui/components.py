"""Reusable Streamlit UI components."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from src.pipeline import ProcessingResult


def save_uploaded_file(uploaded_file: Any, input_directory: Path) -> Path:
    input_directory.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    destination = input_directory / f"{uuid4().hex[:8]}_{safe_name}"
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def render_result(result: ProcessingResult) -> None:
    st.subheader("요약 결과")
    st.markdown(result.summary)

    with st.expander("타임스탬프 스크립트"):
        for segment in result.transcription.segments:
            minutes, seconds = divmod(int(segment.start), 60)
            st.markdown(f"`{minutes:02d}:{seconds:02d}` {segment.text}")

    st.download_button(
        "마크다운으로 다운로드",
        data=result.output_path.read_bytes(),
        file_name=result.output_path.name,
        mime="text/markdown",
        use_container_width=True,
    )
