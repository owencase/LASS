"""Streamlit entrypoint for LASS."""

from __future__ import annotations

import streamlit as st

from src.config import Settings
from src.llm import PromptManager
from src.pipeline import LASSPipeline
from src.stt.audio_utils import SUPPORTED_MEDIA_EXTENSIONS
from src.ui.components import render_result, save_uploaded_file


st.set_page_config(page_title="LASS", page_icon="🎙️", layout="centered")


@st.cache_resource
def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.ensure_directories()
    return settings


def main() -> None:
    settings = get_settings()
    prompt_manager = PromptManager(settings.prompt_dir)
    templates = prompt_manager.list_templates()

    st.title("🎙️ LASS")
    st.caption("오디오와 비디오를 로컬에서 텍스트로 변환하고 요약합니다.")

    uploaded_file = st.file_uploader(
        "미디어 파일",
        type=sorted(extension.lstrip(".") for extension in SUPPORTED_MEDIA_EXTENSIONS),
        help="파일은 외부 서버로 전송되지 않고 이 컴퓨터에서만 처리됩니다.",
    )
    selected_key = st.selectbox(
        "요약 형식",
        options=[template.key for template in templates],
        format_func=lambda key: next(t.name for t in templates if t.key == key),
    )
    language_choice = st.selectbox(
        "음성 언어",
        options=["자동 감지", "한국어", "영어"],
    )
    language_map = {"자동 감지": None, "한국어": "ko", "영어": "en"}

    if st.button(
        "변환 및 요약",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    ):
        assert uploaded_file is not None
        media_path = save_uploaded_file(uploaded_file, settings.input_dir)
        status = st.status("처리를 시작합니다.", expanded=True)
        try:
            pipeline = LASSPipeline(settings)
            result = pipeline.process(
                media_path,
                prompt_key=selected_key,
                language=language_map[language_choice],
                progress=lambda message: status.write(message),
            )
        except Exception as exc:
            status.update(label="처리에 실패했습니다.", state="error")
            st.error(str(exc))
        else:
            status.update(label="요약이 완료되었습니다.", state="complete", expanded=False)
            render_result(result)


if __name__ == "__main__":
    main()
