"""Attach first, then process inside an existing note."""

from pathlib import Path
from uuid import uuid4
from datetime import datetime
import json
import os
from urllib.request import Request, urlopen

import streamlit as st

from src.pdf_utils import DOCUMENT_EXTENSIONS, extract_document_text
from src.stt.audio_utils import SUPPORTED_MEDIA_EXTENSIONS
from src.rag import SQLiteTextStore, SQLiteVectorStore
from src.ui.components import render_info, format_bytes


def process_source(repository, source, settings, pipeline_factory, prompt_key, language,
                   summarize, progress):
    """Save transcription before attempting summary, so retries never lose work."""
    note = repository.get(source["note_id"])
    if source["status"] != "ready":
        try:
            repository.begin_source_processing(source["id"])
            progress("텍스트 추출 중…")
            if source["media_type"] in DOCUMENT_EXTENSIONS:
                document = extract_document_text(
                    Path(source["path"]),
                    progress=progress,
                )
                values = {"transcript": document.text}
            else:
                result = pipeline_factory().transcribe(Path(source["path"]), language=language)
                values = {"transcript": result.text, "segments": result.segments,
                          "duration": result.duration, "language": result.language}
            if note.vector_collection:
                SQLiteVectorStore.delete_collection(settings.vector_database_path, note.vector_collection)
                SQLiteTextStore.delete_collection(settings.vector_database_path, note.vector_collection)
            note = repository.complete_source(source["id"], **values)
            if note.folder_id:
                SQLiteVectorStore.delete_collection(settings.vector_database_path, f"folder-{note.folder_id}")
                SQLiteTextStore.delete_collection(settings.vector_database_path, f"folder-{note.folder_id}")
        except Exception as exc:
            repository.fail_source(source["id"], str(exc))
            raise
    if summarize:
        progress(f"'{note.title}' 요약 중…")
        result = pipeline_factory().resummarize(note.transcript, notes=note.notes,
                                               prompt_key=prompt_key, collection_key=note.id,
                                               progress=progress)
        repository.replace_summary(note.id, summary=result.summary, prompt_key=prompt_key,
                                   prompt_name=result.prompt_name, vector_collection=result.vector_collection)


def render_source_panel(repository, note, settings, pipeline_factory, model_ready, prompt_manager,
                        render_preview):
    sources = repository.list_sources(note.id)
    with st.expander("＋ 녹음 · 파일 추가", expanded=not sources and not note.source_name):
        modes = ["파일 선택", "바로 녹음", "로컬 파일 경로"]
        mode_key = f"source-mode-{note.id}"
        mode = st.session_state.get(mode_key, modes[0])
        st.caption("자료 추가 방식")
        mode_columns = st.columns(len(modes), gap="small")
        for option, column in zip(modes, mode_columns):
            with column:
                if st.button(
                    option,
                    type="primary" if option == mode else "secondary",
                    use_container_width=True,
                    key=f"{mode_key}-{option}",
                ):
                    mode = option
                    st.session_state[mode_key] = mode
        version_key = f"source-version-{note.id}"
        version = st.session_state.get(version_key, 0)
        upload, local_path = None, ""
        if mode == "파일 선택":
            upload = st.file_uploader("녹음, 영상, PDF, PowerPoint", type=sorted(
                extension.lstrip(".") for extension in {*SUPPORTED_MEDIA_EXTENSIONS, *DOCUMENT_EXTENSIONS}),
                key=f"source-file-{note.id}-{version}")
        elif mode == "바로 녹음":
            upload = st.audio_input("마이크로 녹음",
                                    key=f"source-record-{note.id}-{version}")
            st.caption("긴 녹음은 구간별로 저장해 주세요. 저장 전 녹음은 화면 메모리에 보관됩니다.")
        else:
            picker = os.environ.get("LASS_DESKTOP_PICKER")
            if picker and st.button("컴퓨터에서 파일 선택", key=f"native-pick-{note.id}"):
                try:
                    with urlopen(Request(picker, data=b"", method="POST"), timeout=300) as response:
                        picked = json.load(response).get("path", "")
                    if picked:
                        st.session_state[f"source-path-{note.id}-{version}"] = picked
                except Exception as exc:
                    st.error(f"파일 선택 창을 열지 못했습니다: {exc}")
            local_path = st.text_input("파일 절대 경로", key=f"source-path-{note.id}-{version}")
            st.caption("대용량 녹음·영상·PDF를 복사 없이 엽니다. 추가한 원본 파일의 위치를 유지해 주세요.")
        if st.button("이 노트에 자료 저장", type="primary", disabled=upload is None and not local_path.strip(),
                     key=f"source-save-{note.id}"):
            managed = None
            try:
                if upload is not None:
                    name = f"녹음-{datetime.now():%Y%m%d-%H%M%S}.wav" if mode == "바로 녹음" else Path(upload.name).name
                    managed_dir = settings.data_dir / "media"
                    managed_dir.mkdir(parents=True, exist_ok=True)
                    managed = managed_dir / f"{uuid4().hex}{Path(name).suffix.lower()}"
                    upload.seek(0)
                    with managed.open("wb") as output:
                        while chunk := upload.read(1024 * 1024):
                            output.write(chunk)
                    path = managed
                else:
                    path = Path(local_path.strip().strip('"')).expanduser().resolve()
                    name = path.name
                if path.suffix.lower() not in {*SUPPORTED_MEDIA_EXTENSIONS, *DOCUMENT_EXTENSIONS}:
                    raise ValueError("녹음, 영상, PDF 또는 PowerPoint 파일을 선택해 주세요.")
                attached = repository.add_source(note.id, path, name)
            except Exception as exc:
                if managed:
                    managed.unlink(missing_ok=True)
                st.error(str(exc))
            else:
                st.session_state[version_key] = version + 1
                st.session_state[f"source-selected-{note.id}"] = attached["id"]
                st.session_state["flash_message"] = "노트에 자료를 저장했습니다. 아래에서 전사·요약을 시작할 수 있습니다."
                st.rerun()

    if sources:
        selected_id = st.selectbox("첨부 자료", [source["id"] for source in sources],
            format_func=lambda value: next(item["name"] for item in sources if item["id"] == value),
            key=f"source-selected-{note.id}")
        source = next(item for item in sources if item["id"] == selected_id)
        st.caption(f"{format_bytes(source['file_size'])} · " +
                   {"attached": "처리 대기", "processing": "처리 중단 · 다시 시작 가능",
                    "ready": "텍스트 저장 완료", "failed": "처리 실패"}.get(source["status"], source["status"]))
        if source["error_message"]:
            st.error(source["error_message"])
        templates = prompt_manager.list_templates()
        controls = st.columns(3)
        with controls[0]:
            prompt_key = st.selectbox("요약 형식", [item.key for item in templates],
                format_func=lambda value: next(item.name for item in templates if item.key == value),
                key=f"source-template-{note.id}")
        with controls[1]:
            language = st.selectbox("음성 언어", [None, "ko", "en"],
                format_func=lambda value: {None: "자동 감지", "ko": "한국어", "en": "영어"}[value],
                key=f"source-language-{note.id}")
        with controls[2]:
            summarize = st.checkbox("추출 후 전체 노트 요약", value=model_ready, disabled=not model_ready,
                                      key=f"source-summary-{note.id}") and model_ready
        if not model_ready:
            st.caption("요약 모델이 없어도 자료 저장과 전사·문서 OCR은 가능합니다.")
        if st.button("전사 · 요약 시작" if source["status"] != "ready" else "전체 자료 다시 요약",
                     type="primary", key=f"source-process-{note.id}",
                     disabled=source["status"] == "ready" and not summarize):
            status = st.empty()
            status.info("텍스트 추출 중…")
            try:
                process_source(repository, source, settings, pipeline_factory, prompt_key,
                               language, summarize, status.info)
            except Exception as exc:
                status.error("처리를 완료하지 못했습니다. 저장된 자료와 전사는 유지됩니다.")
                st.error(str(exc))
            else:
                status.success("자료 처리가 완료되었습니다.")
                st.session_state["flash_message"] = "자료 처리 결과를 노트에 저장했습니다."
                st.rerun()
        if st.checkbox("선택한 원본 미리보기", key=f"source-preview-{source['id']}"):
            render_preview(source)
    elif not note.source_name:
        render_info("빈 노트", "먼저 메모를 작성하거나 녹음·영상·PDF·PowerPoint를 추가해 보세요.")
