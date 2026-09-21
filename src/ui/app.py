"""Streamlit entrypoint for the LASS note workspace."""

from __future__ import annotations

import base64
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Optional
from dataclasses import replace
from html import escape
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import streamlit as st

from src.config import Settings
from src.llm import LocalModelManager, ModelStatus, PromptManager
from src.model_registry import ModelProfile, ModelRegistry
from src.pdf_utils import DOCUMENT_EXTENSIONS
from src.pdf_stream import LocalMediaServer
from src.ui.source_panel import process_source, render_source_panel
from src.pipeline import LASSPipeline
from src.rag import SQLiteTextStore, SQLiteVectorStore
from src.storage import (
    FolderRecord,
    LectureRecord,
    LectureRepository,
    LectureSegment,
    migrate_legacy_storage,
)
from src.stt.audio_utils import SUPPORTED_MEDIA_EXTENSIONS
from src.stt import create_transcriber
from src.stt.transcriber import assess_transcription_quality
from src.ui.components import (
    build_markdown_export,
    format_bytes,
    inject_theme,
    render_empty_workspace,
    render_error,
    render_info,
    render_lecture_header,
    render_metrics,
    render_model_status,
    render_pane_title,
    render_timeline,
    render_topbar,
    render_warning,
)
from src.ui.editable_content import (
    render_editable_summary,
    render_editable_transcript,
    render_memo_editor,
)
from src.ui.library_tree import render_library_tree
from src.ui.layout_resizer import render_layout_resizer
from src.ui.live_recorder import render_live_recorder


st.set_page_config(
    page_title="LASS · Note Workspace",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
PDF_EXTENSION = ".pdf"
PRESENTATION_EXTENSIONS = {".ppt", ".pptx"}
SUPPORTED_SOURCE_EXTENSIONS = {*SUPPORTED_MEDIA_EXTENSIONS, *DOCUMENT_EXTENSIONS}


@st.cache_resource
def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.ensure_directories()
    return settings


@st.cache_resource
def get_repository() -> LectureRepository:
    return LectureRepository(get_settings().database_path)


@st.cache_resource
def get_vector_database() -> SQLiteVectorStore:
    return SQLiteVectorStore(get_settings().vector_database_path)


@st.cache_resource(show_spinner=False, max_entries=1)
def _get_pipeline_for_profiles(
    stt_location: str,
    stt_backend: str,
    stt_device: str,
    stt_compute_type: str,
    llm_identifier: str,
    llm_path: str,
    llm_device: str,
    llm_backend: str,
) -> LASSPipeline:
    return LASSPipeline(
        get_settings(),
        stt_model=stt_location,
        stt_backend=stt_backend,
        stt_device=stt_device,
        stt_compute_type=stt_compute_type,
        llm_model_id=llm_identifier,
        llm_model_dir=Path(llm_path),
        llm_device=llm_device,
        llm_backend=llm_backend,
    )


def get_model_registry() -> ModelRegistry:
    return ModelRegistry.from_settings(get_settings())


def get_pipeline() -> LASSPipeline:
    registry = get_model_registry()
    stt_profile = registry.active("stt")
    llm_profile = registry.active("llm")
    return _get_pipeline_for_profiles(
        stt_profile.runtime_location,
        stt_profile.runtime_backend,
        stt_profile.device,
        stt_profile.compute_type,
        llm_profile.identifier,
        llm_profile.runtime_location,
        llm_profile.device,
        llm_profile.runtime_backend,
    )


def _selected_note(
    repository: LectureRepository, notes: list[LectureRecord]
) -> Optional[LectureRecord]:
    if st.session_state.get("new_note_mode") or st.session_state.get(
        "selected_folder_id"
    ):
        return None
    selected_id = st.session_state.get("selected_note_id")
    selected = repository.get_workspace(selected_id) if selected_id else None
    if selected is None and notes:
        selected = repository.get_workspace(notes[0].id)
        st.session_state["selected_note_id"] = selected.id
    return selected


def _remove_vector_index(note: LectureRecord, settings: Settings) -> list[str]:
    warnings: list[str] = []
    if note.vector_collection:
        for store in (SQLiteTextStore, SQLiteVectorStore):
            try:
                store.delete_collection(
                    settings.vector_database_path, note.vector_collection
                )
            except Exception as exc:
                warnings.append(f"검색 인덱스 삭제 실패: {exc}")
    return warnings


def _remove_managed_pdf(note: LectureRecord, settings: Settings) -> list[str]:
    if note.media_type.lower() != PDF_EXTENSION or not note.source_path:
        return []
    source = Path(note.source_path).expanduser().resolve()
    try:
        source.relative_to(settings.document_dir.resolve())
    except ValueError:
        return []
    try:
        source.unlink(missing_ok=True)
    except OSError as exc:
        return [f"관리 PDF 삭제 실패: {exc}"]
    return []


def _remove_folder_vector_index(folder_id: Optional[str], settings: Settings) -> None:
    if not folder_id:
        return
    collection = LASSPipeline.folder_collection_name(folder_id)
    SQLiteTextStore.delete_collection(settings.vector_database_path, collection)
    SQLiteVectorStore.delete_collection(settings.vector_database_path, collection)


def _handle_library_event(
    event: Optional[dict[str, Any]],
    repository: LectureRepository,
    settings: Settings,
) -> None:
    if not isinstance(event, dict):
        return
    nonce = event.get("nonce")
    if not nonce or nonce == st.session_state.get("processed_library_event"):
        return
    st.session_state["processed_library_event"] = nonce
    action = event.get("action")

    try:
        if action == "select_note":
            note_id = str(event["note_id"])
            if repository.get(note_id) is None:
                return
            st.session_state["selected_note_id"] = note_id
            st.session_state.pop("new_note_mode", None)
            st.session_state.pop("selected_folder_id", None)
        elif action == "select_folder":
            folder_id = str(event["folder_id"])
            if repository.get_folder(folder_id) is None:
                return
            st.session_state["selected_folder_id"] = folder_id
            st.session_state.pop("selected_note_id", None)
            st.session_state.pop("new_note_mode", None)
        elif action == "create_note":
            folder_id = event.get("folder_id")
            created = repository.create_empty(
                title=str(event["title"]),
                folder_id=str(folder_id) if folder_id else None,
            )
            st.session_state["selected_note_id"] = created.id
            st.session_state.pop("selected_folder_id", None)
            st.session_state.pop("new_note_mode", None)
            st.session_state["flash_message"] = "새 노트를 만들었습니다."
        elif action == "create_folder":
            repository.create_folder(str(event["name"]))
            st.session_state["flash_message"] = "새 폴더를 만들었습니다."
        elif action == "rename_note":
            repository.update(str(event["note_id"]), title=str(event["name"]))
            st.session_state["flash_message"] = "노트 이름을 변경했습니다."
        elif action == "move_note":
            note = repository.get(str(event["note_id"]))
            if note is None:
                return
            folder_id = event.get("folder_id")
            repository.move(
                note.id, str(folder_id) if folder_id else None
            )
            _remove_folder_vector_index(note.folder_id, settings)
            _remove_folder_vector_index(str(folder_id) if folder_id else None, settings)
            st.session_state["flash_message"] = "노트를 이동했습니다."
        elif action == "delete_note":
            note = repository.get(str(event["note_id"]))
            if note is None:
                return
            attachment_paths = [Path(source["path"]) for source in repository.list_sources(note.id)]
            repository.delete(note.id)
            for path in attachment_paths:
                try:
                    path.resolve().relative_to((settings.data_dir / "media").resolve())
                    path.unlink(missing_ok=True)
                except (ValueError, OSError):
                    pass
            (settings.data_dir / "preview" / f"{note.id}{note.media_type}").unlink(missing_ok=True)
            st.session_state["delete_warnings"] = [
                *_remove_vector_index(note, settings),
                *_remove_managed_pdf(note, settings),
            ]
            _remove_folder_vector_index(note.folder_id, settings)
            if st.session_state.get("selected_note_id") == note.id:
                st.session_state.pop("selected_note_id", None)
            st.session_state["flash_message"] = "노트를 삭제했습니다."
        elif action == "rename_folder":
            repository.rename_folder(str(event["folder_id"]), str(event["name"]))
            st.session_state["flash_message"] = "폴더 이름을 변경했습니다."
        elif action == "delete_folder":
            folder_id = str(event["folder_id"])
            repository.delete_folder(folder_id)
            _remove_folder_vector_index(folder_id, settings)
            if st.session_state.get("selected_folder_id") == folder_id:
                st.session_state.pop("selected_folder_id", None)
            st.session_state["flash_message"] = (
                "폴더를 삭제하고 안의 노트를 미분류로 옮겼습니다."
            )
        else:
            return
    except (KeyError, ValueError) as exc:
        message = {
            "folder name already exists": "같은 이름의 폴더가 이미 있습니다.",
            "folder name cannot be empty": "폴더 이름을 입력해 주세요.",
            "title cannot be empty": "노트 이름을 입력해 주세요.",
        }.get(str(exc), str(exc))
        st.session_state["flash_error"] = message
    st.rerun()


def _media_mime(media_type: str) -> str:
    return {
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".wav": "audio/wav",
        ".mkv": "video/x-matroska",
        ".mov": "video/quicktime",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }.get(media_type.lower(), "application/octet-stream")


@st.cache_data(show_spinner=False, max_entries=24)
def _cached_pdf_page_image(
    source: str, modified_ns: int, file_size: int, page_number: int
) -> tuple[bytes, int]:
    """Render and cache only the visible PDF page as an image."""
    del modified_ns, file_size
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PDF 미리보기에는 PyMuPDF가 필요합니다.") from exc

    document = pymupdf.open(source)
    try:
        if document.needs_pass and not document.authenticate(""):
            raise ValueError("암호로 잠긴 PDF는 먼저 잠금을 해제해 주세요.")
        page_count = document.page_count
        if page_count <= 0:
            raise ValueError("PDF에 표시할 페이지가 없습니다.")
        if not 1 <= page_number <= page_count:
            raise ValueError(f"PDF 페이지는 1에서 {page_count} 사이여야 합니다.")
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2),
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        return pixmap.tobytes("png"), page_count
    finally:
        document.close()


def _media_path(repository: LectureRepository, note: LectureRecord) -> Optional[Path]:
    source = Path(note.source_path).expanduser() if note.source_path else None
    if source and source.is_file():
        return source
    if not note.has_media:
        return None
    source = get_settings().data_dir / "preview" / f"{note.id}{note.media_type}"
    if not source.is_file() or source.stat().st_size != note.file_size:
        repository.export_media(note.id, source)
    return source


def _pdf_preview(
    repository: LectureRepository, note: LectureRecord
) -> Optional[tuple[Path, bytes, int, int]]:
    source = _media_path(repository, note)
    if source is None or note.media_type.lower() != PDF_EXTENSION:
        return None
    page_key = f"pdf-page-{note.id}"
    page_number = max(1, int(st.session_state.get(page_key, 1)))
    stat = source.stat()
    try:
        page_data, page_count = _cached_pdf_page_image(
            str(source.resolve()), stat.st_mtime_ns, stat.st_size, page_number
        )
    except ValueError:
        page_number = 1
        st.session_state[page_key] = 1
        page_data, page_count = _cached_pdf_page_image(
            str(source.resolve()), stat.st_mtime_ns, stat.st_size, page_number
        )
    return source, page_data, page_number, page_count


def _render_pdf_viewer(
    note: LectureRecord,
    preview: tuple[Path, bytes, int, int],
    *,
    zoom: int = 100,
    fit: bool = True,
) -> None:
    """Render one page without a second, conflicting PDF toolbar."""
    _, page_data, page_number, _ = preview
    encoded = base64.b64encode(page_data).decode("ascii")
    st.markdown(
        f'<div class="lass-pdf-stage" data-zoom="{zoom / 100:.2f}" '
        f'data-fit="{str(fit).lower()}">'
        f'<img class="lass-pdf-page" data-page="{page_number}" '
        f'src="data:image/png;base64,{encoded}" alt="{escape(note.title, quote=True)} · '
        f'{page_number}페이지"></div>',
        unsafe_allow_html=True,
    )


def _render_recording(
    repository: LectureRepository,
    note: LectureRecord,
    *,
    pdf_preview: Optional[tuple[Path, bytes, int, int]] = None,
    pdf_zoom: int = 100,
    pdf_fit: bool = True,
) -> None:
    path = _media_path(repository, note)
    if path is None:
        render_info("자료 없음", "원본 파일이 없거나 이동되었습니다. 자료를 다시 추가해 주세요.")
        return
    if note.media_type.lower() == PDF_EXTENSION:
        try:
            preview = pdf_preview or _pdf_preview(repository, note)
        except (OSError, RuntimeError, ValueError) as exc:
            render_error(str(exc))
            return
        if preview is not None:
            _render_pdf_viewer(note, preview, zoom=pdf_zoom, fit=pdf_fit)
    elif note.media_type.lower() in PRESENTATION_EXTENSIONS:
        render_info(
            "PowerPoint 원본",
            "PowerPoint는 텍스트와 슬라이드 이미지를 OCR로 처리합니다. 원본 미리보기는 파일에서 확인해 주세요.",
        )
    else:
        url = _get_media_server().register(path)
        tag = "video" if note.media_type.lower() in VIDEO_EXTENSIONS else "audio"
        st.markdown(f'<{tag} controls preload="none" src="{escape(url, quote=True)}" style="width:100%"></{tag}>',
                    unsafe_allow_html=True)


@st.cache_resource
def _get_media_server() -> LocalMediaServer:
    return LocalMediaServer()


def _handle_editor_event(
    event: Optional[dict[str, Any]],
    repository: LectureRepository,
    note: LectureRecord,
    settings: Settings,
) -> None:
    if not isinstance(event, dict):
        return
    nonce = event.get("nonce")
    if not nonce or nonce == st.session_state.get("processed_editor_event"):
        return
    st.session_state["processed_editor_event"] = nonce
    action = event.get("action")
    try:
        if action == "save_summary":
            repository.update(note.id, summary=str(event.get("text", "")))
            _remove_folder_vector_index(note.folder_id, settings)
            st.session_state["flash_message"] = "AI 요약 수정 내용을 저장했습니다."
        elif action == "save_memo":
            repository.update(note.id, notes=str(event.get("text", "")))
            _remove_folder_vector_index(note.folder_id, settings)
        elif action == "save_segment":
            index = int(event["index"])
            if index < 0 or index >= len(note.segments):
                raise ValueError("수정할 녹취 구간을 찾을 수 없습니다.")
            segments = list(note.segments)
            segments[index] = LectureSegment(
                start=float(event["start"]),
                end=float(event["end"]),
                text=str(event["text"]),
            )
            warnings = _remove_vector_index(note, settings)
            repository.update_segments(note.id, segments)
            _remove_folder_vector_index(note.folder_id, settings)
            st.session_state["delete_warnings"] = warnings
            st.session_state["flash_message"] = (
                "타임스탬프 녹취를 수정했습니다. 다음 재요약과 채팅에 반영됩니다."
            )
        else:
            return
    except (KeyError, TypeError, ValueError) as exc:
        st.session_state["flash_error"] = str(exc)
    st.rerun()


def _resummarize_note(
    repository: LectureRepository,
    note: LectureRecord,
    settings: Settings,
    prompt_key: str,
    include_memo: bool,
    status: Optional[Any] = None,
) -> None:
    if status is None:
        status = st.empty()
    status.info(f"'{note.title}' 요약 중…")
    try:
        workspace_note = repository.get_workspace(note.id)
        if workspace_note is None:
            raise KeyError(f"Unknown lecture: {note.id}")
        result = get_pipeline().resummarize_pages(
            _interactive_summary_pages(repository, workspace_note, settings),
            prompt_key=prompt_key,
            notes=workspace_note.notes if include_memo else "",
            progress=lambda message: status.info(message),
        )
        if note.vector_collection and note.vector_collection != result.vector_collection:
            SQLiteVectorStore.delete_collection(
                settings.vector_database_path, note.vector_collection
            )
            SQLiteTextStore.delete_collection(
                settings.vector_database_path, note.vector_collection
            )
        repository.replace_summary(
            note.id,
            summary=result.summary,
            prompt_key=prompt_key,
            prompt_name=result.prompt_name,
            vector_collection=result.vector_collection,
        )
        _remove_folder_vector_index(note.folder_id, settings)
    except Exception as exc:
        status.error("AI 재요약에 실패했습니다.")
        st.error(str(exc))
    else:
        status.success("AI 재요약을 저장했습니다.")
        st.session_state["flash_message"] = "새 AI 요약을 저장했습니다."
        st.rerun()


def _remember_note_section(note_id: str) -> None:
    widget_key = f"note-section-{note_id}"
    st.session_state[f"active-note-section-{note_id}"] = st.session_state[widget_key]


def _button_choice(
    label: str,
    options: list[str],
    current: str,
    *,
    key: str,
    show_label: bool = True,
) -> str:
    """Render a stable, high-contrast segmented choice from native buttons."""
    if show_label:
        st.caption(label)
    selected = st.session_state.get(key, current)
    if selected not in options:
        selected = options[0]
        st.session_state[key] = selected
    columns = st.columns(len(options), gap="small")
    for option, column in zip(options, columns):
        with column:
            st.button(
                option,
                type="primary" if option == selected else "secondary",
                use_container_width=True,
                key=f"{key}-{option}",
                on_click=lambda value=option: st.session_state.__setitem__(key, value),
            )
    return st.session_state.get(key, selected)


def _render_note_memo(
    repository: LectureRepository,
    note: LectureRecord,
    settings: Settings,
    *,
    height: int,
    theme_mode: str = "light",
    text_scale: int = 100,
) -> None:
    event = render_memo_editor(
        note.notes,
        key=f"memo-editor-{note.id}",
        theme=theme_mode,
        text_scale=text_scale,
        height=height,
    )
    _handle_editor_event(event, repository, note, settings)


def _render_summary_panel(
    repository: LectureRepository,
    note: Optional[LectureRecord],
    settings: Settings,
    model_status: ModelStatus,
    prompt_manager: PromptManager,
    theme_mode: str,
    text_scale: int,
) -> None:
    render_pane_title("AI 노트", "STREAMING · ON DEVICE")
    st.caption("모든 자료를 한 조각씩 읽고, 짧은 근거 메모만 디스크에서 합칩니다.")
    if note is None:
        render_info("AI 패널 대기 중", "왼쪽에서 노트를 선택하면 요약과 재요약 도구가 여기에 열립니다.")
        return

    templates = prompt_manager.list_templates()
    template_keys = [template.key for template in templates]
    default_index = template_keys.index(note.prompt_key) if note.prompt_key in template_keys else 0
    selected_key = st.selectbox(
        "요약 형식",
        options=template_keys,
        index=default_index,
        format_func=lambda key: next(template.name for template in templates if template.key == key),
        key=f"resummary-template-{note.id}",
    )
    include_memo = st.checkbox(
        "내 메모 포함",
        value=True,
        disabled=not bool(note.notes.strip()),
        key=f"resummary-memo-{note.id}",
    )
    if st.button(
        "↻ AI 다시 요약하기",
        type="primary",
        use_container_width=True,
        disabled=not model_status.ready or note.transcript_chars <= 0,
        key=f"resummary-{note.id}",
    ):
        _resummarize_note(
            repository,
            note,
            settings,
            selected_key,
            include_memo,
        )
    summary_event = render_editable_summary(
        note.summary,
        key=f"editable-summary-{note.id}",
        theme=theme_mode,
        text_scale=text_scale,
    )
    _handle_editor_event(summary_event, repository, note, settings)
    export_key = f"markdown-export-{note.id}"
    if st.button("Markdown 내보내기 준비", use_container_width=True, key=f"prepare-{export_key}"):
        export_note = repository.get(note.id)
        if export_note is not None:
            st.session_state[export_key] = build_markdown_export(export_note).encode("utf-8")
    if export_key in st.session_state:
        st.download_button(
            "준비된 Markdown 다운로드",
            data=st.session_state[export_key],
            file_name=f"{note.title}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"download-{export_key}",
        )


def _render_workspace(
    repository: LectureRepository,
    note: Optional[LectureRecord],
    settings: Settings,
    model_status: ModelStatus,
    prompt_manager: PromptManager,
    theme_mode: str,
    text_scale: int,
) -> None:
    render_pane_title("노트 작업공간", "RECORDING · SUMMARY · TRANSCRIPT · MEMO")
    if note is None:
        render_empty_workspace()
        return

    render_lecture_header(note)
    render_metrics(note)
    if note.status == "failed":
        render_error(note.error_message or "노트 처리에 실패했습니다.")

    is_document = note.media_type.lower() in DOCUMENT_EXTENSIONS
    # Only mount the active view: PDF/media/editors otherwise stay in browser memory.
    sections = ["자료 · 녹음", "전체 텍스트", "내 메모"]
    remembered = st.session_state.get(f"active-note-section-{note.id}", sections[0])
    section = _button_choice(
        "노트 화면",
        sections,
        remembered,
        key=f"note-section-{note.id}",
        show_label=False,
    )
    st.session_state[f"active-note-section-{note.id}"] = section
    if section == "전체 텍스트":
        full_note = repository.get(note.id)
        if full_note is not None:
            note = full_note
        quality = assess_transcription_quality(note.segments, note.duration)
        if quality.suspicious:
            render_warning(
                "녹취 품질 확인 필요",
                " ".join(quality.warnings)
                + " 이 녹취로 재요약하면 잘못된 내용이 유지될 수 있습니다. "
                "개선된 음성 인식으로 원본을 다시 처리해 주세요.",
            )
    if section == "자료 · 녹음":
        def preview(source):
            _render_recording(repository, replace(note, id=source["id"], source_name=source["name"],
                source_path=source["path"], media_type=source["media_type"], file_size=source["file_size"],
                has_media=False))
        render_source_panel(repository, note, settings, get_pipeline, model_status.ready,
                            prompt_manager, preview)
        if note.source_name and st.checkbox("기존 원본 미리보기", key=f"legacy-preview-{note.id}"):
            _render_recording(repository, note)
        st.divider()
        st.markdown("#### 빠른 메모")
        _render_note_memo(repository, note, settings, height=180)

    elif section == "전체 텍스트":
        if is_document or repository.list_sources(note.id):
            if note.transcript.strip():
                st.markdown(note.transcript)
                if note.segments:
                    with st.expander("음성 녹취 수정"):
                        event = render_editable_transcript(note.segments, key=f"mixed-transcript-{note.id}",
                            theme=theme_mode, text_scale=text_scale)
                        _handle_editor_event(event, repository, note, settings)
            else:
                render_info("PDF 텍스트 없음", "이 PDF에서 추출해 저장한 텍스트가 없습니다.")
        else:
            transcript_event = render_editable_transcript(
                note.segments,
                key=f"editable-transcript-{note.id}",
                theme=theme_mode,
                text_scale=text_scale,
            )
            _handle_editor_event(transcript_event, repository, note, settings)

    elif section == "내 메모":
        _render_note_memo(repository, note, settings, height=420)


def _render_folder_workspace(
    repository: LectureRepository,
    folder: FolderRecord,
    model_status: ModelStatus,
) -> None:
    notes = repository.list_by_folder(folder.id, lightweight=True)
    render_pane_title(folder.name, "FOLDER · LOCAL AI CHAT")
    st.markdown(
        f"### 📁 {folder.name}\n"
        f"이 폴더의 노트 {len(notes)}개를 한 번에 살펴보고 로컬 AI와 대화할 수 있습니다."
    )

    if notes:
        st.caption("노트를 열려면 아래 항목을 클릭하세요.")
        columns = st.columns(min(3, len(notes)), gap="small")
        for index, folder_note in enumerate(notes):
            with columns[index % len(columns)]:
                if st.button(
                    f"▤  {folder_note.title}",
                    key=f"folder-note-{folder_note.id}",
                    use_container_width=True,
                ):
                    st.session_state["selected_note_id"] = folder_note.id
                    st.session_state.pop("selected_folder_id", None)
                    st.rerun()
    else:
        render_info(
            "빈 폴더",
            "왼쪽에서 노트를 우클릭해 이 폴더로 이동하면 폴더 전용 채팅을 시작할 수 있습니다.",
        )

    st.divider()
    chat_title, clear_column = st.columns([4, 1], gap="small")
    with chat_title:
        st.markdown("#### 폴더 자료와 대화")
        st.caption("AI 요약, 내 메모, 전체 녹취를 검색해 답합니다.")
    with clear_column:
        if st.button(
            "대화 지우기",
            use_container_width=True,
            disabled=not bool(repository.list_chat_messages(folder.id, limit=1)),
            key=f"clear-chat-{folder.id}",
        ):
            repository.clear_chat_messages(folder.id)
            st.rerun()

    messages = repository.list_chat_messages(folder.id)
    if not messages:
        st.markdown(
            '<div class="lass-chat-empty">이 폴더의 핵심 내용을 물어보거나, '
            '노트 사이의 공통점과 차이점을 질문해 보세요.</div>',
            unsafe_allow_html=True,
        )
    for message in messages:
        with st.chat_message(message.role):
            st.markdown(message.content)

    has_reference = repository.folder_has_content(folder.id)
    disabled_reason = None
    if not model_status.ready:
        disabled_reason = "로컬 EXAONE 모델을 먼저 준비해 주세요."
    elif not has_reference:
        disabled_reason = "이 폴더에 요약, 메모 또는 녹취가 있는 노트가 필요합니다."
    if disabled_reason:
        st.caption(disabled_reason)
    question = st.chat_input(
        f"'{folder.name}' 폴더의 자료에 질문하세요",
        key=f"folder-chat-input-{folder.id}",
        disabled=disabled_reason is not None,
    )
    if question:
        history = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        repository.add_chat_message(folder.id, "user", question)
        try:
            with st.spinner("답변 생성 중…"):
                answer = get_pipeline().answer_folder_question(
                    folder.id,
                    folder.name,
                    lambda: repository.iter_by_folder(folder.id),
                    question,
                    history,
                )
        except Exception as exc:
            st.error(str(exc))
        else:
            repository.add_chat_message(folder.id, "assistant", answer)
            st.rerun()


def _ui_preferences(repository: LectureRepository) -> tuple[str, int]:
    """The supplied reference has one fixed light appearance."""
    del repository
    return "light", 100


def _render_model_profile_manager(
    registry: ModelRegistry,
    kind: str,
) -> None:
    profiles = registry.profiles(kind)
    active = registry.active(kind)
    labels = {profile.key: profile.name for profile in profiles}
    selected_key = st.selectbox(
        f"사용할 {kind.upper()} 모델",
        options=[profile.key for profile in profiles],
        index=[profile.key for profile in profiles].index(active.key),
        format_func=lambda key: labels[key],
        key=f"active-model-{kind}",
    )
    if selected_key != active.key:
        registry.set_active(kind, selected_key)
        _get_pipeline_for_profiles.clear()
        st.session_state["flash_message"] = f"{kind.upper()} 모델을 변경했습니다."
        st.rerun()

    selected = next(profile for profile in profiles if profile.key == selected_key)
    location = selected.runtime_location
    st.caption(
        f"식별자: {selected.identifier} · 위치: {location} · "
        f"엔진: {selected.runtime_backend} · 장치: {selected.device}"
    )

    if selected.runtime_backend == "mlx":
        devices = ["auto", "mps"]
    else:
        devices = ["auto", "cpu", "cuda"]
    if selected.runtime_backend not in {"faster-whisper", "mlx"}:
        devices.append("mps")
    current_device = selected.device if selected.device in devices else "auto"
    device_choice = st.selectbox("실행 장치", devices, index=devices.index(current_device),
        format_func=lambda value: {"auto": "자동 (사용 가능한 GPU 우선)", "cpu": "CPU",
                                  "cuda": "NVIDIA GPU (CUDA)", "mps": "Apple M칩 GPU (Metal / MPS)"}[value],
        key=f"profile-device-{selected.key}-{selected.device}")
    if device_choice != selected.device:
        registry.update_device(selected.key, device_choice)
        _get_pipeline_for_profiles.clear()
        st.rerun()
    if selected.runtime_backend == "mlx":
        st.caption("4비트 가중치와 8비트 KV 캐시를 사용하는 Apple Silicon 메모리 절약 엔진입니다.")
    elif selected.runtime_backend == "faster-whisper":
        st.caption("faster-whisper는 Mac에서 CPU로 실행됩니다. Apple GPU 전사에는 SenseVoice를 선택하세요.")
    elif selected.runtime_backend == "bigvox":
        st.caption(
            "한국어 전용 Bigvox 체크포인트와 포함된 Whisper-large-v3 "
            "음성 인코더를 사용합니다. 최초 실행 시 모델을 다운로드합니다."
        )
    else:
        st.caption("Apple M칩에서는 자동 또는 Apple GPU를 선택하면 MPS를 사용합니다. GPU도 통합 메모리를 함께 사용합니다.")

    with st.form(f"add-model-{kind}", clear_on_submit=True):
        st.markdown(f"**{kind.upper()} 모델 추가**")
        name = st.text_input("표시 이름", key=f"model-name-{kind}")
        identifier = st.text_input(
            "모델 식별자",
            placeholder=("large-v3-turbo" if kind == "stt" else "organization/model-name"),
            key=f"model-id-{kind}",
        )
        path = st.text_input(
            "로컬 경로 (선택)" if kind == "stt" else "로컬 모델 경로",
            placeholder="/absolute/path/to/model",
            key=f"model-path-{kind}",
        )
        if kind == "stt":
            backend = st.selectbox(
                "실행 엔진",
                ["faster-whisper", "sensevoice", "bigvox"],
                key="model-backend-stt",
            )
        else:
            llm_backends = ["transformers"]
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                llm_backends.insert(0, "mlx")
            backend = st.selectbox(
                "실행 엔진",
                llm_backends,
                key="model-backend-llm",
            )
        if backend == "mlx":
            device_options = ["auto", "mps"]
        elif backend == "faster-whisper":
            device_options = ["auto", "cpu", "cuda"]
        else:
            device_options = ["auto", "cpu", "cuda", "mps"]
        device = st.selectbox(
            "장치", device_options, key=f"model-device-{kind}-{backend}"
        )
        if kind == "stt":
            if backend == "sensevoice":
                compute_options = ["float32", "float16"]
            elif backend == "bigvox":
                compute_options = ["float16", "float32"]
            else:
                compute_options = ["int8", "float16", "float32"]
            compute_type = st.selectbox(
                "연산 정밀도",
                compute_options,
                key=f"model-compute-stt-{backend}",
            )
        else:
            compute_type = "int4" if backend == "mlx" else "auto"
        add_clicked = st.form_submit_button("목록에 추가", use_container_width=True)
        if add_clicked:
            if kind == "llm" and not path.strip():
                st.error("LLM은 로컬 모델 폴더를 입력해 주세요.")
            else:
                try:
                    profile = registry.add(
                        kind=kind,
                        name=name,
                        identifier=identifier,
                        path=path,
                        device=device,
                        compute_type=compute_type,
                        backend=backend,
                    )
                    registry.set_active(kind, profile.key)
                    _get_pipeline_for_profiles.clear()
                except (KeyError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    st.session_state["flash_message"] = f"{profile.name} 모델을 추가했습니다."
                    st.rerun()

    remove_column, remove_button = st.columns([2.2, 1], gap="small")
    with remove_column:
        remove_key = st.selectbox(
            "목록에서 제거",
            options=[profile.key for profile in profiles],
            format_func=lambda key: labels[key],
            label_visibility="collapsed",
            key=f"remove-model-{kind}",
        )
    with remove_button:
        if st.button(
            "제거",
            key=f"remove-model-button-{kind}",
            disabled=len(profiles) <= 1,
            use_container_width=True,
        ):
            removed = registry.remove(remove_key)
            _get_pipeline_for_profiles.clear()
            st.session_state["flash_message"] = (
                f"{removed.name}을 목록에서 제거했습니다. 모델 파일은 유지됩니다."
            )
            st.rerun()


def _render_preferences(
    repository: LectureRepository,
    registry: ModelRegistry,
    theme_mode: str,
    text_scale: int,
) -> None:
    del repository, theme_mode, text_scale
    st.caption("화면은 LASS 작업공간 디자인으로 고정됩니다.")
    st.caption("저메모리 모드: 모델을 단계별로 해제하고 요약 입력을 제한합니다.")
    stt_tab, llm_tab = st.tabs(["STT", "LLM"])
    with stt_tab:
        _render_model_profile_manager(registry, "stt")
    with llm_tab:
        _render_model_profile_manager(registry, "llm")


def _render_layout_controls(
    repository: LectureRepository,
    registry: ModelRegistry,
    theme_mode: str,
    text_scale: int,
    *,
    has_ai: bool,
) -> tuple[bool, bool, str]:
    """Keep secondary panes optional and expose practical width presets."""
    with st.container(key="lass-layout-toolbar"):
        visibility_column, width_column, settings_column = st.columns(
            [1.2, 1, 0.22], gap="small"
        )
        with visibility_column:
            st.caption("패널 표시 · 선택된 버튼을 다시 누르면 닫힙니다")
            panel_columns = st.columns(2 if has_ai else 1, gap="small")
            library_key = f"layout-library-{'note' if has_ai else 'folder'}"
            show_library = st.session_state.get(library_key, True)
            with panel_columns[0]:
                if st.button(
                    "✓ 자료 목록" if show_library else "자료 목록",
                    type="primary" if show_library else "secondary",
                    use_container_width=True,
                    key=f"toggle-{library_key}",
                ):
                    show_library = not show_library
                    st.session_state[library_key] = show_library
            show_ai = False
            if has_ai:
                ai_key = "layout-ai-note"
                show_ai = st.session_state.get(ai_key, True)
                with panel_columns[1]:
                    if st.button(
                        "✓ AI 노트" if show_ai else "AI 노트",
                        type="primary" if show_ai else "secondary",
                        use_container_width=True,
                        key=f"toggle-{ai_key}",
                    ):
                        show_ai = not show_ai
                        st.session_state[ai_key] = show_ai
        with width_column:
            if show_ai:
                saved_width = st.session_state.get("layout-width-mode", "균형")
                width_mode = _button_choice(
                    "패널 너비",
                    ["본문 넓게", "균형", "AI 넓게"],
                    saved_width,
                    key="layout-width-mode",
                )
                st.session_state["layout-width-mode"] = width_mode
            else:
                st.caption("필요한 패널만 켜 두면 작업 공간이 자동으로 넓어집니다.")
                width_mode = "본문 넓게"
        with settings_column:
            st.caption("설정")
            with st.popover(
                "설정",
                icon=":material/settings:",
                use_container_width=True,
            ):
                _render_preferences(
                    repository, registry, theme_mode, text_scale
                )
    return show_library, show_ai, width_mode


def _layout_ratios(show_library: bool, show_ai: bool, width_mode: str) -> list[float]:
    if show_library and show_ai:
        return {
            "본문 넓게": [0.66, 2.12, 0.92],
            "AI 넓게": [0.66, 1.28, 1.56],
        }.get(width_mode, [0.72, 1.66, 1.12])
    if show_library:
        return [0.72, 2.78]
    if show_ai:
        return [2.12, 1.38] if width_mode != "AI 넓게" else [1.55, 1.95]
    return [1.0]


def _toggle_library_panel() -> None:
    st.session_state["library_open"] = not st.session_state.get("library_open", True)


def _toggle_right_panel() -> None:
    st.session_state["right_panel_open"] = not st.session_state.get(
        "right_panel_open", True
    )


def _toggle_center_layout() -> None:
    current = st.session_state.get("center-layout", "stack")
    st.session_state["center-layout"] = "side" if current == "stack" else "stack"


def _open_summary_panel() -> None:
    st.session_state["right_panel_open"] = True
    st.session_state["right-panel-mode"] = "요약"


def _toggle_library_folder(folder_id: str) -> None:
    key = f"library-folder-open-{folder_id}"
    st.session_state[key] = not st.session_state.get(key, True)


def _render_sidebar_header() -> None:
    """Render the reference app identity inside the fixed library pane."""
    with st.container(key="lass-sidebar-header"):
        brand, close = st.columns([1, 0.18], gap="small")
        with brand:
            st.markdown(
                '<div class="lass-global-brand"><span class="lass-brand-mark">LA</span>'
                '<span><strong>LASS</strong><small>Local AI Study Studio</small></span></div>',
                unsafe_allow_html=True,
            )
        with close:
            st.button(
                "목록 닫기",
                icon=":material/left_panel_close:",
                key="close-library-panel",
                on_click=_toggle_library_panel,
                use_container_width=True,
            )


def _render_sidebar_footer(
    repository: LectureRepository,
    registry: ModelRegistry,
    theme_mode: str,
    text_scale: int,
) -> None:
    with st.container(key="lass-sidebar-footer"):
        with st.container(key="lass-header-settings"):
            with st.popover("설정", icon=":material/settings:", use_container_width=True):
                _render_preferences(repository, registry, theme_mode, text_scale)


def _render_library_buttons(
    repository: LectureRepository,
    settings: Settings,
    notes: list[LectureRecord],
    selected_id: Optional[str],
    theme_mode: str,
    text_scale: int,
) -> None:
    """Render the library with its folder/note context-management menu."""
    with st.container(key="lass-library-tools"):
        chatting_column, quiz_column = st.columns(2, gap="small")
        with chatting_column:
            st.button(
                "Chating",
                icon=":material/chat:",
                use_container_width=True,
                key="chatting-button",
            )
        with quiz_column:
            st.button(
                "Quiz",
                icon=":material/quiz:",
                use_container_width=True,
                key="quiz-button",
            )

    render_pane_title("폴더 및 노트", "목록")
    folders = repository.list_folders()
    with st.container(key="lass-library-content"):
        event = render_library_tree(
            folders,
            notes,
            selected_id,
            theme=theme_mode,
            text_scale=text_scale,
        )
    _handle_library_event(event, repository, settings)


def _desktop_file_path(kind: str) -> Optional[Path]:
    """Use the desktop chooser so even multi-GB sources never enter browser RAM."""
    picker = os.environ.get("LASS_DESKTOP_PICKER")
    if not picker:
        raise RuntimeError("데스크톱 LASS에서 파일 선택을 사용할 수 있습니다.")
    separator = "&" if "?" in picker else "?"
    request = Request(f"{picker}{separator}kind={kind}", data=b"", method="POST")
    try:
        with urlopen(request, timeout=300) as response:
            payload = json.load(response)
    except HTTPError as exc:
        # Launchers built before the split picker did not accept query
        # parameters.  Falling back keeps an existing LASS.app usable while
        # the selected extension is still validated below.
        if exc.code != 404:
            raise
        with urlopen(Request(picker, data=b"", method="POST"), timeout=300) as response:
            payload = json.load(response)
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    selected = str(payload.get("path") or "").strip()
    return Path(selected).expanduser().resolve() if selected else None


def _attach_and_process_path(
    repository: LectureRepository,
    note: LectureRecord,
    settings: Settings,
    path: Path,
    *,
    kind: str,
    copied_upload: bool = False,
    display_name: Optional[str] = None,
    prompt_key: str = "default_summary",
) -> None:
    allowed = DOCUMENT_EXTENSIONS if kind == "document" else SUPPORTED_MEDIA_EXTENSIONS
    label = "자료" if kind == "document" else "녹음 파일"
    source = None
    try:
        path = path.expanduser().resolve()
        if path.suffix.lower() not in allowed:
            raise ValueError(f"지원하는 {label} 형식이 아닙니다.")
        source = repository.add_source(note.id, path, display_name or path.name)
        st.session_state[f"active-{kind}-source-{note.id}"] = source["id"]
        if kind == "audio":
            _release_live_transcriber()
        status = st.empty()
        status.info("텍스트 추출 중…")
        process_source(
            repository,
            source,
            settings,
            get_pipeline,
            prompt_key,
            None,
            False,
            lambda message: status.info(message),
        )
    except Exception as exc:
        if source is None and copied_upload:
            path.unlink(missing_ok=True)
        if "status" in locals():
            status.error(f"{label} 처리를 완료하지 못했습니다.")
        st.error(str(exc))
    else:
        status.success(f"{label}을 저장했습니다.")
        if kind == "document" and copied_upload:
            state_key = f"browser-upload-open-document-{note.id}"
            version_key = f"browser-upload-version-document-{note.id}"
            st.session_state[state_key] = False
            st.session_state[version_key] = int(st.session_state.get(version_key, 0)) + 1
        st.session_state["flash_message"] = (
            "업로드한 원본과 추출된 텍스트를 노트에 저장했습니다."
            if copied_upload
            else "원본은 복사하지 않고, 추출된 텍스트만 노트에 저장했습니다."
        )
        st.rerun()


def _attach_and_process_local_file(
    repository: LectureRepository,
    note: LectureRecord,
    settings: Settings,
    *,
    kind: str,
) -> None:
    try:
        path = _desktop_file_path(kind)
    except Exception:
        st.session_state[f"browser-upload-open-{kind}-{note.id}"] = True
        st.rerun()
        return
    if path is not None:
        _attach_and_process_path(repository, note, settings, path, kind=kind)


def _interactive_summary_pages(
    repository: LectureRepository,
    note: LectureRecord,
    settings: Settings,
):
    """Build one bounded context spanning a long transcript for interactive use."""
    context_budget = min(3_200, max(1_200, settings.exaone_max_input_tokens))
    if note.transcript_chars <= context_budget:
        yield from repository.iter_transcript_pages(
            note.id, page_chars=context_budget
        )
        return

    source_page_chars = max(1_200, settings.chunk_size)
    source_page_count = max(1, math.ceil(note.transcript_chars / source_page_chars))
    excerpt_count = min(8, source_page_count)
    selected_indexes = {
        round(index * (source_page_count - 1) / max(1, excerpt_count - 1))
        for index in range(excerpt_count)
    }
    excerpt_budget = max(160, (context_budget - excerpt_count * 32) // excerpt_count)
    excerpts = []
    for index, page in enumerate(
        repository.iter_transcript_pages(note.id, page_chars=source_page_chars)
    ):
        if index not in selected_indexes:
            continue
        text = page.strip()
        if len(text) > excerpt_budget:
            half = max(1, (excerpt_budget - 3) // 2)
            text = f"{text[:half]} … {text[-half:]}"
        excerpts.append(
            f"[전체 자료 위치 {index + 1}/{source_page_count}]\n{text}"
        )
    if excerpts:
        yield "\n\n".join(excerpts)


def _attach_and_process_upload(
    repository: LectureRepository,
    note: LectureRecord,
    settings: Settings,
    upload: Any,
    *,
    kind: str,
) -> None:
    allowed = DOCUMENT_EXTENSIONS if kind == "document" else SUPPORTED_MEDIA_EXTENSIONS
    name = Path(str(upload.name)).name
    suffix = Path(name).suffix.lower()
    if suffix not in allowed:
        st.error("지원하는 파일 형식이 아닙니다.")
        return
    managed_dir = settings.data_dir / "media"
    managed_dir.mkdir(parents=True, exist_ok=True)
    managed = managed_dir / f"{uuid4().hex}{suffix}"
    try:
        upload.seek(0)
        with managed.open("wb") as output:
            while chunk := upload.read(1024 * 1024):
                output.write(chunk)
    except Exception as exc:
        managed.unlink(missing_ok=True)
        st.error(f"파일을 저장하지 못했습니다: {exc}")
        return
    _attach_and_process_path(
        repository,
        note,
        settings,
        managed,
        kind=kind,
        copied_upload=True,
        display_name=name,
    )


def _render_browser_upload(
    repository: LectureRepository,
    note: LectureRecord,
    settings: Settings,
    *,
    kind: str,
) -> None:
    state_key = f"browser-upload-open-{kind}-{note.id}"
    if not st.session_state.get(state_key):
        return
    allowed = DOCUMENT_EXTENSIONS if kind == "document" else SUPPORTED_MEDIA_EXTENSIONS
    label = "PDF · PowerPoint" if kind == "document" else "녹음 · 영상 파일"
    version_key = f"browser-upload-version-{kind}-{note.id}"
    version = int(st.session_state.get(version_key, 0))
    upload = st.file_uploader(
        label,
        type=sorted(extension.lstrip(".") for extension in allowed),
        key=f"browser-upload-{kind}-{note.id}-{version}",
    )
    add_column, cancel_column = st.columns([1, 0.45], gap="small")
    with add_column:
        if st.button(
            "선택한 파일 추가",
            type="primary",
            use_container_width=True,
            disabled=upload is None,
            key=f"browser-upload-save-{kind}-{note.id}",
        ):
            _attach_and_process_upload(
                repository, note, settings, upload, kind=kind
            )
    with cancel_column:
        if st.button(
            "취소",
            use_container_width=True,
            key=f"browser-upload-cancel-{kind}-{note.id}",
        ):
            st.session_state[state_key] = False
            st.session_state[version_key] = version + 1
            st.rerun()


def _dismiss_document_upload_dialog() -> None:
    note_id = st.session_state.get("document-upload-dialog-note-id")
    if note_id:
        st.session_state[f"browser-upload-open-document-{note_id}"] = False


@st.dialog(
    "PDF · PPT 추가", width="small", on_dismiss=_dismiss_document_upload_dialog
)
def _render_document_upload_dialog(
    repository: LectureRepository,
    note: LectureRecord,
    settings: Settings,
) -> None:
    st.caption("아래 영역을 클릭하거나 파일을 끌어다 놓으세요.")
    _render_browser_upload(repository, note, settings, kind="document")


def _document_source_note(note: LectureRecord, source: dict[str, Any]) -> LectureRecord:
    return replace(
        note,
        id=f"{note.id}-{source['id']}",
        source_name=source["name"],
        source_path=source["path"],
        media_type=source["media_type"],
        file_size=source["file_size"],
        has_media=False,
    )


def _document_sources(
    repository: LectureRepository, note: LectureRecord
) -> list[dict[str, Any]]:
    sources = [
        source
        for source in repository.list_sources(note.id)
        if source["media_type"] in DOCUMENT_EXTENSIONS
    ]
    if note.source_path and note.media_type in DOCUMENT_EXTENSIONS:
        sources.insert(
            0,
            {
                "id": "legacy",
                "name": note.source_name,
                "path": note.source_path,
                "media_type": note.media_type,
                "file_size": note.file_size,
                "status": note.status,
                "error_message": note.error_message,
            },
        )
    return sources


def _active_document_source(
    note: LectureRecord, sources: list[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    if not sources:
        return None
    active_key = f"active-document-source-{note.id}"
    source_ids = [source["id"] for source in sources]
    current_id = st.session_state.get(active_key, source_ids[-1])
    if current_id not in source_ids:
        current_id = source_ids[-1]
        st.session_state[active_key] = current_id
    return next(source for source in sources if source["id"] == current_id)


def _render_document_upload_action(
    repository: LectureRepository,
    note: LectureRecord,
    settings: Settings,
) -> None:
    with st.container(key="lass-document-actions"):
        if st.button(
            "PDF · PPT 추가",
            icon=":material/upload_file:",
            key=f"document-upload-{note.id}",
        ):
            if os.environ.get("LASS_DESKTOP_PICKER"):
                _attach_and_process_local_file(
                    repository, note, settings, kind="document"
                )
            else:
                upload_key = f"browser-upload-open-document-{note.id}"
                st.session_state[upload_key] = not st.session_state.get(
                    upload_key, False
                )
                st.rerun()


def _render_center_panel(
    repository: LectureRepository,
    note: Optional[LectureRecord],
    settings: Settings,
) -> None:
    sources = _document_sources(repository, note) if note else []
    source = _active_document_source(note, sources) if note else None
    document_note = _document_source_note(note, source) if note and source else None
    preview = None
    if document_note and document_note.media_type.lower() == PDF_EXTENSION:
        try:
            preview = _pdf_preview(repository, document_note)
        except (OSError, RuntimeError, ValueError):
            preview = None

    folder_name = "내 노트"
    if note and note.folder_id:
        folder = repository.get_folder(note.folder_id)
        if folder is not None:
            folder_name = folder.name
    source_name = (
        str(source["name"]) if source else (note.source_name or note.title)
    ) if note else "새 노트를 선택하세요"
    status_label = {
        "ready": "Processed",
        "processing": "Processing",
        "failed": "Needs attention",
        "draft": "Draft",
    }.get(note.status, note.status) if note else "Waiting"
    status_class = note.status if note else "draft"
    page_number = preview[2] if preview else 0
    page_count = preview[3] if preview else 0
    pdf_key = document_note.id if document_note else "empty"
    page_key = f"pdf-page-{pdf_key}"
    zoom_key = f"pdf-zoom-{pdf_key}"
    fit_key = f"pdf-fit-{pdf_key}"
    zoom = min(200, max(50, int(st.session_state.get(zoom_key, 100))))
    fit = bool(st.session_state.get(fit_key, True))
    center_layout = st.session_state.get("center-layout", "stack")

    with st.container(key="lass-center-header"):
        (
            sidebar_column,
            breadcrumb_column,
            layout_column,
            upload_column,
            pages_column,
            zoom_column,
            right_column,
        ) = st.columns(
            [0.12, 1.15, 0.56, 0.56, 0.42, 0.44, 0.12], gap="small"
        )
        with sidebar_column:
            st.button(
                "노트 목록",
                icon=":material/left_panel_open:",
                key="toggle-library-panel",
                on_click=_toggle_library_panel,
                use_container_width=True,
            )
        with breadcrumb_column:
            st.markdown(
                '<div class="lass-document-breadcrumb">'
                f'<span>{escape(folder_name)}</span><i>›</i><b>{escape(source_name)}</b>'
                f'<em class="{escape(status_class)}">{escape(status_label)}</em></div>',
                unsafe_allow_html=True,
            )
        with pages_column:
            previous, counter, following = st.columns([1, 1.25, 1], gap="small")
            with previous:
                if st.button(
                    "←",
                    key=f"pdf-previous-{pdf_key}",
                    disabled=page_number <= 1,
                    use_container_width=True,
                ):
                    st.session_state[page_key] = page_number - 1
                    st.rerun()
            with counter:
                st.markdown(
                    f'<span class="lass-pdf-counter">{page_number} / {page_count}</span>',
                    unsafe_allow_html=True,
                )
            with following:
                if st.button(
                    "→",
                    key=f"pdf-next-{pdf_key}",
                    disabled=not page_count or page_number >= page_count,
                    use_container_width=True,
                ):
                    st.session_state[page_key] = page_number + 1
                    st.rerun()
        with zoom_column:
            zoom_out, zoom_value, zoom_in, fit_column = st.columns([1, 1, 1, 2.2], gap="small")
            with zoom_out:
                if st.button(
                    "−",
                    key=f"pdf-zoom-out-{pdf_key}",
                    disabled=preview is None or zoom <= 50,
                    use_container_width=True,
                ):
                    st.session_state[zoom_key] = zoom - 25
                    st.session_state[fit_key] = False
                    st.rerun()
            with zoom_value:
                st.markdown(
                    f'<span class="lass-pdf-zoom-value" data-zoom="{zoom / 100:.2f}">'
                    f'{zoom}%</span>',
                    unsafe_allow_html=True,
                )
            with zoom_in:
                if st.button(
                    "＋",
                    key=f"pdf-zoom-in-{pdf_key}",
                    disabled=preview is None or zoom >= 200,
                    use_container_width=True,
                ):
                    st.session_state[zoom_key] = zoom + 25
                    st.session_state[fit_key] = False
                    st.rerun()
            with fit_column:
                if st.button(
                    "높이 맞춤" if center_layout == "stack" else "너비 맞춤",
                    key=f"pdf-fit-toggle-{pdf_key}",
                    disabled=preview is None,
                    type="primary" if fit else "secondary",
                    use_container_width=True,
                ):
                    st.session_state[fit_key] = not fit
                    if not fit:
                        st.session_state[zoom_key] = 100
                    st.rerun()
        with layout_column:
            st.button(
                "좌우 배치" if center_layout == "stack" else "상하 배치",
                icon=(
                    ":material/view_column:"
                    if center_layout == "stack"
                    else ":material/view_agenda:"
                ),
                key="toggle-center-layout",
                on_click=_toggle_center_layout,
                use_container_width=True,
            )
        with upload_column:
            if note is not None:
                _render_document_upload_action(repository, note, settings)
        with right_column:
            st.button(
                "AI 패널",
                icon=":material/right_panel_open:",
                key="toggle-right-panel",
                on_click=_toggle_right_panel,
                use_container_width=True,
            )

    with st.container(key="lass-document-memo-layout"):
        with st.container(key="lass-document-pane"):
            if note is None:
                render_empty_workspace()
            elif not sources:
                with st.container(key="lass-document-empty-viewer"):
                    st.markdown(
                        '<div class="lass-document-empty-copy"><span>▱</span>'
                        '<b>PDF가 아직 없습니다.</b>'
                        '<small>이 뷰어에서 PDF나 PowerPoint를 바로 추가하세요.</small></div>',
                        unsafe_allow_html=True,
                    )
            else:
                active_key = f"active-document-source-{note.id}"
                source_ids = [source["id"] for source in sources]
                current_id = source["id"]
                if len(sources) > 1:
                    selected_id = st.selectbox(
                        "표시 자료",
                        source_ids,
                        index=source_ids.index(current_id),
                        format_func=lambda value: next(
                            source["name"] for source in sources if source["id"] == value
                        ),
                        key=active_key,
                        label_visibility="collapsed",
                    )
                    # The header controls were built from `source` earlier in this
                    # run. Never render a viewer with a newly selected source and
                    # the previous source's page controls; restart with one source
                    # as the single authority for both header and body.
                    if selected_id != source["id"]:
                        st.rerun()
                if source.get("error_message"):
                    render_warning("텍스트 추출 확인 필요", str(source["error_message"]))
                if source.get("status") != "ready" and source.get("id") != "legacy":
                    render_warning(
                        "텍스트 추출이 완료되지 않았습니다",
                        "중단된 PDF/PPT 처리를 다시 시작하면 요약 기능이 활성화됩니다.",
                    )
                    if st.button(
                        "텍스트 추출 다시 시작",
                        key=f"retry-document-processing-{source['id']}",
                        use_container_width=True,
                    ):
                        extraction_status = st.empty()
                        try:
                            process_source(
                                repository,
                                source,
                                settings,
                                get_pipeline,
                                "default_summary",
                                None,
                                False,
                                extraction_status.info,
                            )
                        except Exception as exc:
                            extraction_status.error("텍스트 추출에 실패했습니다.")
                            st.error(str(exc))
                        else:
                            st.session_state["flash_message"] = (
                                "문서 텍스트 추출을 완료했습니다."
                            )
                            st.rerun()
                _render_recording(
                    repository,
                    document_note,
                    pdf_preview=preview,
                    pdf_zoom=zoom,
                    pdf_fit=fit,
                )

        if note is not None:
            with st.container(key="lass-memo-pane"):
                st.markdown(
                    '<div class="lass-memo-title"><span>✎</span><b>PERSONAL MEMO</b></div>',
                    unsafe_allow_html=True,
                )
                _render_note_memo(
                    repository,
                    note,
                    settings,
                    height=190,
                    theme_mode="light",
                    text_scale=100,
                )

    if note is not None and st.session_state.get(
        f"browser-upload-open-document-{note.id}"
    ):
        st.session_state["document-upload-dialog-note-id"] = note.id
        _render_document_upload_dialog(repository, note, settings)


def _get_live_transcriber(registry: ModelRegistry):
    profile = registry.active("stt")
    signature = (
        profile.runtime_location,
        profile.runtime_backend,
        profile.device,
        profile.compute_type,
    )
    if st.session_state.get("_live_transcriber_signature") != signature:
        _release_live_transcriber()
        st.session_state["_live_transcriber"] = create_transcriber(
            backend=profile.runtime_backend,
            model_name=profile.runtime_location,
            device=profile.device,
            compute_type=profile.compute_type,
            target_lufs=get_settings().stt_target_lufs,
        )
        st.session_state["_live_transcriber_signature"] = signature
    return st.session_state["_live_transcriber"]


def _release_live_transcriber() -> None:
    transcriber = st.session_state.pop("_live_transcriber", None)
    st.session_state.pop("_live_transcriber_signature", None)
    if transcriber is not None:
        transcriber.unload()


def _handle_live_recorder_event(
    event: Optional[dict[str, Any]],
    repository: LectureRepository,
    registry: ModelRegistry,
    note: LectureRecord,
) -> None:
    if not isinstance(event, dict):
        return
    nonce = str(event.get("nonce") or "")
    processed_key = f"live-processed-{note.id}"
    if not nonce or nonce == st.session_state.get(processed_key):
        return
    st.session_state[processed_key] = nonce
    action = event.get("action")
    if action == "error":
        st.session_state[f"live-error-{note.id}"] = str(event.get("message") or "녹음을 시작하지 못했습니다.")
        return
    if action != "audio_chunk":
        return

    final = bool(event.get("final"))
    temporary_path: Optional[Path] = None
    try:
        encoded = str(event.get("data") or "")
        if not encoded or len(encoded) > 14_000_000:
            raise ValueError("녹음 구간의 크기가 올바르지 않습니다.")
        audio = base64.b64decode(encoded, validate=True)
        suffix = str(event.get("extension") or ".webm")
        if suffix not in {".webm", ".m4a", ".ogg", ".wav"}:
            suffix = ".webm"
        with tempfile.NamedTemporaryFile(prefix="lass-live-", suffix=suffix, delete=False) as output:
            output.write(audio)
            temporary_path = Path(output.name)
        del audio, encoded
        with st.spinner("텍스트 추출 중…"):
            result = _get_live_transcriber(registry).transcribe(temporary_path)
        session_key = f"live-session-started-{note.id}"
        repository.append_live_transcription(
            note.id,
            transcript=result.text,
            segments=result.segments,
            duration=result.duration,
            language=result.language,
            start_session=not st.session_state.get(session_key, False),
        )
        st.session_state[session_key] = not final
        st.session_state.pop(f"live-error-{note.id}", None)
    except Exception as exc:
        st.session_state[f"live-error-{note.id}"] = str(exc)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        st.session_state[f"live-ack-{note.id}"] = nonce
        if final:
            st.session_state[f"live-session-started-{note.id}"] = False
            _release_live_transcriber()
    st.rerun()


def _render_recording_panel(
    repository: LectureRepository,
    registry: ModelRegistry,
    note: LectureRecord,
    settings: Settings,
    theme_mode: str,
) -> None:
    with st.container(key="lass-transcript-stream"):
        error = st.session_state.get(f"live-error-{note.id}")
        if error:
            render_warning("녹음 확인 필요", error)
        segments = repository.recent_segments(note.id, limit=120)
        if segments:
            render_timeline(segments)
            if note.segment_count > len(segments):
                st.caption(f"최근 {len(segments)}개 구간을 표시합니다. 전체 텍스트는 디스크에 저장되어 있습니다.")
        else:
            st.markdown(
                '<div class="lass-live-empty"><span>●</span><b>녹음을 시작하면 실시간 STT가 여기에 표시됩니다.</b></div>',
                unsafe_allow_html=True,
            )

    with st.container(key="lass-recording-actions"):
        if st.button(
            "음성 파일 업로드",
            icon=":material/audio_file:",
            key=f"audio-upload-{note.id}",
            use_container_width=True,
        ):
            if os.environ.get("LASS_DESKTOP_PICKER"):
                _attach_and_process_local_file(
                    repository, note, settings, kind="audio"
                )
            else:
                upload_key = f"browser-upload-open-audio-{note.id}"
                st.session_state[upload_key] = not st.session_state.get(
                    upload_key, False
                )
                st.rerun()
        _render_browser_upload(repository, note, settings, kind="audio")
        event = render_live_recorder(
            ack_nonce=st.session_state.get(f"live-ack-{note.id}", ""),
            theme=theme_mode,
            key=f"live-recorder-{note.id}",
        )
    _handle_live_recorder_event(event, repository, registry, note)


def _render_right_panel(
    repository: LectureRepository,
    registry: ModelRegistry,
    note: Optional[LectureRecord],
    settings: Settings,
    model_status: ModelStatus,
    prompt_manager: PromptManager,
    theme_mode: str,
    text_scale: int,
) -> None:
    mode = st.session_state.get("right-panel-mode", "녹음")
    with st.container(key="lass-right-header"):
        close_tab, transcription_tab, summary_tab = st.columns([0.24, 1, 1], gap=None)
        with close_tab:
            st.button(
                "AI 패널 닫기",
                icon=":material/right_panel_close:",
                key="close-right-panel",
                on_click=_toggle_right_panel,
                use_container_width=True,
            )
        with transcription_tab:
            st.button(
                "Transcription",
                icon=":material/mic:",
                type="primary" if mode == "녹음" else "secondary",
                key="right-tab-transcription",
                on_click=lambda: st.session_state.__setitem__("right-panel-mode", "녹음"),
                use_container_width=True,
            )
        with summary_tab:
            st.button(
                "AI Summary",
                icon=":material/auto_awesome:",
                type="primary" if mode == "요약" else "secondary",
                key="right-tab-summary",
                on_click=_open_summary_panel,
                use_container_width=True,
            )
    if note is None:
        with st.container(key="lass-transcript-stream"):
            render_info("노트를 먼저 선택하세요", "녹음과 요약은 선택한 노트에 저장됩니다.")
        return
    if mode == "녹음":
        _render_recording_panel(repository, registry, note, settings, theme_mode)
        return

    with st.container(key="lass-summary-stream"):
        summary_event = render_editable_summary(
            note.summary,
            key=f"right-summary-{note.id}",
            theme=theme_mode,
            text_scale=text_scale,
        )
        _handle_editor_event(summary_event, repository, note, settings)
        templates = prompt_manager.list_templates()
        template_keys = [template.key for template in templates]
        default_key = note.prompt_key if note.prompt_key in template_keys else template_keys[0]
        with st.expander("요약 설정"):
            prompt_key = st.selectbox(
                "요약 형식",
                template_keys,
                index=template_keys.index(default_key),
                format_func=lambda value: next(
                    template.name for template in templates if template.key == value
                ),
                key=f"right-summary-template-{note.id}",
            )
            include_memo = st.checkbox(
                "내 메모 포함",
                value=True,
                disabled=not bool(note.notes.strip()),
                key=f"right-summary-memo-{note.id}",
            )
        summary_progress = st.empty()
    with st.container(key="lass-summary-actions"):
        if not model_status.ready:
            st.caption(
                "요약 모델 파일이 준비되지 않았습니다. 설정 → 모델 → LLM에서 "
                "현재 운영체제용 로컬 모델 경로를 확인해 주세요."
            )
        elif note.transcript_chars <= 0:
            st.caption("요약할 녹취 또는 문서 텍스트가 필요합니다.")
        summarize_column, export_column = st.columns([1, 0.9], gap="small")
        with summarize_column:
            if st.button(
                "다시 요약하기" if note.summary.strip() else "요약하기",
                type="primary",
                use_container_width=True,
                disabled=not model_status.ready or note.transcript_chars <= 0,
                key=f"right-resummary-{note.id}",
            ):
                _release_live_transcriber()
                _resummarize_note(
                    repository,
                    note,
                    settings,
                    prompt_key,
                    include_memo,
                    status=summary_progress,
                )
        with export_column:
            export_note = repository.get(note.id) or note
            st.download_button(
                "Markdown 다운로드",
                data=build_markdown_export(export_note).encode("utf-8"),
                file_name=f"{note.title.replace('/', '_')}.md",
                mime="text/markdown",
                icon=":material/download:",
                use_container_width=True,
                key=f"right-markdown-export-{note.id}",
            )


def main() -> None:
    settings = get_settings()
    repository = get_repository()
    get_vector_database()
    prompt_manager = PromptManager(settings.prompt_dir)
    migration = migrate_legacy_storage(repository, settings.data_dir)
    registry = get_model_registry()
    llm_profile = registry.active("llm")
    model_status = LocalModelManager(
        llm_profile.identifier, Path(llm_profile.runtime_location)
    ).status()
    theme_mode, text_scale = _ui_preferences(repository)
    notes = repository.list(lightweight=True)
    st.session_state.pop("selected_folder_id", None)
    selected = _selected_note(repository, notes)

    inject_theme(theme_mode, text_scale)

    migrated_count = migration.imported_reports + migration.embedded_media
    if migrated_count:
        st.toast(f"기존 노트 {migrated_count}개를 lass.db 안으로 옮겼습니다.")
    if migration.warnings:
        st.toast(" ".join(migration.warnings), icon="⚠️")
    flash_message = st.session_state.pop("flash_message", None)
    if flash_message:
        st.toast(flash_message)
    flash_error = st.session_state.pop("flash_error", None)
    if flash_error:
        st.toast(flash_error, icon="⚠️")
    if st.session_state.pop("upload_complete", False):
        st.toast("녹음·요약·전체 녹취·메모를 노트 하나로 저장했습니다.")
    delete_warnings = st.session_state.pop("delete_warnings", [])
    if delete_warnings:
        st.toast(" ".join(delete_warnings), icon="⚠️")

    show_library = st.session_state.get("library_open", True)
    show_right = st.session_state.get("right_panel_open", True)
    with st.container(key="lass-layout"):
        library_column, workspace_column, right_column = st.columns(3, gap=None)
        with library_column:
            if show_library:
                with st.container(key="lass-library-pane"):
                    _render_sidebar_header()
                    with st.container(key="lass-library-body"):
                        _render_library_buttons(
                            repository,
                            settings,
                            notes,
                            selected.id if selected else None,
                            theme_mode,
                            text_scale,
                        )
                    _render_sidebar_footer(
                        repository, registry, theme_mode, text_scale
                    )
        with workspace_column:
            with st.container(key="lass-workspace-pane"):
                _render_center_panel(repository, selected, settings)
        with right_column:
            if show_right:
                with st.container(key="lass-ai-pane"):
                    _render_right_panel(
                        repository,
                        registry,
                        selected,
                        settings,
                        model_status,
                        prompt_manager,
                        theme_mode,
                        text_scale,
                    )
        render_layout_resizer(
            st.session_state.get("center-layout", "stack"),
            transcript_version=selected.segment_count if selected else 0,
            transcript_active=(
                show_right
                and st.session_state.get("right-panel-mode", "녹음") == "녹음"
            ),
            key="lass-layout-resizer",
        )


if __name__ == "__main__":
    main()
