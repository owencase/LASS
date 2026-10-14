"""Streamlit entrypoint for the LASS note workspace."""

from __future__ import annotations

import base64
import json
import os
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
from src.pdf_utils import DOCUMENT_EXTENSIONS
from src.pdf_stream import LocalMediaServer
from src.ui.source_panel import process_source, render_source_panel
from src.pipeline import LASSPipeline
from src.summary_jobs import summary_jobs
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
    render_pane_title,
    render_timeline,
    render_warning,
)
from src.ui.editable_content import (
    render_editable_summary,
    render_editable_transcript,
    render_memo_editor,
)
from src.ui.library_tree import render_library_tree
from src.ui.layout_resizer import render_layout_resizer
from src.ui.document_viewer import render_document_viewer
from src.ui.live_recorder import render_live_recorder


st.set_page_config(
    page_title="LASS · Note Workspace",
    page_icon=str(Path(__file__).resolve().parents[1] / "assets/lass.png"),
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


@st.cache_resource(show_spinner=False)
def get_pipeline() -> LASSPipeline:
    pipeline = LASSPipeline(get_settings())
    pipeline.llm_client.keep_loaded = False
    pipeline.llm_client.temperature = 0
    return pipeline


def _selected_note(
    repository: LectureRepository, notes: list[LectureRecord]
) -> Optional[LectureRecord]:
    if st.session_state.get("new_note_mode") or st.session_state.get(
        "selected_folder_id"
    ):
        return None
    selected_id = st.session_state.get("selected_note_id") or repository.get_metadata("selected_note_id")
    selected = repository.get_workspace(selected_id) if selected_id else None
    if selected is None and notes:
        selected = repository.get_workspace(notes[0].id)
        st.session_state["selected_note_id"] = selected.id
    if selected is not None:
        st.session_state["selected_note_id"] = selected.id
        if repository.get_metadata("selected_note_id") != selected.id:
            repository.set_metadata("selected_note_id", selected.id)
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
            st.session_state["workspace-mode"] = "notes"
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
            # Expanding the library must never change the active workspace.
            st.session_state["library-browsed-folder-id"] = folder_id
        elif action == "create_note":
            st.session_state["workspace-mode"] = "notes"
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
def _cached_document_page(source, modified_ns, file_size, page_number, cache_dir):
    from src.document_preview import preview_page
    # Stat values invalidate this small metadata cache when a source changes.
    return preview_page(Path(source), Path(cache_dir), page_number)[:2]


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


def _document_preview(
    repository: LectureRepository, note: LectureRecord
) -> Optional[tuple[Path, Path, int, int]]:
    source = _media_path(repository, note)
    if source is None or note.media_type.lower() not in DOCUMENT_EXTENSIONS:
        return None
    page_key = f"pdf-page-{note.id}"
    page_number = max(1, int(st.session_state.get(page_key, 1)))
    stat = source.stat()
    try:
        page_data, page_count = _cached_document_page(
            str(source.resolve()), stat.st_mtime_ns, stat.st_size, page_number, str(get_settings().data_dir / "document-preview")
        )
    except ValueError:
        page_number = 1
        st.session_state[page_key] = 1
        page_data, page_count = _cached_document_page(
            str(source.resolve()), stat.st_mtime_ns, stat.st_size, page_number, str(get_settings().data_dir / "document-preview")
        )
    return source, page_data, page_number, page_count


def _render_pdf_viewer(
    note: LectureRecord,
    preview: tuple[Path, Path, int, int],
    *,
    zoom: int = 100,
    fit: bool = True,
) -> None:
    """Render one page without a second, conflicting PDF toolbar."""
    _, page_data, page_number, _ = preview
    url = _get_media_server().register(page_data)
    if page_data.suffix == ".svg":
        st.caption("PPTX 기본 미리보기 · 복잡한 도형·차트는 원본과 다를 수 있습니다.")
    render_document_viewer(url, note.title, page_number, zoom=zoom, fit=fit,
        layout=st.session_state.get("center-layout", "stack"), key=f"document-viewer-{note.id}")


def _render_recording(
    repository: LectureRepository,
    note: LectureRecord,
    *,
    pdf_preview: Optional[tuple[Path, Path, int, int]] = None,
    pdf_zoom: int = 100,
    pdf_fit: bool = True,
) -> None:
    path = _media_path(repository, note)
    if path is None:
        render_info("자료 없음", "원본 파일이 없거나 이동되었습니다. 자료를 다시 추가해 주세요.")
        return
    if note.media_type.lower() in DOCUMENT_EXTENSIONS:
        try:
            preview = pdf_preview or _document_preview(repository, note)
        except (OSError, RuntimeError, ValueError) as exc:
            render_error(str(exc))
            return
        if preview is not None:
            _render_pdf_viewer(note, preview, zoom=pdf_zoom, fit=pdf_fit)
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
            job = summary_jobs.snapshot(_summary_job_key(repository, note.id))
            if job and job.running:
                raise ValueError("요약이 완료된 후 수정할 수 있습니다.")
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


def _interactive_summary_pages(
    repository: LectureRepository,
    note: LectureRecord,
    settings: Settings,
):
    """Build one bounded context spanning a long transcript for interactive use."""
    context_budget = min(3_200, max(1_200, settings.exaone_max_input_tokens))
    source_page_chars = max(1_200, settings.chunk_size)
    source_page_count = 0
    source_chars = 0
    for page in repository.iter_transcript_pages(note.id, page_chars=source_page_chars):
        source_page_count += 1
        source_chars += len(page)
    if source_chars <= context_budget:
        yield from repository.iter_transcript_pages(
            note.id, page_chars=context_budget
        )
        return

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


def _resummarize_note(
    repository: LectureRepository,
    note: LectureRecord,
    settings: Settings,
    prompt_key: str,
    include_memo: bool,
) -> None:
    pipeline = get_pipeline()

    def work(progress, on_token):
        workspace_note = repository.get_workspace(note.id)
        if workspace_note is None:
            raise KeyError(f"Unknown lecture: {note.id}")
        result = pipeline.resummarize_pages(
            _interactive_summary_pages(repository, workspace_note, settings),
            prompt_key=prompt_key,
            notes=workspace_note.notes if include_memo else "",
            progress=progress,
            on_token=on_token,
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
        return result.summary

    summary_jobs.start(_summary_job_key(repository, note.id), work)
    st.rerun()


def _summary_job_key(repository, note_id):
    return str(repository.database_path), note_id


def _memo_option(note_id, key, running=False):
    # Widget state is discarded when navigating away; keep the choice separately.
    preference = f"include-memo-preference-{note_id}"

    def remember():
        st.session_state[preference] = st.session_state[key]

    return st.checkbox(
        "내 메모 포함", value=st.session_state.get(preference, True),
        key=key, disabled=running, on_change=remember,
    )


def _render_live_summary(repository, note, settings, theme_mode, text_scale, key):
    job = summary_jobs.snapshot(_summary_job_key(repository, note.id))
    interval = 0.5 if job and job.running else None
    st.fragment(run_every=interval)(_render_summary_contents)(repository, note, settings, theme_mode, text_scale, key)


def _render_summary_contents(repository, note, settings, theme_mode, text_scale, key):
    """Refresh only the single summary editor, leaving the document/memo intact."""
    job = summary_jobs.snapshot(_summary_job_key(repository, note.id))
    fresh = repository.get_workspace(note.id)
    if fresh is None:
        return
    running = bool(job and job.running)
    text = job.text if running else fresh.summary
    event = render_editable_summary(
        text, key=key, theme=theme_mode, text_scale=text_scale, streaming=running,
        progress=job.progress if running else "",
        error=f"{job.progress} {job.error}" if job and job.error else "",
    )
    if not running:
        _handle_editor_event(event, repository, fresh, settings)
    if job:
        seen_key = f"summary-job-seen-{key}"
        current = (job.id, job.running)
        previous = st.session_state.get(seen_key)
        st.session_state[seen_key] = current
        if previous == (job.id, True) and not running:
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

    selected_key = "lecture_detail"
    job = summary_jobs.snapshot(_summary_job_key(repository, note.id))
    include_memo = _memo_option(note.id, f"resummary-memo-{note.id}", bool(job and job.running))
    if st.button(
        "↻ AI 다시 요약하기",
        type="primary",
        use_container_width=True,
        disabled=not model_status.ready or note.transcript_chars <= 0 or bool(job and job.running),
        key=f"resummary-{note.id}",
    ):
        _resummarize_note(
            repository,
            note,
            settings,
            selected_key,
            include_memo,
        )
    _render_live_summary(repository, note, settings, theme_mode, text_scale, f"editable-summary-{note.id}")
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
            cached_pages = repository.list_document_pages(note.id)
            if cached_pages:
                index = st.selectbox("문서 텍스트 페이지", range(len(cached_pages)),
                    format_func=lambda i: f"{cached_pages[i]['name']} · {cached_pages[i]['page_number']}페이지/슬라이드",
                    key=f"document-text-page-{note.id}")
                page = cached_pages[index]
                st.markdown(next(repository.iter_document_pages(note.id, source_id=page['source_id'], page_number=page['page_number']))['text'])
            note = repository.get(note.id)
            if note.transcript.strip():
                st.markdown(note.transcript)
                if note.segments:
                    with st.expander("음성 녹취 수정"):
                        event = render_editable_transcript(note.segments, key=f"mixed-transcript-{note.id}",
                            theme=theme_mode, text_scale=text_scale)
                        _handle_editor_event(event, repository, note, settings)
            elif not cached_pages:
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


def _render_chat_sessions(repository: LectureRepository) -> None:
    with st.container(key="lass-chat-panel-header"):
        title, close = st.columns([4, 1], gap="small")
        with title:
            st.markdown("**대화 세션**")
        with close:
            st.button("대화 목록 닫기", icon=":material/left_panel_close:", key="close-chat-panel",
                      on_click=lambda: st.session_state.__setitem__("chat-panel-open", False))
    if st.button("새 대화", icon=":material/edit_square:", key="clear-global-chat", use_container_width=True):
        repository.create_chat_session()
        st.rerun()
    selected = repository.active_chat_session()
    event = render_library_tree([], [], selected, key="chat-session-tree", theme=_ui_preferences(repository)[0],
                                sessions=repository.list_chat_sessions())
    _handle_chat_session_event(event, repository)


def _handle_chat_session_event(event, repository):
    if not isinstance(event, dict) or not event.get("nonce") or event["nonce"] == st.session_state.get("chat-session-event"):
        return
    st.session_state["chat-session-event"] = event["nonce"]
    try:
        session_id = str(event["session_id"])
        if event["action"] == "rename_session":
            repository.rename_chat_session(session_id, str(event["name"]))
        elif event["action"] == "delete_session":
            repository.delete_chat_session(session_id)
        elif event["action"] == "select_session":
            if not any(row["id"] == session_id for row in repository.list_chat_sessions()):
                return
            repository.set_metadata("active_chat_session", session_id)
        else:
            return
    except (KeyError, ValueError) as exc:
        st.session_state["flash_error"] = str(exc)
    st.rerun()


@st.fragment(run_every=1)
def _render_global_chat(repository: LectureRepository, model_status: ModelStatus) -> None:
    session_id = repository.active_chat_session()
    messages = repository.list_chat_messages(None, limit=None, session_id=session_id)
    job_key = (str(repository.database_path), "chat", session_id)
    job = summary_jobs.snapshot(job_key)
    running = bool(job and job.running)
    with st.container(key="lass-chat-heading"):
        last_user_id = next((message.id for message in reversed(messages) if message.role == "user"), "")
        st.markdown(f'<span id="lass-chat-draft" data-session="{session_id}" data-revision="{last_user_id}" hidden></span>', unsafe_allow_html=True)
        st.markdown("**LASS** · 전체 노트와 대화")
        st.caption("로컬 AI · 한국어 답변 · 대화 자동 저장")
    with st.container(key="lass-chat-thread"):
        if not messages:
            st.markdown('<div class="lass-chat-welcome"><span>✧</span><h1>무엇을 도와드릴까요?</h1>'
                        '<p>궁금한 내용을 편하게 질문하세요.<br>필요한 근거는 전체 노트에서 찾아드립니다.</p></div>', unsafe_allow_html=True)
        for message in messages:
            with st.chat_message(message.role, avatar="👤" if message.role == "user" else ":material/auto_awesome:"):
                st.markdown(message.content)
        if running:
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                st.markdown((job.text + " ▌") if job.text else "답변을 생성중입니다")
        elif job and job.error:
            st.error(job.error)
    pending = bool(messages and messages[-1].role == "user")
    retry = pending and not running and st.button("답변 다시 생성", key=f"chat-retry-{session_id}", disabled=not model_status.ready)
    if not model_status.ready:
        st.caption("모델 준비가 끝나면 대화할 수 있습니다.")
    question = st.chat_input("무엇이든 질문하세요 · 전체 노트 자동 검색",
                             key=f"global-chat-input-{session_id}", disabled=not model_status.ready or running or pending)
    if question or retry:
        if retry:
            question = messages[-1].content
            history = messages[:-1]
        else:
            history = messages
            repository.add_chat_message(None, "user", question, session_id=session_id)
        pipeline = get_pipeline()
        conversation = [{"role": message.role, "content": message.content} for message in history[-100:]]
        def work(progress, token):
            answer = pipeline.answer_question(
                repository.iter_notes, question, conversation,
                on_token=token,
                fingerprint=repository.notes_revision(),
            )
            repository.add_chat_message(None, "assistant", answer, session_id=session_id)
        summary_jobs.start(job_key, work)
        st.rerun()


def _ui_preferences(repository: LectureRepository) -> tuple[str, int]:
    mode = repository.get_metadata("theme_mode")
    return ("dark" if mode == "dark" else "light"), 100


def _toggle_library_panel() -> None:
    st.session_state["library_open"] = not st.session_state.get("library_open", True)


def _toggle_right_panel() -> None:
    st.session_state["right_panel_open"] = not st.session_state.get(
        "right_panel_open", True
    )


def _enter_workspace(mode: str) -> None:
    if st.session_state.get("workspace-mode", "notes") == "notes":
        st.session_state["notes-panel-snapshot"] = {
            key: st.session_state.get(key, True) for key in ("library_open", "right_panel_open")
        }
    st.session_state["workspace-mode"] = mode
    st.session_state["library_open"] = False


def _return_to_notes() -> None:
    st.session_state.update(st.session_state.pop("notes-panel-snapshot", {}))
    st.session_state["workspace-mode"] = "notes"
    st.session_state.pop("selected_folder_id", None)


def _toggle_center_layout() -> None:
    current = st.session_state.get("center-layout", "stack")
    st.session_state["center-layout"] = "side" if current == "stack" else "stack"


def _open_summary_panel() -> None:
    st.session_state["right_panel_open"] = True
    st.session_state["right-panel-mode"] = "요약"


def _toggle_library_folder(folder_id: str) -> None:
    key = f"library-folder-open-{folder_id}"
    st.session_state[key] = not st.session_state.get(key, True)


def _render_sidebar_header(repository: LectureRepository) -> None:
    """Render the library heading; the desktop title bar carries the brand."""
    with st.container(key="lass-sidebar-header"):
        brand, close = st.columns([1, 0.18], gap="small")
        with brand:
            st.markdown(
                '<div class="lass-global-brand"><strong>내 노트</strong></div>',
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


def _change_stt_language(repository):
    language = st.session_state['moonshine-language']
    repository.set_metadata('stt_language', language)
    os.environ['LASS_STT_LANGUAGE'] = language
    _release_live_transcriber()


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
            if st.button(
                "Chatting",
                icon=":material/chat:",
                use_container_width=True,
                key="chatting-button",
            ):
                _enter_workspace("chat")
                st.rerun()
        with quiz_column:
            if st.button(
                "Quiz",
                icon=":material/quiz:",
                use_container_width=True,
                key="quiz-button",
            ):
                _enter_workspace("quiz")
                st.rerun()

    language = repository.get_metadata('stt_language') or settings.stt_language
    os.environ['LASS_STT_LANGUAGE'] = language
    live_key = st.session_state.get('_live-job-key')
    live_job = summary_jobs.snapshot(live_key) if live_key else None
    st.selectbox('음성 인식 언어', ['ko', 'en'], index=0 if language == 'ko' else 1,
                 format_func=lambda code: '한국어' if code == 'ko' else '영어',
                 key='moonshine-language', disabled=bool(live_job and live_job.running),
                 on_change=_change_stt_language, args=(repository,))
    st.caption('Powered by Moonshine AI · 한국어 모델: Community License')
    render_pane_title("폴더 및 노트", "목록")
    from src.ui.app_update import render_app_update
    render_app_update(settings)
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
        # parameters. The fallback keeps earlier desktop launchers usable;
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
    prompt_key: str = "lecture_detail",
) -> None:
    allowed = DOCUMENT_EXTENSIONS if kind == "document" else SUPPORTED_MEDIA_EXTENSIONS
    label = "자료" if kind == "document" else "녹음 파일"
    source = None
    try:
        path = path.expanduser().resolve()
        if path.suffix.lower() not in allowed:
            raise ValueError(f"지원하는 {label} 형식이 아닙니다.")
        source = repository.add_source(note.id, path, display_name or path.name)
        if copied_upload:
            st.session_state[f"browser-upload-open-{kind}-{note.id}"] = False
            version_key = f"browser-upload-version-{kind}-{note.id}"
            st.session_state[version_key] = int(st.session_state.get(version_key, 0)) + 1
        st.session_state[f"active-{kind}-source-{note.id}"] = source["id"]
        if kind == "audio":
            _release_live_transcriber()
            pipeline = get_pipeline()
            job_key = ("source", str(repository.database_path), note.id, source["id"])
            def work(progress, token):
                process_source(repository, source, settings, lambda: pipeline, prompt_key, None,
                    False, progress, on_segment=lambda segment: token(segment.text + "\n\n"))
            summary_jobs.start(job_key, work)
            st.rerun()
            return
        progress_area = st.session_state.get("_audio-progress-area") if kind == "audio" else None
        status = progress_area.empty() if progress_area is not None else st.empty()
        status.info("텍스트 추출 중…")
        partial_transcript = progress_area.empty() if progress_area is not None else st.empty()
        partial_segments = []
        def show_segment(segment):
            partial_segments.append(segment)
            with partial_transcript.container():
                render_timeline(partial_segments[-120:])
        process_source(
            repository,
            source,
            settings,
            get_pipeline,
            prompt_key,
            None,
            False,
            lambda message: status.info(message),
            on_segment=show_segment,
        )
    except Exception as exc:
        if source is None and copied_upload:
            path.unlink(missing_ok=True)
        st.session_state["flash_error"] = f"{label} 처리를 완료하지 못했습니다: {exc}"
        st.rerun()
    else:
        status.success(f"{label}을 저장했습니다.")
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
    upload_area = st.empty()
    with upload_area.container():
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


@st.fragment
def _render_center_panel(
    repository: LectureRepository,
    note: Optional[LectureRecord],
    settings: Settings,
) -> None:
    sources = _document_sources(repository, note) if note else []
    source = _active_document_source(note, sources) if note else None
    document_note = _document_source_note(note, source) if note and source else None
    preview = None
    if document_note and document_note.media_type.lower() in DOCUMENT_EXTENSIONS:
        try:
            preview = _document_preview(repository, document_note)
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
            breadcrumb_column,
            pages_column,
            zoom_column,
            upload_column,
            layout_column,
        ) = st.columns(
            [1.4, 0.7, 0.8, 0.7, 0.7], gap="small"
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
                st.button(
                    "←",
                    key=f"pdf-previous-{pdf_key}",
                    disabled=page_number <= 1,
                    on_click=st.session_state.update,
                    args=({page_key: page_number - 1},),
                    use_container_width=True,
                )
            with counter:
                st.markdown(
                    f'<span class="lass-pdf-counter">{page_number} / {page_count}</span>',
                    unsafe_allow_html=True,
                )
            with following:
                st.button(
                    "→",
                    key=f"pdf-next-{pdf_key}",
                    disabled=not page_count or page_number >= page_count,
                    on_click=st.session_state.update,
                    args=({page_key: page_number + 1},),
                    use_container_width=True,
                )
        with zoom_column:
            zoom_out, zoom_value, zoom_in = st.columns([1, 1.5, 1], gap="small")
            with zoom_out:
                st.button(
                    "−",
                    key=f"pdf-zoom-out-{pdf_key}",
                    disabled=preview is None or zoom <= 50,
                    on_click=st.session_state.update,
                    args=({zoom_key: zoom - 25, fit_key: False},),
                    use_container_width=True,
                )
            with zoom_value:
                st.button(
                    f"{zoom}%", key=f"pdf-fit-toggle-{pdf_key}",
                    disabled=preview is None, type="primary" if fit else "secondary",
                    on_click=st.session_state.update,
                    args=({fit_key: not fit},), use_container_width=True,
                )
            with zoom_in:
                st.button(
                    "＋",
                    key=f"pdf-zoom-in-{pdf_key}",
                    disabled=preview is None or zoom >= 200,
                    on_click=st.session_state.update,
                    args=({zoom_key: zoom + 25, fit_key: False},),
                    use_container_width=True,
                )
        with layout_column:
            if st.button(
                "좌우 배치" if center_layout == "stack" else "상하 배치",
                icon=(
                    ":material/view_column:"
                    if center_layout == "stack"
                    else ":material/view_agenda:"
                ),
                key="toggle-center-layout",
                on_click=_toggle_center_layout,
                use_container_width=True,
            ):
                st.rerun()
        with upload_column:
            if note is not None:
                _render_document_upload_action(repository, note, settings)

    with st.container(key="lass-document-memo-layout"):
        with st.container(key="lass-document-pane"):
            if note is None:
                render_empty_workspace()
            elif not sources:
                with st.container(key="lass-document-empty-viewer"):
                    st.markdown(
                        '<div class="lass-document-empty-copy"><span>▱</span>'
                        '<b>문서가 아직 없습니다.</b>'
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
                if source.get("status") != "ready" and source.get("id") != "legacy":
                    st.caption("추출할 문서가 있습니다. 다시 시작할 수 있습니다.")
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
                                "lecture_detail",
                                None,
                                False,
                                extraction_status.info,
                            )
                        except Exception as exc:
                            st.session_state["flash_error"] = f"텍스트 추출에 실패했습니다: {exc}"
                            st.rerun()
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
                    theme_mode=_ui_preferences(repository)[0],
                    text_scale=100,
                )

def _get_live_transcriber():
    if "_live_transcriber" not in st.session_state:
        settings = get_settings()
        st.session_state["_live_transcriber"] = create_transcriber(
            model_dir=settings.stt_model_dir, target_lufs=settings.stt_target_lufs,
        )
    return st.session_state["_live_transcriber"]


def _release_live_transcriber() -> None:
    live_key = st.session_state.get("_live-job-key")
    job = summary_jobs.snapshot(live_key) if live_key else None
    if job and job.running:
        return
    transcriber = st.session_state.pop("_live_transcriber", None)
    st.session_state.pop("_live_transcriber_signature", None)
    if transcriber is not None:
        transcriber.unload()


def _handle_live_recorder_event(
    event: Optional[dict[str, Any]],
    repository: LectureRepository,
    note: LectureRecord,
) -> None:
    if not isinstance(event, dict):
        return
    nonce = str(event.get("nonce") or "")
    processed_key = f"live-processed-{note.id}"
    if not nonce or nonce == st.session_state.get(processed_key):
        return
    action = event.get("action")
    if action != "audio_chunk":
        st.session_state[processed_key] = nonce
        st.session_state["live-recorder-ack"] = nonce
    if action == "state":
        st.session_state["recording-note-id"] = note.id if event.get("active") else None
        st.rerun()
        return
    if action == "error":
        st.session_state[f"live-error-{note.id}"] = str(event.get("message") or "녹음을 시작하지 못했습니다.")
        st.rerun()
        return
    if action != "audio_chunk":
        return

    final = bool(event.get("final"))
    job_key = (str(repository.database_path), "live", note.id, nonce)
    job = summary_jobs.snapshot(job_key)
    if job and job.running:
        return
    if job is None:
        encoded = str(event.get("data") or "")
        try:
            if not encoded or len(encoded) > 14_000_000:
                raise ValueError("invalid audio size")
            base64.b64decode(encoded, validate=True)
        except ValueError:
            st.session_state[f"live-error-{note.id}"] = "녹음 구간의 크기가 올바르지 않습니다."
            st.session_state[processed_key] = nonce
            st.session_state["live-recorder-ack"] = nonce
            st.rerun()
            return
        settings = get_settings()
        transcriber = _get_live_transcriber()
        start_session = not st.session_state.get(f"live-session-started-{note.id}", False)
        def work(progress, token):
            media_dir = settings.data_dir / "media"
            media_dir.mkdir(parents=True, exist_ok=True)
            suffix = str(event.get("extension") or ".webm")
            if suffix not in {".webm", ".m4a", ".ogg", ".wav"}:
                suffix = ".webm"
            with tempfile.NamedTemporaryFile(prefix="lass-live-", suffix=suffix, dir=media_dir, delete=False) as output:
                output.write(base64.b64decode(encoded, validate=True))
                path = Path(output.name)
            try:
                result = transcriber.transcribe(path, on_segment=lambda segment: token(segment.text + "\n\n"))
                repository.append_live_transcription(
                    note.id, transcript=result.text, segments=result.segments,
                    duration=result.duration, language=result.language,
                    start_session=start_session, event_nonce=nonce,
                )
            except Exception as exc:
                source = repository.add_source(note.id, path, "복구할 녹음" + suffix)
                repository.fail_source(source["id"], str(exc))
                raise
            else:
                path.unlink(missing_ok=True)
        summary_jobs.start(job_key, work)
        st.session_state["_live-job-key"] = job_key
        return
    if job.error:
        st.session_state[f"live-error-{note.id}"] = job.error + " · 원본 녹음을 자료 목록에 보관했습니다."
    else:
        st.session_state[f"live-session-started-{note.id}"] = not final
        st.session_state.pop(f"live-error-{note.id}", None)
    # Acknowledge only after the worker has committed text or retained failed audio.
    st.session_state[processed_key] = nonce
    st.session_state["live-recorder-ack"] = nonce
    st.session_state[f"live-ack-{note.id}"] = nonce
    if final:
        st.session_state[f"live-session-started-{note.id}"] = False
        _release_live_transcriber()
    st.rerun()


@st.fragment(run_every=1)
def _render_persistent_recorder(repository, theme_mode):
    with st.container(key="lass-recorder-host"):
        event = render_live_recorder(
            ack_nonce=st.session_state.get("live-recorder-ack", ""), theme=theme_mode,
            note_id=st.session_state.get("selected_note_id", ""), key="persistent-live-recorder",
        )
        if isinstance(event, dict):
            target = repository.get(str(event.get("note_id") or ""))
            if target is not None:
                _handle_live_recorder_event(event, repository, target)
            elif event.get("nonce") != st.session_state.get("live-recorder-ack"):
                st.session_state["live-recorder-ack"] = str(event.get("nonce") or "")
                st.rerun()


@st.fragment(run_every=.5)
def _render_recording_panel(
    repository: LectureRepository,
    note: LectureRecord,
    settings: Settings,
    theme_mode: str,
) -> None:
    with st.container(key="lass-transcript-stream"):
        # Actions below write streaming output back into this upper pane.
        st.session_state["_audio-progress-area"] = st.container()
        error = st.session_state.get(f"live-error-{note.id}")
        job_key = st.session_state.get("_live-job-key")
        live_job = summary_jobs.snapshot(job_key) if job_key and job_key[2] == note.id else None
        if live_job and live_job.running:
            st.info("실시간 전사 중…")
            if live_job.text:
                st.markdown(live_job.text + " ▌")
        if error:
            render_warning("녹음 확인 필요", error)
        for source in repository.list_sources(note.id):
            if source["status"] == "failed" and source["name"].startswith("복구할 녹음"):
                if st.button("보관된 녹음 다시 전사", key=f"recover-live-{source['id']}"):
                    try:
                        process_source(repository, source, settings, get_pipeline, "lecture_detail",
                                       None, False, lambda message: None)
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.pop(f"live-error-{note.id}", None)
                        st.rerun()
        for source in repository.list_sources(note.id):
            source_job = summary_jobs.snapshot(("source", str(repository.database_path), note.id, source["id"]))
            if source_job:
                seen_key = f"source-job-seen-{source_job.id}"
                was_running = st.session_state.get(seen_key)
                st.session_state[seen_key] = source_job.running
                if source_job.running:
                    st.caption(source_job.progress)
                    if source_job.text:
                        st.markdown(source_job.text)
                elif was_running is True:
                    if source_job.error:
                        st.session_state["flash_error"] = source_job.error
                    st.rerun()
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
        st.markdown('<div class="lass-recorder-slot" style="height:118px"></div>', unsafe_allow_html=True)


def _render_right_panel(
    repository: LectureRepository,
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
        _render_recording_panel(repository, note, settings, theme_mode)
        return

    with st.container(key="lass-summary-stream"):
        _render_live_summary(repository, note, settings, theme_mode, text_scale, f"right-summary-{note.id}")
        prompt_key = "lecture_detail"
        job = summary_jobs.snapshot(_summary_job_key(repository, note.id))
        running = bool(job and job.running)
    with st.container(key="lass-summary-actions"):
        include_memo = _memo_option(note.id, f"right-summary-memo-{note.id}", running)
        try:
            device_info = json.loads((settings.data_dir / "runtime-device.json").read_text(encoding="utf-8"))
            st.caption(f"실행 환경: CUDA · {device_info['gpu']}" if device_info["cuda_available"] else
                       "실행 환경: CPU · CUDA GPU 또는 호환 드라이버를 확인해 주세요.")
        except (OSError, ValueError, KeyError):
            pass
        if not model_status.ready:
            st.caption(
                "EXAONE 모델을 준비하지 못했습니다. 인터넷 연결을 확인하고 LASS를 다시 실행해 주세요."
            )
        elif note.transcript_chars <= 0:
            st.caption("요약할 녹취 또는 문서 텍스트가 필요합니다.")
        summarize_column, export_column = st.columns([1, 0.9], gap="small")
        with summarize_column:
            if st.button(
                "요약 중…" if running else ("다시 요약하기" if note.summary.strip() else "요약하기"),
                type="primary",
                use_container_width=True,
                disabled=running or not model_status.ready or note.transcript_chars <= 0,
                key=f"right-resummary-{note.id}",
            ):
                _release_live_transcriber()
                _resummarize_note(
                    repository,
                    note,
                    settings,
                    prompt_key,
                    include_memo,
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
    # Desktop first-run setup is exclusive; browser users can still open saved notes.
    from src.ui.model_preparation import models_ready, render_model_preparation
    if os.environ.get("LASS_DESKTOP_PICKER") and not models_ready(settings):
        inject_theme()
        st.html("""<style>
            .st-key-lass-model-setup { flex:1; min-height:0; overflow:auto;
                width:min(100%, 960px); margin:auto; padding:clamp(24px, 6vw, 72px); }
            .st-key-lass-model-setup [data-testid="stHorizontalBlock"] > :last-child { min-width:100px; }
        </style>""")
        with st.container(key="lass-model-setup"):
            render_model_preparation(settings, required=True)
        return
    repository = get_repository()
    prompt_manager = PromptManager(settings.prompt_dir)
    migration = migrate_legacy_storage(repository, settings.data_dir)
    model_status = LocalModelManager(
        settings.exaone_model_id, settings.exaone_model_dir
    ).status()
    theme_mode, text_scale = _ui_preferences(repository)
    notes = repository.list(lightweight=True)
    mode = st.session_state.get("workspace-mode", "notes")
    selected = _selected_note(repository, notes) if mode == "notes" else None

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

    show_library = st.session_state.get("chat-panel-open", False) if mode == "chat" else st.session_state.get("library_open", True)
    show_right = mode == "notes" and st.session_state.get("right_panel_open", True)
    with st.container(key="lass-panel-openers"):
        left_opener, right_opener = st.columns(2)
        with left_opener:
            if mode == "notes":
                st.button("노트 목록", icon=":material/left_panel_open:", key="toggle-library-panel", on_click=_toggle_library_panel)
            elif mode == "chat":
                if not show_library:
                    st.button("대화 목록 열기", icon=":material/left_panel_open:", key="toggle-chat-panel",
                              on_click=lambda: st.session_state.__setitem__("chat-panel-open", True))
            else:
                st.button("노트로 돌아가기", icon=":material/arrow_back:", key=f"{mode}-close", on_click=_return_to_notes)
        with right_opener:
            if mode == "chat":
                st.button("대화 나가기", icon=":material/logout:", key="chat-close", on_click=_return_to_notes)
            else:
                st.button("AI 패널", icon=":material/right_panel_open:", key="toggle-right-panel", on_click=_toggle_right_panel)
    with st.container(key="lass-layout"):
        library_column, workspace_column, right_column = st.columns(3, gap=None)
        with library_column:
            with st.container(key="lass-library-pane"):
                if mode == "chat":
                    _render_chat_sessions(repository)
                else:
                    _render_sidebar_header(repository)
                with st.container(key="lass-library-body"):
                    if mode != "chat":
                        _render_library_buttons(
                            repository,
                            settings,
                            notes,
                            selected.id if selected else None,
                            theme_mode,
                            text_scale,
                        )
        with workspace_column:
            with st.container(key="lass-workspace-pane"):
                if mode == "quiz":
                    from src.ui.quiz_view import render_quiz_view
                    render_quiz_view(repository, model_status, get_pipeline)
                elif mode == "chat":
                    with st.container(key="lass-chat-view"):
                        _render_global_chat(repository, model_status)
                else:
                    _render_center_panel(repository, selected, settings)
                if not model_status.ready:
                    with st.container(key="lass-model-preparation"):
                        render_model_preparation(settings, model_status.ready)
        with right_column:
            with st.container(key="lass-ai-pane"):
                if mode == "notes":
                    _render_right_panel(
                        repository,
                        selected,
                        settings,
                        model_status,
                        prompt_manager,
                        theme_mode,
                        text_scale,
                    )
    # Dialogs own their own fragment. Keep them outside the center-panel fragment
    # so upload, cancellation and full-app reruns cannot discard the workspace.
    if selected is not None and st.session_state.get(
        f"browser-upload-open-document-{selected.id}"
    ):
        st.session_state["document-upload-dialog-note-id"] = selected.id
        _render_document_upload_dialog(repository, selected, settings)
    # One persistent iframe owns the microphone, independently of panels/views.
    _render_persistent_recorder(repository, theme_mode)
    render_layout_resizer(
        st.session_state.get("center-layout", "stack"),
        library_open=show_library,
        right_open=show_right,
        transcript_version=selected.segment_count if selected else 0,
        transcript_active=(
            show_right
            and st.session_state.get("right-panel-mode", "녹음") == "녹음"
        ),
        key="lass-layout-resizer",
    )
    st.html('<span id="lass-screen-ready" hidden></span>')


if __name__ == "__main__":
    main()
