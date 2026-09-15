"""End-to-end orchestration for the local summarization pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, Union

from src.config import Settings
from src.llm import LocalEXAONEClient, PromptManager, Summarizer
from src.pdf_utils import extract_pdf_text
from src.rag import LocalEmbedder, SQLiteTextStore, TextChunk, iter_text_chunks
from src.stt import Transcription, create_transcriber
from src.runtime import memory_managed, release_memory


ProgressCallback = Callable[[str], None]
NoteSource = Union[Sequence[Any], Callable[[], Iterable[Any]]]


@dataclass(frozen=True)
class ProcessingResult:
    summary: str
    transcription: Transcription
    prompt_name: str
    vector_collection: Optional[str] = None


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    prompt_name: str
    vector_collection: Optional[str] = None


@dataclass(frozen=True)
class DocumentProcessingResult:
    summary: str
    transcript: str
    prompt_name: str
    page_count: int
    vector_collection: Optional[str] = None


@dataclass(frozen=True)
class ContextWindow:
    """A repeatable, bounded view over context pages selected for one model job."""

    page_factory: Callable[[], Iterator[str]]
    page_count: int
    char_count: int

    def __iter__(self) -> Iterator[str]:
        return self.page_factory()

    def as_text(self) -> str:
        return "\n\n".join(self.page_factory())


class LASSPipeline:
    """Connect STT, retrieval, prompting, and local LLM generation."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        stt_model: Optional[str] = None,
        stt_backend: str = "faster-whisper",
        stt_device: Optional[str] = None,
        stt_compute_type: Optional[str] = None,
        llm_model_id: Optional[str] = None,
        llm_model_dir: Optional[Path] = None,
        llm_device: Optional[str] = None,
        llm_backend: str = "transformers",
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.prompt_manager = PromptManager(self.settings.prompt_dir)
        self.stt_model = stt_model or self.settings.whisper_model
        self.stt_backend = stt_backend
        self.stt_device = stt_device or self.settings.whisper_device
        self.stt_compute_type = stt_compute_type or self.settings.whisper_compute_type
        self.llm_client = LocalEXAONEClient(
            model_dir=llm_model_dir or self.settings.exaone_model_dir,
            model_id=llm_model_id or self.settings.exaone_model_id,
            device=llm_device or self.settings.exaone_device,
            backend=llm_backend,
            max_new_tokens=self.settings.exaone_max_new_tokens,
            temperature=self.settings.exaone_temperature,
            max_input_tokens=self.settings.exaone_max_input_tokens,
        )
        self._embedder: Optional[LocalEmbedder] = None

    def _release_embedder(self) -> None:
        if self._embedder is not None:
            self._embedder.unload()
            self._embedder = None
        release_memory()

    def unload(self) -> None:
        self.llm_client.unload()
        self._release_embedder()

    @memory_managed
    def transcribe(self, media_path: Path, language: Optional[str] = None) -> Transcription:
        transcriber = create_transcriber(
            backend=self.stt_backend, model_name=self.stt_model,
            device=self.stt_device, compute_type=self.stt_compute_type,
            target_lufs=self.settings.stt_target_lufs,
        )
        try:
            return transcriber.transcribe(media_path, language=language)
        finally:
            transcriber.unload()

    @staticmethod
    def _notify(callback: Optional[ProgressCallback], message: str) -> None:
        if callback is not None:
            callback(message)

    @staticmethod
    def _collection_name(text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
        return f"lass-{digest}"

    @staticmethod
    def folder_collection_name(folder_id: str) -> str:
        return f"folder-{folder_id}"

    def _get_embedder(self) -> LocalEmbedder:
        if self._embedder is None:
            self._embedder = LocalEmbedder(self.settings.embedding_model)
        return self._embedder

    def _text_chunks(self, text: str) -> Iterator[TextChunk]:
        return iter_text_chunks(
            text,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )

    @staticmethod
    def _context_pages(context: ContextWindow | str) -> Iterator[str]:
        if isinstance(context, ContextWindow):
            return iter(context)
        return iter((context,))

    def _index_chunks(
        self,
        store: SQLiteTextStore,
        chunks: Iterable[TextChunk],
        *,
        fingerprint: str,
    ) -> None:
        """Persist a streaming FTS index without loading an embedding model."""
        store.upsert_stream(
            chunks,
            fingerprint=fingerprint,
            batch_size=32,
        )

    def _all_context(self, text: str) -> ContextWindow:
        """Return every source chunk as a repeatable, one-page-at-a-time view."""
        chunk_count = 0
        char_count = 0
        for chunk in self._text_chunks(text):
            chunk_count += 1
            char_count += len(chunk.text)
        if not chunk_count:
            raise ValueError("No text was produced from the uploaded media")
        return ContextWindow(
            lambda: (chunk.text for chunk in self._text_chunks(text)),
            chunk_count,
            char_count,
        )

    def _select_context_and_collection(
        self,
        transcript: str,
        query: str,
        top_k: int,
        collection_key: Optional[str] = None,
    ) -> tuple[ContextWindow, Optional[str]]:
        chunk_count = 0
        selected_chars = 0
        for chunk in self._text_chunks(transcript):
            chunk_count += 1
            selected_chars += len(chunk.text)
        if not chunk_count:
            raise ValueError("No text was produced from the uploaded media")
        if chunk_count <= top_k:
            return ContextWindow(
                lambda: (chunk.text for chunk in self._text_chunks(transcript)),
                chunk_count,
                selected_chars,
            ), None

        fingerprint = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
        collection_name = self._collection_name(collection_key or transcript)
        store = SQLiteTextStore(
            self.settings.vector_database_path,
            collection_name=collection_name,
        )
        if store.get_fingerprint() != fingerprint or store.count() != chunk_count:
            self._index_chunks(
                store,
                self._text_chunks(transcript),
                fingerprint=fingerprint,
            )
        results = store.search(query, limit=min(top_k, chunk_count))
        results.sort(key=lambda item: int(item.metadata.get("index", 0)))
        selected_pages = tuple(result.text for result in results)
        return ContextWindow(
            lambda: iter(selected_pages),
            len(selected_pages),
            sum(len(page) for page in selected_pages),
        ), collection_name

    @staticmethod
    def _folder_source_pages(notes: NoteSource) -> Iterator[str]:
        note_iterator = notes() if callable(notes) else iter(notes)
        for note in note_iterator:
            title = str(getattr(note, "title", "제목 없음"))
            if str(getattr(note, "summary", "")).strip():
                yield f"# 노트: {title}\n[AI 요약]\n{note.summary.strip()}"
            if str(getattr(note, "notes", "")).strip():
                yield f"# 노트: {title}\n[사용자 메모]\n{note.notes.strip()}"
            if str(getattr(note, "transcript", "")).strip():
                yield f"# 노트: {title}\n[전체 녹취]\n{note.transcript.strip()}"

    def _folder_chunks(self, notes: NoteSource) -> Iterator[TextChunk]:
        index = 0
        offset = 0
        for source_page in self._folder_source_pages(notes):
            for chunk in self._text_chunks(source_page):
                yield TextChunk(
                    index=index,
                    text=chunk.text,
                    start=offset + chunk.start,
                    end=offset + chunk.end,
                )
                index += 1
            offset += len(source_page) + 2

    def _select_folder_context(
        self,
        folder_id: str,
        notes: NoteSource,
        question: str,
        top_k: int,
    ) -> ContextWindow:
        chunk_count = 0
        selected_chars = 0
        for chunk in self._folder_chunks(notes):
            chunk_count += 1
            selected_chars += len(chunk.text)
        if not chunk_count:
            raise ValueError("이 폴더에는 대화에 참고할 내용이 없습니다.")
        if chunk_count <= top_k:
            return ContextWindow(
                lambda: (chunk.text for chunk in self._folder_chunks(notes)),
                chunk_count,
                selected_chars,
            )

        fingerprint_hash = hashlib.sha256()
        for chunk in self._folder_chunks(notes):
            fingerprint_hash.update(chunk.text.encode("utf-8"))
            fingerprint_hash.update(b"\0")
        fingerprint = fingerprint_hash.hexdigest()
        store = SQLiteTextStore(
            self.settings.vector_database_path,
            collection_name=self.folder_collection_name(folder_id),
        )
        if store.get_fingerprint() != fingerprint or store.count() != chunk_count:
            self._index_chunks(
                store,
                self._folder_chunks(notes),
                fingerprint=fingerprint,
            )
        results = store.search(question, limit=min(top_k, chunk_count))
        results.sort(key=lambda item: int(item.metadata.get("index", 0)))
        selected_pages = tuple(result.text for result in results)
        return ContextWindow(
            lambda: iter(selected_pages),
            len(selected_pages),
            sum(len(page) for page in selected_pages),
        )

    def _select_context(self, transcript: str, query: str, top_k: int) -> str:
        """Compatibility helper used by callers that only need selected text."""
        context, _ = self._select_context_and_collection(transcript, query, top_k)
        return context.as_text()

    @memory_managed
    def process(
        self,
        media_path: Path,
        prompt_key: str = "default_summary",
        language: Optional[str] = None,
        top_k: int = 8,
        collection_key: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> ProcessingResult:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.settings.ensure_directories()
        template = self.prompt_manager.load(prompt_key)

        self._notify(
            progress,
            f"음량을 {self.settings.stt_target_lufs:g} LUFS로 정규화한 뒤 음성을 인식하고 있습니다.",
        )
        transcription = self.transcribe(media_path, language=language)

        self._notify(progress, "요약에 필요한 문맥을 구성하고 있습니다.")
        context = self._all_context(transcription.text)
        vector_collection = None

        self._notify(progress, "로컬 언어 모델이 요약을 생성하고 있습니다.")
        self._release_embedder()
        summary = Summarizer(self.llm_client).summarize_pages(
            self._context_pages(context),
            template,
            progress=progress,
        )

        return ProcessingResult(
            summary=summary,
            transcription=transcription,
            prompt_name=template.name,
            vector_collection=vector_collection,
        )

    @memory_managed
    def resummarize(
        self,
        transcript: str,
        *,
        prompt_key: str = "default_summary",
        notes: str = "",
        top_k: int = 8,
        collection_key: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> SummaryResult:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        template = self.prompt_manager.load(prompt_key)
        self._notify(progress, "수정된 녹취에서 핵심 문맥을 찾고 있습니다.")
        context = self._all_context(transcript)
        vector_collection = None
        context_pages: Iterable[str] = self._context_pages(context)
        if notes.strip():
            memo_page = f"## 사용자가 작성한 메모\n{notes.strip()}"
            if isinstance(context, ContextWindow) and context.page_count == 1:
                combined = f"{context.as_text()}\n\n{memo_page}"
                context_pages = (
                    iter((combined,))
                    if len(combined) <= self.settings.chunk_size
                    else (chunk.text for chunk in self._text_chunks(combined))
                )
            elif isinstance(context, str) and len(context) + len(memo_page) <= self.settings.chunk_size:
                context_pages = iter((f"{context}\n\n{memo_page}",))
            else:
                context_pages = chain(context_pages, (memo_page,))
        self._notify(progress, "로컬 언어 모델이 노트를 다시 요약하고 있습니다.")
        self._release_embedder()
        summary = Summarizer(self.llm_client).summarize_pages(
            context_pages,
            template,
            progress=progress,
        )
        return SummaryResult(
            summary=summary,
            prompt_name=template.name,
            vector_collection=vector_collection,
        )

    @memory_managed
    def resummarize_pages(
        self,
        pages: Iterable[str],
        *,
        prompt_key: str = "default_summary",
        notes: str = "",
        progress: Optional[ProgressCallback] = None,
    ) -> SummaryResult:
        """Summarize a disk-backed transcript iterator one bounded page at a time."""
        template = self.prompt_manager.load(prompt_key)
        self._notify(progress, "전체 녹취를 디스크에서 작은 조각으로 읽고 있습니다.")
        context_pages: Iterable[str] = (page for page in pages if page and page.strip())
        if notes.strip():
            context_pages = chain(
                context_pages,
                (f"## 사용자가 작성한 메모\n{notes.strip()}",),
            )
        self._release_embedder()
        summary = Summarizer(self.llm_client).summarize_pages(
            context_pages,
            template,
            progress=progress,
        )
        return SummaryResult(
            summary=summary,
            prompt_name=template.name,
            vector_collection=None,
        )

    @memory_managed
    def process_pdf(
        self,
        pdf_path: Path,
        *,
        prompt_key: str = "default_summary",
        top_k: int = 8,
        collection_key: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> DocumentProcessingResult:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.settings.ensure_directories()
        template = self.prompt_manager.load(prompt_key)
        self._notify(progress, "PDF 페이지에서 텍스트를 추출하고 있습니다.")
        document = extract_pdf_text(pdf_path)
        self._notify(progress, "요약에 필요한 PDF 문맥을 구성하고 있습니다.")
        context = self._all_context(document.text)
        vector_collection = None
        self._notify(progress, "로컬 언어 모델이 PDF를 요약하고 있습니다.")
        self._release_embedder()
        summary = Summarizer(self.llm_client).summarize_pages(
            self._context_pages(context),
            template,
            progress=progress,
        )
        return DocumentProcessingResult(
            summary=summary,
            transcript=document.text,
            prompt_name=template.name,
            page_count=document.page_count,
            vector_collection=vector_collection,
        )

    @memory_managed
    def answer_folder_question(
        self,
        folder_id: str,
        folder_name: str,
        notes: NoteSource,
        question: str,
        history: Sequence[Mapping[str, str]] = (),
        *,
        top_k: int = 8,
    ) -> str:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("질문을 입력해 주세요.")
        context_window = self._select_folder_context(
            folder_id, notes, clean_question, top_k
        )
        self._release_embedder()
        if (
            context_window.page_count > 1
            and context_window.char_count > self.settings.chunk_size
        ):
            context = Summarizer(self.llm_client).condense_pages(
                context_window,
                objective=f"'{clean_question}' 질문에 답할 근거 선별",
            )
        else:
            context = context_window.as_text()
        system_prompt = (
            f"당신은 '{folder_name}' 폴더 전용 로컬 학습 도우미입니다. "
            "제공된 노트 자료만 근거로 한국어로 답하세요. 자료에 없는 내용은 "
            "없다고 명확히 말하고, 가능하면 근거가 된 노트 제목을 밝혀 주세요."
        )
        conversation = [
            {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
            for item in history[-8:]
            if item.get("role") in {"user", "assistant"}
            and str(item.get("content", "")).strip()
        ]
        conversation.append(
            {
                "role": "user",
                "content": (
                    "다음은 폴더에서 검색한 참고 자료입니다.\n\n"
                    f"{context}\n\n[질문]\n{clean_question}"
                ),
            }
        )
        return self.llm_client.chat_messages(system_prompt, conversation)
