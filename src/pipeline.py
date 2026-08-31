"""End-to-end orchestration for the local summarization pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.config import Settings
from src.llm import OllamaClient, PromptManager, Summarizer
from src.rag import ChromaVectorStore, LocalEmbedder, split_text
from src.stt import Transcription, WhisperTranscriber


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class ProcessingResult:
    summary: str
    transcription: Transcription
    prompt_name: str
    output_path: Path


class LASSPipeline:
    """Connect STT, retrieval, prompting, and local LLM generation."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings.from_env()
        self.prompt_manager = PromptManager(self.settings.prompt_dir)

    @staticmethod
    def _notify(callback: Optional[ProgressCallback], message: str) -> None:
        if callback is not None:
            callback(message)

    @staticmethod
    def _collection_name(text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
        return f"lass-{digest}"

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        whole_seconds = max(0, int(seconds))
        hours, remainder = divmod(whole_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _select_context(self, transcript: str, query: str, top_k: int) -> str:
        chunks = split_text(
            transcript,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No text was produced from the uploaded media")
        if len(chunks) <= top_k:
            return "\n\n".join(chunk.text for chunk in chunks)

        embedder = LocalEmbedder(self.settings.embedding_model)
        embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
        store = ChromaVectorStore(
            self.settings.vector_db_dir,
            collection_name=self._collection_name(transcript),
        )
        store.upsert(chunks, embeddings)
        results = store.search(embedder.embed_query(query), limit=min(top_k, len(chunks)))
        results.sort(key=lambda item: int(item.metadata.get("index", 0)))
        return "\n\n".join(result.text for result in results)

    def _write_report(
        self,
        media_path: Path,
        summary: str,
        transcription: Transcription,
        prompt_name: str,
    ) -> Path:
        lines = [
            f"# {media_path.stem} 요약",
            "",
            f"- 요약 형식: {prompt_name}",
            f"- 감지 언어: {transcription.language or 'unknown'}",
            "",
            "## 요약",
            "",
            summary,
            "",
            "## 타임스탬프 스크립트",
            "",
        ]
        lines.extend(
            f"- `{self._format_timestamp(segment.start)}` {segment.text}"
            for segment in transcription.segments
        )
        destination = self.settings.output_dir / f"{media_path.stem}_summary.md"
        destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return destination

    def process(
        self,
        media_path: Path,
        prompt_key: str = "default_summary",
        language: Optional[str] = None,
        top_k: int = 8,
        progress: Optional[ProgressCallback] = None,
    ) -> ProcessingResult:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.settings.ensure_directories()
        template = self.prompt_manager.load(prompt_key)

        self._notify(progress, "음성을 텍스트로 변환하고 있습니다.")
        transcriber = WhisperTranscriber(
            model_name=self.settings.whisper_model,
            device=self.settings.whisper_device,
            compute_type=self.settings.whisper_compute_type,
        )
        transcription = transcriber.transcribe(media_path, language=language)

        self._notify(progress, "요약에 필요한 문맥을 구성하고 있습니다.")
        query = f"{template.description} 핵심 내용, 결론, 중요 사항"
        context = self._select_context(transcription.text, query=query, top_k=top_k)

        self._notify(progress, "로컬 언어 모델이 요약을 생성하고 있습니다.")
        client = OllamaClient(
            model=self.settings.ollama_model,
            base_url=self.settings.ollama_base_url,
        )
        summary = Summarizer(client).summarize(context, template)

        self._notify(progress, "결과 파일을 저장하고 있습니다.")
        output_path = self._write_report(
            Path(media_path), summary, transcription, template.name
        )
        return ProcessingResult(
            summary=summary,
            transcription=transcription,
            prompt_name=template.name,
            output_path=output_path,
        )
