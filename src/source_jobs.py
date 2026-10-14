"""Persist extracted sources before summarization so retries preserve work."""
from pathlib import Path
from src.pdf_utils import DOCUMENT_EXTENSIONS, iter_document_pages
from src.rag import SQLiteTextStore, SQLiteVectorStore


def process_source(repository, source, settings, pipeline_factory, prompt_key, language,
                   summarize, progress, on_token=None, on_segment=None):
    """Save transcription before attempting summary, so retries never lose work."""
    note = repository.get_workspace(source["note_id"])
    if source["status"] != "ready":
        try:
            repository.begin_source_processing(source["id"])
            progress("텍스트 추출 중…")
            if source["media_type"] in DOCUMENT_EXTENSIONS:
                previous_collection = note.vector_collection
                note = repository.complete_document_source(source["id"], iter_document_pages(Path(source["path"]), progress=progress))
            else:
                result = pipeline_factory().transcribe(Path(source["path"]), language=language,
                    **({"on_segment": on_segment} if on_segment else {}))
                values = {"transcript": result.text, "segments": result.segments,
                          "duration": result.duration, "language": result.language}
                previous_collection = note.vector_collection
                note = repository.complete_source(source["id"], **values)
            if previous_collection:
                SQLiteVectorStore.delete_collection(settings.vector_database_path, previous_collection)
                SQLiteTextStore.delete_collection(settings.vector_database_path, previous_collection)
            if note.folder_id:
                SQLiteVectorStore.delete_collection(settings.vector_database_path, f"folder-{note.folder_id}")
                SQLiteTextStore.delete_collection(settings.vector_database_path, f"folder-{note.folder_id}")
        except Exception as exc:
            if repository.get_source(source["id"])["status"] != "ready":
                repository.fail_source(source["id"], str(exc))
            raise
    if summarize:
        progress(f"'{note.title}' 요약 중…")
        result = pipeline_factory().resummarize_pages(repository.iter_transcript_pages(note.id), notes=note.notes,
                                               prompt_key=prompt_key,
                                               progress=progress,
                                               **({"on_token": on_token} if on_token else {}))
        repository.replace_summary(note.id, summary=result.summary, prompt_key=prompt_key,
                                   prompt_name=result.prompt_name, vector_collection=result.vector_collection)

