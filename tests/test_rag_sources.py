import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace

from src.config import Settings
from src.pipeline import LASSPipeline
from src.rag import SQLiteTextStore, TextChunk
from src.rag.sources import note_excerpts
from src.storage import LectureRepository


class SourceRetrievalTests(unittest.TestCase):
    def test_quiz_evidence_uses_original_text_instead_of_generated_summary(self):
        note = SimpleNamespace(id='n', title='강의', source_name='lecture.pdf',
            transcript='스택은 후입선출 구조이다.', summary='스택은 선입선출 구조이다.', notes='복습할 메모')
        originals = list(note_excerpts([note], original_only=True))
        self.assertEqual([item.text for item in originals], ['스택은 후입선출 구조이다.'])
        self.assertEqual(len(list(note_excerpts([note]))), 3)
        note.transcript = ''
        self.assertEqual([item.text for item in note_excerpts([note], original_only=True)], ['복습할 메모'])
        note.notes = ''
        self.assertEqual(list(note_excerpts([note], original_only=True)), [])

    def test_korean_similarity_and_no_arbitrary_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteTextStore(Path(directory) / "index.db", "course")
            store.upsert_stream([TextChunk(0, "신경망 학습은 가중치를 갱신한다.", 0, 20,
                {"source_reference": "노트.pdf - 3페이지"}), TextChunk(1, "통계 검정 방법", 20, 30)])
            result = store.search_similar("신경망의 학습이란?", limit=1)[0]
            self.assertGreater(result.score, 0)
            self.assertEqual(result.metadata["source_reference"], "노트.pdf - 3페이지")
            self.assertEqual(store.search_similar("zzzzzzzz"), [])
            self.assertEqual(store.search("zzzzzzzz", fallback=False), [])

    def test_source_boundaries_cached_queries_and_edit_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(data_dir=Path(directory), chunk_size=40, chunk_overlap=5)
            repo = LectureRepository(settings.database_path)
            folder = repo.create_folder("과목")
            note = repo.create_empty("AI", folder.id)
            repo.complete(note.id, prompt_name="", language=None, duration=None, segments=[], summary="", transcript="## a.pdf\n\n## 3페이지\n\n신경망 학습의 가중치 갱신에 관한 내용입니다. " * 4 + "\n## b.pptx\n\n## 2슬라이드\n\n회귀 분석과 통계 검정입니다.")
            excerpts = list(note_excerpts(lambda: repo.iter_by_folder(folder.id)))
            self.assertTrue(any("b.pptx" in item.reference and "2슬라이드" in item.reference for item in excerpts))
            pipeline = LASSPipeline(settings)
            revision = repo.folder_revision(folder.id)
            context = pipeline._select_folder_context(folder.id, lambda: repo.iter_by_folder(folder.id), "가중치", 3, revision)
            self.assertIn("a.pdf - 전체 녹취 / 3페이지", context.as_text())
            self.assertTrue(all("[출처:" in page for page in context))
            unused = Mock(side_effect=AssertionError("cached query must not read note bodies"))
            again = pipeline._select_folder_context(folder.id, unused, "가중치", 3, revision)
            self.assertEqual(again.as_text(), context.as_text())
            repo.complete(note.id, prompt_name="", language=None, duration=None, segments=[], summary="", transcript="## a.pdf\n\n## 9페이지\n\n가중치 갱신의 새로운 원문입니다.")
            self.assertNotEqual(repo.folder_revision(folder.id), revision)
            updated = pipeline._select_folder_context(folder.id, lambda: repo.iter_by_folder(folder.id), "가중치", 3, repo.folder_revision(folder.id))
            self.assertIn("9페이지", updated.as_text())


if __name__ == "__main__":
    unittest.main()
