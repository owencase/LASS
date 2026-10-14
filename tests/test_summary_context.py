"""The UI must use the bounded interactive context from Setup 1.3.0."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.config import Settings
from src.llm import PromptManager, Summarizer
from src.storage import LectureRepository
from src.ui.app import _interactive_summary_pages


class SummaryContextTests(unittest.TestCase):
    def test_long_transcript_uses_eight_excerpts_and_one_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = LectureRepository(Path(directory) / "lass.db")
            transcript = "".join(f"PAGE-{i:02d} " + "내용 " * 397 + "끝" for i in range(40))
            note = repo.create_empty("test")
            repo.complete(note.id, transcript=transcript, summary="", segments=[], prompt_name="", language=None, duration=None)
            pages = list(_interactive_summary_pages(repo, repo.get_workspace(note.id), Settings()))
            self.assertEqual(len(pages), 1)
            self.assertEqual(pages[0].count("[전체 자료 위치 "), 8)
            self.assertIn("[전체 자료 위치 1/", pages[0])
            client = SimpleNamespace(chat=Mock(return_value="완료"))
            template = PromptManager(Path(__file__).parents[1] / "prompts").load("lecture_detail")
            self.assertEqual(Summarizer(client).summarize_pages(pages, template), "완료")
            client.chat.assert_called_once()

    def test_document_only_note_also_uses_one_bounded_context(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = LectureRepository(Path(directory) / "lass.db")
            note = repo.create_empty()
            path = Path(directory) / "sample.pdf"
            path.write_bytes(b"%PDF")
            source = repo.add_source(note.id, path)
            repo.complete_document_source(source["id"], iter(SimpleNamespace(number=i, text="PDF " * 1000) for i in range(1, 21)))
            pages = list(_interactive_summary_pages(repo, repo.get_workspace(note.id), Settings()))
            self.assertEqual(len(pages), 1)
            self.assertEqual(pages[0].count("[전체 자료 위치 "), 8)
            self.assertLess(len(pages[0]), 2200)

    def test_short_transcript_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = LectureRepository(Path(directory) / "lass.db")
            note = repo.create_empty("test")
            repo.complete(note.id, transcript="짧은 원문", summary="", segments=[], prompt_name="", language=None, duration=None)
            self.assertEqual(list(_interactive_summary_pages(repo, repo.get_workspace(note.id), Settings())), ["짧은 원문"])
