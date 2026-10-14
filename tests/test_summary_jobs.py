import os
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from src.summary_jobs import SummaryJobs
from src.storage import LectureRepository


class SummaryJobTests(unittest.TestCase):
    def test_rerun_navigation_and_memo_edit_do_not_cancel_or_duplicate_summary(self):
        from streamlit.testing.v1 import AppTest
        import streamlit as st
        st.cache_resource.clear()

        entered, finish = Event(), Event()
        jobs = SummaryJobs()

        def summarize(pages, *, on_token, progress, **kwargs):
            self.assertIn("테스트 강의 내용", "".join(pages))
            on_token("실시간 내용")
            entered.set()
            if not finish.wait(10):
                raise RuntimeError("test timed out")
            return SimpleNamespace(summary="완료된 강의 노트", prompt_name="강의 상세 노트", vector_collection=None)

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "LASS_DATA_DIR": directory, "LASS_EXAONE_MODEL_DIR": str(Path(directory) / "model")
        }):
            repository = LectureRepository(Path(directory) / "lass.db")
            note = repository.create_empty("첫 노트")
            source_path = Path(directory) / "sample.pdf"
            source_path.write_bytes(b"%PDF")
            source = repository.add_source(note.id, source_path)
            repository.complete_source(source["id"], transcript="테스트 강의 내용")
            second = repository.create_empty("다른 노트")
            with patch("src.summary_jobs.summary_jobs", jobs), \
                 patch("src.pipeline.LASSPipeline.resummarize_pages", side_effect=summarize), \
                 patch("src.llm.LocalModelManager.status", return_value=SimpleNamespace(ready=True)), \
                 patch("src.document_preview.preview_page", side_effect=ValueError("test document")):
                app = AppTest.from_file(str(Path(__file__).parents[1] / "src/ui/app.py"), default_timeout=30)
                app.session_state["selected_note_id"] = note.id
                app.session_state["right-panel-mode"] = "요약"
                app.run()
                app.checkbox(key=f"right-summary-memo-{note.id}").uncheck().run()
                app.button(key=f"right-resummary-{note.id}").click().run()
                self.assertTrue(entered.wait(5))
                key = (str(repository.database_path), note.id)
                first_id = jobs.snapshot(key).id
                self.assertEqual(jobs.snapshot(key).text, "실시간 내용")
                self.assertFalse(app.info, "Summary progress belongs only in the summary editor")
                app.button(key="toggle-center-layout").click().run()
                self.assertTrue(jobs.snapshot(key).running)
                self.assertTrue(app.button(key=f"right-resummary-{note.id}").disabled)
                self.assertFalse(jobs.start(key, lambda *args: "duplicate"))
                app.session_state["selected_note_id"] = second.id
                repository.update(note.id, notes="진행 중에 추가한 메모")
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(jobs.snapshot(key).id, first_id)
                self.assertTrue(jobs.snapshot(key).running)
                finish.set()
                deadline = time.monotonic() + 5
                while jobs.snapshot(key).running and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertFalse(jobs.snapshot(key).running)
                self.assertFalse(jobs.snapshot(key).error)
                self.assertEqual(repository.get_workspace(note.id).summary, "완료된 강의 노트")
                self.assertEqual(repository.get_workspace(note.id).notes, "진행 중에 추가한 메모")
                app.session_state["selected_note_id"] = note.id
                app.run()
                self.assertFalse(app.exception)
                self.assertFalse(app.button(key=f"right-resummary-{note.id}").disabled)
                self.assertFalse(app.checkbox(key=f"right-summary-memo-{note.id}").value)
                self.assertEqual(sum(component.key == f"right-summary-{note.id}" for component in app.get("component_instance")), 1)
        st.cache_resource.clear()


if __name__ == "__main__":
    unittest.main()
