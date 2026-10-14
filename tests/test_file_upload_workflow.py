"""End-to-end upload and rerun checks with disposable local documents."""
import io
import os
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pymupdf
import streamlit as st
from streamlit.testing.v1 import AppTest
from src.config import Settings
from src.storage import LectureRepository
from src.summary_jobs import SummaryJobs
from src.stt.transcriber import Transcription

ROOT = Path(__file__).parents[1]


def document_bytes(suffix):
    if suffix == ".pdf":
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72, 72), "A document with enough readable text to verify that upload and rendering remain available.")
            return pdf.tobytes()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>PowerPoint upload check with enough readable text.</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>')
    return output.getvalue()


class FileUploadWorkflowTests(unittest.TestCase):
    def test_document_dialog_success_error_and_cancel_keep_workspace(self):
        for suffix, broken, cancel in [(".pdf", False, False), (".pptx", False, False), (".pdf", True, False), (".pptx", True, False), (".pdf", False, True)]:
            with self.subTest(suffix=suffix, broken=broken, cancel=cancel), tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LASS_DATA_DIR": directory}):
                st.cache_resource.clear()
                os.environ.pop("LASS_DESKTOP_PICKER", None)
                repo = LectureRepository(Path(directory) / "lass.db")
                note = repo.create_empty("upload check")
                upload = io.BytesIO(b"broken file" if broken else document_bytes(suffix))
                upload.name = "sample" + suffix
                with patch("src.ui.app.LocalModelManager.status", return_value=SimpleNamespace(ready=True)), patch("src.ui.app.st.file_uploader", return_value=upload), patch("src.document_preview.convert_presentation", side_effect=RuntimeError("No Office in test")):
                    app = AppTest.from_file(str(ROOT / "src/ui/app.py"), default_timeout=20)
                    app.session_state["selected_note_id"] = note.id
                    app.run()
                    app.button(key=f"document-upload-{note.id}").click().run()
                    action = f"browser-upload-cancel-document-{note.id}" if cancel else f"browser-upload-save-document-{note.id}"
                    app.button(key=action).click().run()
                    self.assertFalse(app.exception)
                    self.assertIn(f"document-upload-{note.id}", [button.key for button in app.button])
                    self.assertIn(f"audio-upload-{note.id}", [button.key for button in app.button])
                    self.assertFalse(app.session_state[f"browser-upload-open-document-{note.id}"])
                    sources = repo.list_sources(note.id)
                    self.assertEqual(len(sources), 0 if cancel else 1)
                    if not cancel:
                        self.assertEqual(sources[0]["status"], "failed" if broken else "ready")
                        self.assertTrue(Path(sources[0]["path"]).is_file())
                st.cache_resource.clear()

    def test_audio_video_uploads_finish_after_rerun(self):
        from src.ui.app import _attach_and_process_upload
        for suffix in (".wav", ".mp4"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                repo = LectureRepository(Path(directory) / "lass.db")
                note = repo.create_empty()
                upload = io.BytesIO(b"test media bytes")
                upload.name = "sample" + suffix
                pipeline = Mock()
                pipeline.transcribe.return_value = Transcription("saved transcription", "en", 1, [])
                jobs = SummaryJobs()
                state = {}
                with patch("src.ui.app.st.session_state", state), patch("src.ui.app.st.rerun"), patch("src.ui.app._release_live_transcriber"), patch("src.ui.app.get_pipeline", return_value=pipeline), patch("src.ui.app.summary_jobs", jobs):
                    _attach_and_process_upload(repo, note, Settings(data_dir=Path(directory)), upload, kind="audio")
                    source = repo.list_sources(note.id)[0]
                    key = ("source", str(repo.database_path), note.id, source["id"])
                    deadline = time.monotonic() + 5
                    while jobs.snapshot(key).running and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertFalse(jobs.snapshot(key).error)
                    self.assertEqual(repo.get_source(source["id"])["status"], "ready")
                    self.assertIn("saved transcription", repo.get(note.id).transcript)
                    self.assertFalse(state[f"browser-upload-open-audio-{note.id}"])

    def test_empty_and_unsupported_uploads_do_not_add_sources_or_leave_copies(self):
        from src.ui.app import _attach_and_process_upload
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = LectureRepository(root / "lass.db")
            note = repo.create_empty()
            for name, value in (("empty.pdf", b""), ("program.exe", b"unsupported")):
                upload = io.BytesIO(value)
                upload.name = name
                with patch("src.ui.app.st.session_state", {}), patch("src.ui.app.st.rerun"), patch("src.ui.app.st.error"):
                    _attach_and_process_upload(repo, note, Settings(data_dir=root), upload, kind="document")
                self.assertEqual(repo.list_sources(note.id), [])
                self.assertEqual(list((root / "media").glob("*")), [])
