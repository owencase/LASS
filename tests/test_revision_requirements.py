import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from src.backup import backup_sources
from src.ocr import layout_text
from src.pdf_utils import iter_pdf_pages


def box(x, y, width, height=16):
    return [[x,y],[x+width,y],[x+width,y+height],[x,y+height]]


class RevisionTests(unittest.TestCase):
    def test_word_gaps_and_code_indentation_are_retained(self):
        words = ((('hello',1,box(0,0,50)),('world',1,box(70,0,50))),)
        result = SimpleNamespace(txts=('helloworld',), boxes=[box(0,0,120)], word_results=words)
        self.assertEqual(layout_text(result), 'hello world')
        code = SimpleNamespace(txts=('def f():','return 1;'), boxes=[box(0,0,64),box(32,20,72)],word_results=())
        text = layout_text(code)
        self.assertIn('```\ndef f():\n    return 1;\n```', text)
        fragments = SimpleNamespace(txts=('!= null;', 'return name'),
            boxes=[box(140,20,64,12), box(32,10,88,30)], word_results=())
        self.assertIn('return name != null;', layout_text(fragments))

    def test_uploaded_transcription_finishes_after_screen_leaves(self):
        from threading import Event
        import time
        from src.config import Settings
        from src.storage import LectureRepository
        from src.summary_jobs import SummaryJobs
        from src.ui.app import _attach_and_process_path
        from src.stt.transcriber import Transcription
        entered, finish = Event(), Event()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = LectureRepository(root/'lass.db')
            note = repo.create_empty()
            audio = root/'audio.wav'
            audio.write_bytes(b'audio')
            jobs = SummaryJobs()
            def transcribe(*args, **kwargs):
                entered.set()
                if not finish.wait(5):
                    raise RuntimeError('test timeout')
                return Transcription('persisted after navigation', 'ko', 1, [])
            pipeline = SimpleNamespace(transcribe=transcribe)
            with patch('src.ui.app.st.session_state', {}), patch('src.ui.app.st.rerun'), \
                 patch('src.ui.app.get_pipeline', return_value=pipeline), \
                 patch('src.ui.app._release_live_transcriber'), patch('src.ui.app.summary_jobs', jobs):
                _attach_and_process_path(repo, note, Settings(data_dir=root), audio, kind='audio')
                self.assertTrue(entered.wait(5))
            finish.set()
            source = repo.list_sources(note.id)[0]
            key = ('source', str(repo.database_path), note.id, source['id'])
            deadline = time.monotonic()+5
            while jobs.snapshot(key).running and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertFalse(jobs.snapshot(key).running)
            self.assertFalse(jobs.snapshot(key).error)
            reopened = LectureRepository(repo.database_path)
            self.assertIn('persisted after navigation', reopened.get(note.id).transcript)
            self.assertEqual(reopened.get_source(source['id'])['status'], 'ready')

    def test_image_code_is_ocrd_even_when_pdf_has_native_prose(self):
        import io
        import pymupdf
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'mixed.pdf'
            image = io.BytesIO()
            Image.new('RGB',(320,120),'white').save(image,format='PNG')
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((40,40),'Native explanatory paragraph with more than forty visible characters about programming.')
                page.insert_image(pymupdf.Rect(40,70,360,190),stream=image.getvalue())
                document.save(path)
            with patch('src.pdf_utils.OCRReader') as factory, patch('src.pdf_utils._ocr_pdf_page',return_value='```\npublic void login() { }\n```') as read:
                pages = list(iter_pdf_pages(path))
            self.assertIn('Native explanatory',pages[0].text)
            self.assertIn('public void login()',pages[0].text)
            self.assertTrue(pages[0].ocr)
            self.assertEqual(len(read.call_args.args),3)
            factory.return_value.close.assert_called_once()

    def test_backup_includes_source_and_config_but_not_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'project';root.mkdir()
            for directory in ('src','models'):(root/directory).mkdir()
            (root/'src/main.py').write_text('old source')
            (root/'.env').write_text('SETTING=old')
            (root/'models/large.bin').write_bytes(b'model')
            archive = backup_sources(root, Path(temporary)/'Backups')
            (root/'src/main.py').write_text('new source')
            with zipfile.ZipFile(archive) as saved:
                self.assertEqual(saved.read('src/main.py'),b'old source')
                self.assertEqual(saved.read('.env'),b'SETTING=old')
                self.assertNotIn('models/large.bin',saved.namelist())
                self.assertIsNone(saved.testzip())
