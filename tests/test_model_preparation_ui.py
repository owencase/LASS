import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from src.summary_jobs import SummaryJobs


class ModelPreparationUITests(unittest.TestCase):
    def test_desktop_setup_download_failure_resume_and_automatic_workspace_entry(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        statuses = {'llm':False, 'stt':False}
        jobs = SummaryJobs()
        attempts = []
        def download(settings, key, progress):
            attempts.append(key)
            progress(.4, 'weights')
            if len(attempts) == 1:
                raise OSError('offline')
            statuses[key] = True
            progress(1, '완료')
        def wait_jobs():
            deadline = time.monotonic() + 3
            while jobs.is_running() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertFalse(jobs.is_running())
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            'LASS_DATA_DIR':directory, 'LASS_EXAONE_MODEL_DIR':str(Path(directory)/'llm'),
            'LASS_STT_MODEL_DIR':str(Path(directory)/'stt'), 'LASS_DESKTOP_PICKER':'http://127.0.0.1/test',
        }), patch('src.ui.model_preparation.model_ready', side_effect=lambda settings,key: statuses[key]), \
             patch('src.ui.model_preparation.model_jobs', jobs), \
             patch('src.ui.model_preparation.download_model', side_effect=download):
            app=AppTest.from_file(str(Path(__file__).parents[1]/'src/ui/app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.button),2)
            self.assertEqual(app.button(key='download-llm').label,'다운로드')
            app.button(key='download-llm').click().run()
            wait_jobs()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.button(key='download-llm').label,'이어받기')
            self.assertIn('offline',app.error[0].value)
            app.button(key='download-llm').click().run()
            wait_jobs()
            app.run()
            self.assertTrue(app.button(key='download-llm').disabled)
            self.assertTrue(any('완료' in item.value for item in app.success))
            app.button(key='download-stt').click().run()
            wait_jobs()
            app.run()
            self.assertFalse(app.exception)
            self.assertIn('chatting-button',[button.key for button in app.button])
            self.assertNotIn('download-llm',[button.key for button in app.button])
            self.assertEqual(attempts,['llm','llm','stt'])

    def test_job_progress_preserves_byte_fraction(self):
        from threading import Event
        jobs=SummaryJobs()
        entered,release=Event(),Event()
        def work(progress,token):
            progress('weights',.435)
            entered.set()
            release.wait(3)
        jobs.start('model',work)
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(jobs.snapshot('model').fraction,.435)
            self.assertEqual(jobs.snapshot('model').progress,'weights')
        finally:
            release.set()
