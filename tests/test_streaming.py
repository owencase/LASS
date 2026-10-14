import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch


from src.llm.model_loader import LocalEXAONEClient


class StreamingTests(unittest.TestCase):
    def client(self, generate, capacity=4096):
        model = SimpleNamespace(n_ctx=lambda: capacity, create_completion=generate, reset=lambda: None)
        client = LocalEXAONEClient(Path('.'), temperature=0)
        client._runtime = lambda: model
        client._bounded_input = lambda *args: [4, 5, 6]
        return client

    def test_langchain_streams_before_completion_and_has_no_old_1500_limit(self):
        observed = []
        options = {}
        def generate(**kwargs):
            options.update(kwargs)
            yield {'choices': [{'text': '단어 ', 'finish_reason': None}]}
            self.assertEqual(observed, ['단어 '])
            yield {'choices': [{'text': '단어 ' * 1499, 'finish_reason': 'stop'}]}
        result = self.client(generate).chat('지시', '질문', on_token=observed.append)
        self.assertEqual(result.count('단어'), 1500)
        self.assertEqual(options['max_tokens'], 4093)

    def test_generation_failure_reaches_caller(self):
        def generate(**kwargs):
            raise RuntimeError('generation failed')
            yield
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            self.client(generate).chat('지시', '질문')

    def test_context_exhaustion_is_an_error_even_after_partial_output(self):
        def generate(**kwargs):
            yield {'choices': [{'text': '부분', 'finish_reason': None}]}
            yield {'choices': [{'text': '', 'finish_reason': 'length'}]}
        partial = []
        with self.assertRaisesRegex(RuntimeError, '문맥 용량'):
            self.client(generate, 4).chat('지시', '질문', on_token=partial.append)
        self.assertEqual(partial, ['부분'])

    def test_closing_stream_closes_native_iterator(self):
        finished = Event()
        def generate(**kwargs):
            try:
                yield {'choices': [{'text': '부분', 'finish_reason': None}]}
                self.fail('generation continued after cancellation')
            finally:
                finished.set()
        stream = self.client(generate).stream_messages([{'role': 'user', 'content': '질문'}])
        next(stream)
        stream.close()
        self.assertTrue(finished.is_set())


class SummaryPersistenceTests(unittest.TestCase):
    def test_partial_summary_is_not_saved_on_failure(self):
        import time
        import tempfile
        from src.ui.app import _resummarize_note, _summary_job_key
        from src.summary_jobs import SummaryJobs
        from src.storage import LectureRepository
        from src.config import Settings
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(data_dir=Path(directory))
            repository = LectureRepository(Path(directory) / "lass.db")
            note = repository.create_empty("노트")
            note = repository.update(note.id, summary="saved")
            jobs, pipeline = SummaryJobs(), Mock()

            def fail(*args, on_token, **kwargs):
                on_token("부분 결과")
                raise RuntimeError("문맥 용량")

            pipeline.resummarize_pages.side_effect = fail
            with patch("src.ui.app.get_pipeline", return_value=pipeline), \
                 patch("src.ui.app.summary_jobs", jobs), patch("src.ui.app.st.rerun"):
                _resummarize_note(repository, note, settings, "lecture_detail", False)
                deadline = time.monotonic() + 5
                key = _summary_job_key(repository, note.id)
                while jobs.snapshot(key).running and time.monotonic() < deadline:
                    time.sleep(.01)
            self.assertFalse(jobs.snapshot(key).running)
            self.assertIn("문맥 용량", jobs.snapshot(key).error)
            self.assertEqual(repository.get_workspace(note.id).summary, "saved")
