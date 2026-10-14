import base64
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

from src.config import Settings
from src.storage import LectureRepository, LectureSegment
from src.stt.transcriber import Transcription
from src.summary_jobs import SummaryJobs
from src.ui.source_panel import process_source


class SessionPersistenceTests(unittest.TestCase):
    def test_legacy_migration_sessions_reopen_and_full_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lass.db'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE global_chat_messages(id TEXT PRIMARY KEY, role TEXT, content TEXT, created_at TEXT)')
                db.execute("INSERT INTO global_chat_messages VALUES ('old', 'user', '이전 질문', '2026-01-01')")
            db.close()
            repo = LectureRepository(path)
            self.assertEqual(repo.list_chat_messages(None)[0].content, '이전 질문')
            session = repo.create_chat_session()
            for index in range(105):
                repo.add_chat_message(None, 'user', f'질문 {index}', session_id=session)
            reopened = LectureRepository(path)
            self.assertEqual(reopened.active_chat_session(), session)
            self.assertEqual(len(reopened.list_chat_messages(None, limit=None)), 105)
            self.assertEqual(reopened.list_chat_sessions()[0]['title'], '질문 0')
            reopened.set_metadata('active_chat_session', 'legacy')
            self.assertEqual(reopened.list_chat_messages(None)[0].content, '이전 질문')
            reopened.clear_chat_messages(None)
            self.assertEqual(len(reopened.list_chat_messages(None, limit=None, session_id=session)), 105)

    def test_live_commit_is_durable_and_replay_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lass.db'
            repo = LectureRepository(path)
            note = repo.create_empty('녹음')
            repo.update(note.id, notes='메모', summary='요약')
            for _ in range(2):
                repo = LectureRepository(path)
                repo.append_live_transcription(note.id, transcript='완료된 전사',
                    segments=[LectureSegment(0, 2, '완료된 전사')], duration=2, event_nonce='chunk-1')
            saved = repo.get(note.id)
            self.assertEqual(saved.transcript, '완료된 전사')
            self.assertEqual(saved.duration, 2)
            self.assertEqual(len(saved.segments), 1)
            self.assertEqual((saved.notes, saved.summary), ('메모', '요약'))

    def test_index_cleanup_failure_cannot_lose_completed_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(data_dir=Path(directory))
            repo = LectureRepository(settings.database_path)
            note = repo.create_empty()
            repo.replace_summary(note.id, summary='', prompt_key='', prompt_name='', vector_collection='old-index')
            audio = Path(directory) / 'audio.wav'
            audio.write_bytes(b'audio')
            source = repo.add_source(note.id, audio)
            pipeline = Mock()
            pipeline.transcribe.return_value = Transcription('보존할 전사', 'ko', 2, [])
            with patch('src.ui.source_panel.SQLiteVectorStore.delete_collection', side_effect=RuntimeError('index failed')):
                with self.assertRaisesRegex(RuntimeError, 'index failed'):
                    process_source(repo, source, settings, lambda: pipeline, 'lecture_detail', None, False, lambda value: None)
            repo = LectureRepository(settings.database_path)
            self.assertEqual(repo.get_source(source['id'])['status'], 'ready')
            self.assertIn('보존할 전사', repo.get(note.id).transcript)

    def test_renamed_sessions_persist_and_deleted_legacy_does_not_return(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lass.db'
            repo = LectureRepository(path)
            repo.rename_chat_session('legacy', '직접 지정한 이름')
            repo.add_chat_message(None, 'user', '이름을 덮어쓰면 안 됩니다.', session_id='legacy')
            second = repo.create_chat_session()
            repo.add_chat_message(None, 'user', '남길 대화', session_id=second)
            repo = LectureRepository(path)
            self.assertEqual(next(row['title'] for row in repo.list_chat_sessions() if row['id'] == 'legacy'), '직접 지정한 이름')
            repo.delete_chat_session('legacy')
            reopened = LectureRepository(path)
            self.assertNotIn('legacy', [row['id'] for row in reopened.list_chat_sessions()])
            self.assertEqual(reopened.list_chat_messages(None, session_id='legacy'), [])
            self.assertEqual(reopened.list_chat_messages(None, session_id=second)[0].content, '남길 대화')
            reopened.delete_chat_session(second)
            replacement = reopened.active_chat_session()
            self.assertNotIn(replacement, ('legacy', second))
            with self.assertRaises(KeyError):
                reopened.add_chat_message(None, 'assistant', '늦은 응답', session_id=second)

    def test_fallback_session_stays_selected_when_another_background_reply_arrives(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = LectureRepository(Path(directory) / 'lass.db')
            first = repo.create_chat_session()
            second = repo.create_chat_session()
            repo.delete_chat_session(second)
            self.assertEqual(repo.active_chat_session(), first)
            repo.add_chat_message(None, 'assistant', '다른 세션의 응답', session_id='legacy')
            self.assertEqual(repo.active_chat_session(), first)

    def test_live_worker_saves_after_ui_state_is_lost_and_acks_after_commit(self):
        from src.ui.app import _handle_live_recorder_event, _selected_note
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(data_dir=Path(directory))
            repo = LectureRepository(settings.database_path)
            note = repo.create_empty('선택한 노트')
            other = repo.create_empty('다른 노트')
            repo.set_metadata('selected_note_id', note.id)
            jobs, entered, finish = SummaryJobs(), Event(), Event()
            transcriber = Mock()
            def transcribe(path, *, on_segment=None):
                entered.set()
                self.assertTrue(finish.wait(5))
                return Transcription('백그라운드 전사', 'ko', 2, [])
            transcriber.transcribe.side_effect = transcribe
            state = {}
            event = dict(action='audio_chunk', nonce='live-1', data=base64.b64encode(b'audio').decode(), final=True)
            with patch('src.ui.app.st.session_state', state), patch('src.ui.app.st.rerun'), \
                 patch('src.ui.app.summary_jobs', jobs), patch('src.ui.app.get_settings', return_value=settings), \
                 patch('src.ui.app._get_live_transcriber', return_value=transcriber):
                self.assertEqual(_selected_note(repo, [other, note]).id, note.id)
                _handle_live_recorder_event(event, repo, note)
                self.assertTrue(entered.wait(5))
                self.assertNotIn('live-recorder-ack', state)
                state.clear()
                finish.set()
                deadline = time.monotonic() + 5
                while jobs.is_running() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertFalse(jobs.is_running())
                self.assertIn('백그라운드 전사', LectureRepository(settings.database_path).get(note.id).transcript)
                _handle_live_recorder_event(event, repo, note)
                self.assertEqual(state['live-recorder-ack'], 'live-1')
                transcriber.transcribe.assert_called_once()


if __name__ == '__main__':
    unittest.main()
