import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.config import Settings
from src.storage import LectureRepository, LectureSegment
from src.ui.source_panel import process_source
from src.stt.transcriber import Transcription


class NoteSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = LectureRepository(self.root / 'notes.db')
        self.settings = Settings(data_dir=self.root)
        self.note = self.repo.create_empty()

    def tearDown(self):
        self.temp.cleanup()

    def source(self, name='sample.wav'):
        path = self.root / name
        path.write_bytes(b'local-test-file')
        return self.repo.add_source(self.note.id, path)

    def test_create_before_media_and_preserve_memo_across_multiple_sources(self):
        self.assertEqual(self.note.status, 'draft')
        self.assertFalse(self.note.has_media)
        self.repo.update(self.note.id, notes='먼저 작성한 메모')
        first = self.source()
        self.repo.complete_source(first['id'], transcript='첫 녹음',
            segments=[LectureSegment(0, 2, '첫 녹음')], duration=2)
        second = self.source('slides.pdf')
        note = self.repo.complete_source(second['id'], transcript='PDF 자료')
        self.assertEqual(note.id, self.note.id)
        self.assertEqual(note.notes, '먼저 작성한 메모')
        self.assertIn('첫 녹음', note.transcript)
        self.assertIn('PDF 자료', note.transcript)
        note = self.repo.update_segments(note.id, [LectureSegment(0, 2, '수정된 녹음')])
        self.assertIn('PDF 자료', note.transcript)
        self.assertIn('수정된 녹음', note.transcript)
        third = self.source('next.wav')
        note = self.repo.complete_source(third['id'], transcript='둘째 녹음',
            segments=[LectureSegment(0, 3, '둘째 녹음')], duration=3)
        self.assertEqual(note.segments[-1].start, 2)
        self.assertEqual(note.duration, 5)
        self.assertIn('수정된 녹음', note.transcript)

    def test_completion_is_idempotent_and_does_not_duplicate_transcript(self):
        source = self.source()
        first = self.repo.complete_source(source['id'], transcript='유일한 내용')
        second = self.repo.complete_source(source['id'], transcript='중복 내용')
        self.assertEqual(first.transcript, second.transcript)

    def test_source_processing_state_can_resume_and_complete(self):
        source = self.source('slides.pdf')
        self.repo.begin_source_processing(source['id'])
        self.assertEqual(self.repo.get_source(source['id'])['status'], 'processing')
        note = self.repo.complete_source(source['id'], transcript='PDF 자료')
        self.assertEqual(self.repo.get_source(source['id'])['status'], 'ready')
        self.assertIn('PDF 자료', note.transcript)

    def test_live_chunks_append_and_recent_view_is_bounded(self):
        first = self.repo.append_live_transcription(
            self.note.id,
            transcript='첫 구간',
            segments=[LectureSegment(0, 2, '첫 구간')],
            duration=2,
            language='ko',
            start_session=True,
        )
        second = self.repo.append_live_transcription(
            self.note.id,
            transcript='둘째 구간',
            segments=[LectureSegment(0, 3, '둘째 구간')],
            duration=3,
            language='ko',
        )
        self.assertIn('## 실시간 녹음', first.transcript)
        self.assertEqual(second.segments[-1].start, 2)
        self.assertEqual(second.duration, 5)
        recent = self.repo.recent_segments(self.note.id, limit=1)
        self.assertEqual([item.text for item in recent], ['둘째 구간'])

    def test_transcript_pages_are_read_in_bounded_sqlite_slices(self):
        source = self.source()
        text = '가나다라마바사' * 300
        self.repo.complete_source(source['id'], transcript=text)
        pages = list(self.repo.iter_transcript_pages(self.note.id, page_chars=256))
        self.assertEqual(''.join(pages).split(f'## {source["name"]}')[-1].strip(), text)
        self.assertTrue(all(len(page) <= 256 for page in pages))

    def test_summary_failure_keeps_transcription_and_retry_skips_stt(self):
        source = self.source()
        pipeline = Mock()
        pipeline.transcribe.return_value = Transcription('전사 결과', 'ko', 2, [])
        pipeline.resummarize.side_effect = RuntimeError('summary failed')
        with self.assertRaisesRegex(RuntimeError, 'summary failed'):
            process_source(self.repo, source, self.settings, lambda: pipeline,
                           'default_summary', 'ko', True, lambda value: None)
        self.assertEqual(self.repo.get_source(source['id'])['status'], 'ready')
        self.assertIn('전사 결과', self.repo.get(self.note.id).transcript)
        pipeline.resummarize.side_effect = None
        pipeline.resummarize.return_value = Mock(summary='요약 완료', prompt_name='기본 요약', vector_collection=None)
        process_source(self.repo, self.repo.get_source(source['id']), self.settings, lambda: pipeline,
                       'default_summary', 'ko', True, lambda value: None)
        self.assertEqual(pipeline.transcribe.call_count, 1)
        self.assertEqual(self.repo.get(self.note.id).summary, '요약 완료')

    def test_lightweight_library_excludes_large_text(self):
        self.repo.update(self.note.id, notes='검색할 메모', summary='요약' * 1000)
        row = self.repo.list('검색할', lightweight=True)[0]
        self.assertEqual(row.id, self.note.id)
        self.assertEqual(row.summary, '')
        self.assertEqual(row.notes, '')
        self.assertEqual(row.segments, [])

    def test_legacy_media_exports_without_full_file_join(self):
        data = b'0123456789' * 3000
        self.repo.MEDIA_CHUNK_SIZE = 1024
        note = self.repo.create(title='legacy', source_name='old.wav', media_data=data, media_type='.wav')
        destination = self.repo.export_media(note.id, self.root / 'preview.wav')
        self.assertEqual(destination.read_bytes(), data)
