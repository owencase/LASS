import tempfile
import unittest
import zlib
from contextlib import closing
from pathlib import Path
from src.storage import LectureRepository


class DocumentRollbackTests(unittest.TestCase):
    def setup_library(self, root):
        repo = LectureRepository(root / "library.db")
        note = repo.create_empty("lecture")
        repo.update(note.id, notes="memo", summary="saved")
        path = root / "lecture.pdf"
        path.write_bytes(b"%PDF")
        source = repo.add_source(note.id, path)
        with closing(repo._connect()) as db, db:
            db.execute("UPDATE lectures SET transcript = 'existing' WHERE id = ?", (note.id,))
            db.execute("CREATE TABLE document_text_pages (source_id TEXT, page_number INTEGER, data BLOB, chars INTEGER)")
            for number, text in enumerate(("first OCR page", "last OCR page"), 1):
                db.execute("INSERT INTO document_text_pages VALUES (?, ?, ?, ?)",
                    (source["id"], number, zlib.compress(text.encode()), len(text)))
        return repo, note

    def test_135_pages_convert_once_and_remain_available_to_old_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, note = self.setup_library(Path(temporary))
            repo = LectureRepository(repo.database_path)
            saved = repo.get(note.id)
            self.assertIn("existing", saved.transcript)
            self.assertIn("first OCR page", saved.transcript)
            self.assertIn("last OCR page", saved.transcript)
            self.assertEqual(saved.notes, "memo")
            self.assertEqual(saved.summary, "saved")
            again = LectureRepository(repo.database_path)
            self.assertEqual(again.get(note.id).transcript, saved.transcript)
            self.assertIn("last OCR page", "".join(again.iter_transcript_pages(note.id)))
            self.assertEqual(again.list_document_pages(note.id), [])

    def test_corrupt_page_rolls_back_conversion_without_losing_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo, note = self.setup_library(Path(temporary))
            with closing(repo._connect()) as db, db:
                db.execute("UPDATE document_text_pages SET data = ? WHERE page_number = 2", (b"corrupt",))
            with self.assertRaises(zlib.error):
                LectureRepository(repo.database_path)
            with closing(repo._connect()) as db, db:
                self.assertEqual(db.execute("SELECT transcript FROM lectures WHERE id = ?", (note.id,)).fetchone()[0], "existing")
                self.assertEqual(db.execute("SELECT count(*) FROM document_text_pages").fetchone()[0], 2)

import tempfile
import unittest
import zlib
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch
from src.pdf_utils import DocumentPage, iter_pdf_pages
from src.storage import LectureRepository
from src.rag.sources import note_excerpts


class LazyDocumentTests(unittest.TestCase):
    def test_compressed_pages_load_one_at_a_time_and_reach_summary_and_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = LectureRepository(root/'library.db')
            note = repo.create_empty('압축 문서')
            path = root/'lecture.pdf'
            path.write_bytes(b'%PDF-test')
            source = repo.add_source(note.id, path)
            text = '메모리 관리의 원리를 설명한다. ' * 1000
            def pages():
                yield DocumentPage(1, 2, '## 1페이지\n'+text)
                # OCR staging must not block a memo edit in another connection.
                repo.update(note.id, notes='추출 중 저장한 메모')
                yield DocumentPage(2, 2, '## 2페이지\n마지막 페이지의 중요한 설명이다.')
            repo.complete_document_source(source['id'], pages())
            fresh = repo.get_workspace(note.id)
            self.assertEqual(fresh.transcript, '')
            self.assertGreater(fresh.transcript_chars, len(text))
            self.assertEqual(fresh.notes, '추출 중 저장한 메모')
            with closing(repo._connect()) as connection:
                data = connection.execute('SELECT data FROM source_text_pages WHERE page_number=1').fetchone()[0]
                self.assertLess(len(data), len(text.encode('utf-8'))//10)
            original = zlib.decompress
            with patch('src.storage.note_sources.zlib.decompress', wraps=original) as unpack:
                page = next(repo.iter_document_pages(note.id, source_id=source['id'], page_number=2))
                self.assertIn('마지막 페이지', page['text'])
                self.assertEqual(unpack.call_count, 1)
            self.assertIn('마지막 페이지', ''.join(repo.iter_transcript_pages(note.id)))
            self.assertTrue(any('2페이지' in p.reference and '마지막 페이지' in p.text for p in note_excerpts([fresh])))
            self.assertEqual(len(repo.list_document_pages(note.id)), 2)

    def test_failed_extraction_does_not_publish_partial_pages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = LectureRepository(root/'library.db')
            note = repo.create_empty('노트')
            repo.update(note.id, summary='보존할 요약')
            path = root/'lecture.pdf'
            path.write_bytes(b'%PDF-test')
            source = repo.add_source(note.id, path)
            def broken():
                yield DocumentPage(1, 2, '첫 페이지 내용')
                raise RuntimeError('OCR 실패')
            with self.assertRaisesRegex(RuntimeError, 'OCR 실패'):
                repo.complete_document_source(source['id'], broken())
            self.assertEqual(repo.list_document_pages(note.id), [])
            self.assertEqual(repo.get_workspace(note.id).summary, '보존할 요약')

    def test_pdf_iterator_does_not_ocr_unrequested_pages_and_closes_reader(self):
        import pymupdf
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'scanned.pdf'
            with pymupdf.open() as document:
                for _ in range(3):
                    document.new_page()
                document.save(path)
            reader = Mock()
            with patch('src.pdf_utils.OCRReader', return_value=reader), patch('src.pdf_utils._ocr_pdf_page', return_value='페이지의 OCR 글자') as recognize:
                pages = iter_pdf_pages(path)
                self.assertEqual(next(pages).number, 1)
                recognize.assert_called_once()
                pages.close()
                reader.close.assert_called_once()
