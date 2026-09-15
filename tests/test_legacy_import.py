import tempfile
import unittest
from pathlib import Path

from src.storage import LectureRepository, migrate_legacy_storage


class LegacyImportTests(unittest.TestCase):
    def test_keeps_pdf_on_disk_for_streaming_viewer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "large.pdf"
            source.write_bytes(b"%PDF-large")
            repository = LectureRepository(root / "lass.db")
            note = repository.create(
                title="Large PDF",
                source_name=source.name,
                source_path=source,
                media_type=".pdf",
                file_size=source.stat().st_size,
            )

            result = migrate_legacy_storage(repository, root)

            self.assertEqual(result.embedded_media, 0)
            self.assertEqual(repository.get(note.id).source_path, str(source))
            self.assertFalse(repository.get(note.id).has_media)

    def test_migrates_media_and_removes_legacy_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs = root / "inputs"
            outputs = root / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            (root / "vector_db").mkdir()
            (root / "vector_db" / "derived.index").write_bytes(b"old")
            source = inputs / "1a2b3c4d_sample.mp3"
            source.write_bytes(b"audio")
            (outputs / "1a2b3c4d_sample_summary.md").write_text(
                """# 1a2b3c4d_sample 요약

- 요약 형식: 기본 요약
- 감지 언어: ko

## 요약

핵심 내용입니다.

## 타임스탬프 스크립트

- `00:00:03` 첫 번째 문장
- `00:01:05` 두 번째 문장
""",
                encoding="utf-8",
            )
            repository = LectureRepository(root / "lass.db")

            result = migrate_legacy_storage(repository, root)

            self.assertEqual(result.imported_reports, 1)
            self.assertEqual(result.removed_directories, 3)
            self.assertFalse(result.warnings)
            lectures = repository.list()
            self.assertEqual(len(lectures), 1)
            self.assertEqual(lectures[0].title, "sample")
            self.assertEqual(lectures[0].source_name, "sample.mp3")
            self.assertEqual(lectures[0].segments[1].start, 65)
            self.assertIn("핵심 내용", lectures[0].summary)
            self.assertTrue(lectures[0].has_media)
            self.assertEqual(repository.get_media(lectures[0].id), b"audio")
            self.assertFalse(inputs.exists())
            self.assertFalse(outputs.exists())
            self.assertFalse((root / "vector_db").exists())

            second = migrate_legacy_storage(repository, root)
            self.assertEqual(second.imported_reports, 0)
            self.assertEqual(len(repository.list()), 1)


if __name__ == "__main__":
    unittest.main()
