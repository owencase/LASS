import tempfile
import unittest
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from src.storage import LectureRepository


@dataclass
class Segment:
    start: float
    end: float
    text: str


class LectureRepositoryTests(unittest.TestCase):
    def test_media_is_chunked_inside_the_same_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "lass.db"
            repository = LectureRepository(database_path)
            with patch.object(LectureRepository, "MEDIA_CHUNK_SIZE", 3):
                note = repository.create(
                    title="긴 녹음",
                    source_name="long.wav",
                    media_data=b"12345678",
                    media_type=".wav",
                )

            self.assertEqual(repository.get_media(note.id), b"12345678")
            with closing(repository._connect()) as connection:
                chunk_count = connection.execute(
                    "SELECT count(*) FROM lecture_media_chunks WHERE lecture_id = ?",
                    (note.id,),
                ).fetchone()[0]
                legacy_blob = connection.execute(
                    "SELECT media_blob FROM lectures WHERE id = ?", (note.id,)
                ).fetchone()[0]
            self.assertEqual(chunk_count, 3)
            self.assertIsNone(legacy_blob)

    def test_create_complete_update_search_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "lecture.mp3"
            source.write_bytes(b"audio")
            repository = LectureRepository(root / "lass.db")

            created = repository.create(
                title="벡터 검색 강의",
                source_name=source.name,
                media_data=b"audio",
                media_type=".mp3",
                prompt_key="lecture_detail",
            )
            self.assertEqual(created.status, "processing")
            self.assertTrue(created.has_media)
            self.assertEqual(repository.get_media(created.id), b"audio")

            completed = repository.complete(
                created.id,
                prompt_name="강의 상세 노트",
                language="ko",
                duration=12.5,
                transcript="벡터 데이터베이스 설명",
                segments=[Segment(0, 4, "벡터 데이터베이스 설명")],
                summary="핵심 요약",
                vector_collection="lass-test",
            )
            self.assertEqual(completed.status, "ready")
            self.assertEqual(completed.segments[0].text, "벡터 데이터베이스 설명")

            updated = repository.update(
                created.id, title="수정된 제목", notes="복습할 내용"
            )
            self.assertEqual(updated.title, "수정된 제목")
            self.assertEqual(updated.notes, "복습할 내용")
            self.assertEqual(repository.list("복습")[0].id, created.id)
            workspace = repository.get_workspace(created.id)
            self.assertIsNotNone(workspace)
            assert workspace is not None
            self.assertEqual(workspace.transcript, "")
            self.assertEqual(workspace.transcript_chars, len("벡터 데이터베이스 설명"))
            self.assertEqual(workspace.segments, [])
            self.assertEqual(workspace.segment_count, 1)
            self.assertEqual(workspace.summary, "핵심 요약")
            self.assertEqual(workspace.notes, "복습할 내용")

            deleted = repository.delete(created.id)
            self.assertIsNotNone(deleted)
            self.assertIsNone(repository.get(created.id))
            self.assertIsNone(repository.get_media(created.id))

    def test_failed_lecture_keeps_error_for_library_management(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = LectureRepository(root / "lass.db")
            lecture = repository.create(
                title="실패한 강의",
                source_name="bad.wav",
                media_data=b"broken audio",
                media_type=".wav",
            )

            repository.mark_failed(lecture.id, "음성을 읽을 수 없습니다")
            failed = repository.get(lecture.id)

            self.assertIsNotNone(failed)
            assert failed is not None
            self.assertEqual(failed.status, "failed")
            self.assertIn("읽을 수 없습니다", failed.error_message or "")

    def test_folder_crud_move_and_delete_preserves_notes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = LectureRepository(Path(temporary_directory) / "lass.db")
            study = repository.create_folder("공부")
            archive = repository.create_folder("보관함")
            note = repository.create(
                title="RAG 정리",
                source_name="rag.mp3",
                media_data=b"audio",
                media_type=".mp3",
                folder_id=study.id,
            )

            self.assertEqual(note.folder_id, study.id)
            self.assertEqual([folder.name for folder in repository.list_folders()], ["공부", "보관함"])

            moved = repository.move(note.id, archive.id)
            self.assertEqual(moved.folder_id, archive.id)
            renamed = repository.rename_folder(archive.id, "완료")
            self.assertEqual(renamed.name, "완료")

            with self.assertRaises(ValueError):
                repository.create_folder("공부")
            with self.assertRaises(ValueError):
                repository.rename_folder(archive.id, "공부")

            repository.delete_folder(archive.id)
            preserved = repository.get(note.id)
            self.assertIsNotNone(preserved)
            assert preserved is not None
            self.assertIsNone(preserved.folder_id)
            self.assertEqual(repository.get_media(note.id), b"audio")

    def test_folder_listing_can_stay_lightweight_until_content_is_needed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = LectureRepository(Path(temporary_directory) / "lass.db")
            folder = repository.create_folder("지연 로딩")
            note = repository.create_empty(folder_id=folder.id)
            repository.update(note.id, summary="디스크에서 나중에 읽을 요약")

            listed = repository.list_by_folder(folder.id, lightweight=True)

            self.assertEqual(listed[0].summary, "")
            self.assertTrue(repository.folder_has_content(folder.id))
            self.assertEqual(
                next(repository.iter_by_folder(folder.id)).summary,
                "디스크에서 나중에 읽을 요약",
            )

    def test_transcript_summary_and_folder_chat_updates_are_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = LectureRepository(Path(temporary_directory) / "lass.db")
            folder = repository.create_folder("프로젝트")
            note = repository.create(
                title="회의",
                source_name="meeting.wav",
                media_data=b"audio",
                media_type=".wav",
                folder_id=folder.id,
            )
            repository.complete(
                note.id,
                prompt_name="기존 요약",
                language="ko",
                duration=10,
                transcript="기존 녹취",
                segments=[Segment(0, 4, "기존 녹취")],
                summary="기존 요약문",
                vector_collection="note-vector",
            )

            updated = repository.update_segments(
                note.id,
                [Segment(1, 5, "수정된 녹취"), Segment(5, 8, "두 번째 문장")],
            )
            self.assertEqual(updated.transcript, "수정된 녹취 두 번째 문장")
            self.assertIsNone(updated.vector_collection)

            summarized = repository.replace_summary(
                note.id,
                summary="메모를 포함한 새 요약",
                prompt_key="lecture_detail",
                prompt_name="기본 핵심 요약",
                vector_collection="new-vector",
            )
            self.assertEqual(summarized.summary, "메모를 포함한 새 요약")
            self.assertEqual(summarized.prompt_key, "lecture_detail")

            repository.add_chat_message(folder.id, "user", "핵심은?")
            repository.add_chat_message(folder.id, "assistant", "수정된 녹취입니다.")
            messages = repository.list_chat_messages(folder.id)
            self.assertEqual([message.role for message in messages], ["user", "assistant"])
            self.assertEqual(messages[-1].content, "수정된 녹취입니다.")

            repository.delete_folder(folder.id)
            self.assertEqual(repository.list_chat_messages(folder.id), [])


if __name__ == "__main__":
    unittest.main()
