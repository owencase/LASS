import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from src.storage import LectureRepository
from src.updater import backup_data, check_update, download_update, release_url


class UpdateTests(unittest.TestCase):
    def test_release_requires_new_stable_version_installer_and_checksum(self):
        base = "https://github.com/owencase/LASS/releases/download/v1.3.2/"
        release = {"name": "LASS IQ4 Moonshine", "tag_name": "v1.3.2", "assets": [
            {"name": "LASS-Setup-1.3.2.exe", "size": 100, "browser_download_url": base + "LASS-Setup-1.3.2.exe"},
            {"name": "LASS-Setup-1.3.2.exe.sha256", "browser_download_url": base + "LASS-Setup-1.3.2.exe.sha256"}]}
        def response():
            return io.BytesIO(json.dumps(release).encode())
        with patch("src.updater.__version__", "1.3.1"), patch("src.updater.urlopen", side_effect=lambda *a, **kw: response()):
            self.assertEqual(check_update()["version"], "1.3.2")
            with patch("src.updater.__version__", "1.3.2"):
                self.assertIsNone(check_update())
            release["prerelease"] = True
            self.assertIsNone(check_update())
            release["prerelease"] = False
            release["assets"].pop()
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                check_update()
        with self.assertRaises(ValueError):
            release_url("https://github.com.evil.test/owencase/LASS/releases/download/file")

    def test_install_download_is_published_only_after_checksum_validation(self):
        payload = b"test installer"
        checksum = hashlib.sha256(payload).hexdigest().encode()
        base = "https://github.com/owencase/LASS/releases/download/v1.3.2/"
        release = {"version": "1.3.2", "size": len(payload), "url": base + "LASS-Setup-1.3.2.exe",
                   "checksum_url": base + "LASS-Setup-1.3.2.exe.sha256"}
        def file_response(*args, **kwargs):
            handle = io.BytesIO(payload)
            handle.status = 200
            return handle
        with tempfile.TemporaryDirectory() as directory, patch("src.model_assets.urlopen", side_effect=file_response):
            with patch("src.updater.urlopen", return_value=io.BytesIO(checksum)):
                path = download_update(release, directory)
                self.assertEqual(path.read_bytes(), payload)
            path.unlink()
            with patch("src.updater.urlopen", return_value=io.BytesIO(b"0" * 64)):
                with self.assertRaises(OSError):
                    download_update(release, directory)
            self.assertFalse(path.exists())

    def test_backup_preserves_notes_summary_media_and_committed_wal(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            repo = LectureRepository(data / "lass.db")
            folder = repo.create_folder("강의")
            note = repo.create_empty("노트", folder_id=folder.id)
            repo.update(note.id, notes="메모", summary="기존 요약")
            repo.add_chat_message(None, "user", "대화")
            media = data / "media/test.wav"
            media.parent.mkdir()
            media.write_bytes(b"audio")
            with closing(sqlite3.connect(str(data / "lass.db"))) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("UPDATE lectures SET summary='최신 요약' WHERE id=?", (note.id,))
                connection.commit()
                snapshot = backup_data(data)
            # Reopen just as an upgraded app does; original and backup retain data.
            for root in (data, snapshot):
                restored = LectureRepository(root / "lass.db")
                self.assertEqual(restored.get(note.id).summary, "최신 요약")
                self.assertEqual(restored.get(note.id).notes, "메모")
                self.assertEqual(restored.get(note.id).folder_id, folder.id)
                self.assertEqual(restored.list_chat_messages(None)[0].content, "대화")
                self.assertEqual((root / "media/test.wav").read_bytes(), b"audio")
            self.assertFalse((snapshot / "INCOMPLETE").exists())

    def test_installer_explicit_uninstall_removes_owned_data_and_preserves_other_versions(self):
        script = (Path(__file__).parents[1] / "scripts/installer.iss").read_text(encoding="utf-8")
        self.assertIn("UninstallLogMode=overwrite", script)
        self.assertIn('[UninstallDelete]', script)
        self.assertIn('Type: filesandordirs; Name: "{#MyDataRoot}"',script)
        self.assertIn('Check: DedicatedInstallDirectory',script)
        self.assertNotIn('Name: "{localappdata}\\LASS-IQ4"',script)
        self.assertIn("F3B8D2EA-B2BC-4B81-AFC2-80EC97100A56", script)


if __name__ == "__main__":
    unittest.main()
