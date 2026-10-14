import json
import tempfile
import unittest
from pathlib import Path

from src.llm.model_manager import LocalModelManager, GGUF_FILENAME


class LocalModelManagerTests(unittest.TestCase):
    def test_reports_missing_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = LocalModelManager("test/model", Path(temporary_directory))
            status = manager.status()

            self.assertFalse(status.ready)
            self.assertIn(GGUF_FILENAME, status.missing_files)
            self.assertEqual(status.weight_files, 0)

    def test_verifies_managed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            files = {
                "config.json": b"{}",
                "tokenizer.json": b"{}",
                "tokenizer_config.json": b"{}",
                GGUF_FILENAME: b"GGUFweights",
            }
            for name, content in files.items():
                (model_dir / name).write_bytes(content)

            manager = LocalModelManager("test/model", model_dir)
            manifest_files = {
                name: {"size": len(content), "sha256": manager.sha256(model_dir / name)}
                for name, content in files.items()
            }
            manager.write_manifest("test-revision", manifest_files)

            self.assertTrue(manager.status().ready)
            self.assertEqual(manager.verify_manifest(), (True, ()))

            payload = json.loads(manager.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_id"], "test/model")


if __name__ == "__main__":
    unittest.main()
