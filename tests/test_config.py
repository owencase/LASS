import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import Settings


class SettingsTests(unittest.TestCase):
    def test_loads_dotenv_and_creates_runtime_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            env_file = root / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"LASS_PROJECT_ROOT={root}",
                        "LASS_WHISPER_MODEL=base",
                        "LASS_CHUNK_SIZE=500",
                        "LASS_CHUNK_OVERLAP=50",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(env_file)
            settings.ensure_directories()

            self.assertEqual(settings.whisper_model, "base")
            self.assertEqual(settings.chunk_size, 500)
            self.assertTrue(settings.input_dir.is_dir())
            self.assertTrue(settings.output_dir.is_dir())
            self.assertTrue(settings.vector_db_dir.is_dir())

    def test_rejects_overlap_larger_than_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_file = Path(temporary_directory) / ".env"
            env_file.write_text(
                "LASS_CHUNK_SIZE=100\nLASS_CHUNK_OVERLAP=100\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    Settings.from_env(env_file)


if __name__ == "__main__":
    unittest.main()
