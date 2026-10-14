import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import Settings


class SettingsTests(unittest.TestCase):
    def test_default_llm_budget_is_low_memory(self) -> None:
        settings = Settings()

        self.assertEqual(settings.exaone_max_input_tokens, 2048)
        self.assertFalse(hasattr(settings, "exaone_max_new_tokens"))

    def test_loads_dotenv_and_creates_runtime_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            env_file = root / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"LASS_PROJECT_ROOT={root}",
                        "LASS_WHISPER_MODEL=base",
                        "LASS_STT_TARGET_LUFS=-18",
                        f"LASS_EXAONE_MODEL_DIR={root / 'models' / 'exaone'}",
                        "LASS_EXAONE_MAX_NEW_TOKENS=512",
                        "LASS_CHUNK_SIZE=500",
                        "LASS_CHUNK_OVERLAP=50",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(env_file)
            settings.ensure_directories()

            self.assertEqual(settings.exaone_model_id, "mradermacher/EXAONE-4.0-1.2B-i1-GGUF")
            self.assertFalse(hasattr(settings, "whisper_model"))
            self.assertEqual(settings.stt_target_lufs, -18.0)
            self.assertFalse(hasattr(settings, "exaone_max_new_tokens"))
            self.assertEqual(
                settings.exaone_model_dir, (root / "models" / "exaone").resolve()
            )
            self.assertEqual(settings.chunk_size, 500)
            self.assertTrue(settings.data_dir.is_dir())
            self.assertTrue(settings.exaone_model_dir.is_dir())
            self.assertEqual(
                settings.database_path, (root / "data" / "lass.db").resolve()
            )
            self.assertEqual(
                settings.vector_database_path,
                (root / "data" / "vectors.db").resolve(),
            )

    def test_rejects_unsupported_moonshine_language(self):
        with patch.dict(os.environ, {'LASS_STT_LANGUAGE':'ja'}):
            with self.assertRaisesRegex(ValueError,'LASS_STT_LANGUAGE'):
                Settings.from_env()

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
