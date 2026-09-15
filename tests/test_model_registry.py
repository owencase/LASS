import json
import tempfile
import unittest
from pathlib import Path

from src.config import Settings
from src.model_registry import ModelRegistry


class ModelRegistryTests(unittest.TestCase):
    def test_add_select_remove_profiles_persistently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                project_root=root,
                data_dir=root / "data",
                prompt_dir=root / "prompts",
                exaone_model_dir=root / "models" / "exaone",
                exaone_mlx_model_dir=root / "models" / "exaone-mlx",
            )
            registry = ModelRegistry.from_settings(settings)

            custom = registry.add(
                kind="stt",
                name="빠른 Whisper",
                identifier="large-v3-turbo",
                compute_type="int8",
            )
            registry.set_active("stt", custom.key)

            reloaded = ModelRegistry.from_settings(settings)
            self.assertEqual(reloaded.active("stt").identifier, "large-v3-turbo")
            removed = reloaded.remove(custom.key)
            self.assertEqual(removed.name, "빠른 Whisper")
            self.assertEqual(reloaded.active("stt").key, "sensevoice-small")

    def test_sensevoice_is_default_and_whisper_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                data_dir=root, exaone_mlx_model_dir=root / "missing-mlx"
            )
            registry = ModelRegistry.from_settings(settings)

            profiles = registry.profiles("stt")

            self.assertEqual(registry.active("stt").key, "sensevoice-small")
            self.assertEqual(
                {profile.key for profile in profiles},
                {
                    "sensevoice-small",
                    "bigvox-exaone4-audio",
                    "default-stt",
                },
            )
            self.assertEqual(registry.active("stt").runtime_backend, "sensevoice")

    def test_migration_adds_sensevoice_without_changing_existing_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                data_dir=root, exaone_mlx_model_dir=root / "missing-mlx"
            )
            settings.model_registry_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "active": {"stt": "default-stt", "llm": "default-llm"},
                        "profiles": [
                            {
                                "key": "default-stt",
                                "kind": "stt",
                                "name": "Whisper · large-v3",
                                "identifier": "large-v3",
                            },
                            {
                                "key": "default-llm",
                                "kind": "llm",
                                "name": "EXAONE",
                                "identifier": "exaone",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            registry = ModelRegistry.from_settings(settings)

            self.assertEqual(registry.active("stt").key, "default-stt")
            self.assertIn(
                "sensevoice-small",
                {profile.key for profile in registry.profiles("stt")},
            )
            migrated = json.loads(
                settings.model_registry_path.read_text(encoding="utf-8")
            )
            self.assertEqual(migrated["schema_version"], 4)
            self.assertIn(
                "bigvox-exaone4-audio",
                {profile["key"] for profile in migrated["profiles"]},
            )

    def test_does_not_remove_the_last_profile_of_a_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                data_dir=root, exaone_mlx_model_dir=root / "missing-mlx"
            )
            registry = ModelRegistry.from_settings(settings)

            llm_profiles = registry.profiles("llm")
            for profile in llm_profiles[1:]:
                registry.remove(profile.key)
            with self.assertRaises(ValueError):
                registry.remove(llm_profiles[0].key)

    def test_ready_mlx_profile_becomes_apple_silicon_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_dir = root / "models" / "exaone-mlx"
            model_dir.mkdir(parents=True)
            for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
                (model_dir / name).write_text("{}", encoding="utf-8")
            settings = Settings(data_dir=root / "data", exaone_mlx_model_dir=model_dir)

            registry = ModelRegistry.from_settings(settings)

            mlx_profiles = [p for p in registry.profiles("llm") if p.runtime_backend == "mlx"]
            if mlx_profiles:
                self.assertEqual(registry.active("llm").key, "default-llm-mlx")
                self.assertEqual(registry.active("llm").compute_type, "int4")

    def test_updates_existing_profile_device_and_rejects_whisper_mps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                data_dir=root, exaone_mlx_model_dir=root / "missing-mlx"
            )
            registry = ModelRegistry.from_settings(settings)

            updated = registry.update_device("sensevoice-small", "mps")

            self.assertEqual(updated.device, "mps")
            self.assertEqual(registry.active("stt").device, "mps")
            with self.assertRaisesRegex(ValueError, "Apple GPU"):
                registry.update_device("default-stt", "mps")


if __name__ == "__main__":
    unittest.main()
