"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_dotenv(path: Path) -> Dict[str, str]:
    """Read a small, dependency-free subset of the dotenv format."""
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _as_int(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer: {value!r}") from exc


def _as_float(value: str, name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number: {value!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Runtime settings for local models and storage."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    prompt_dir: Path = PROJECT_ROOT / "prompts"
    stt_model_dir: Path = PROJECT_ROOT / "models" / "moonshine"
    stt_language: str = "ko"
    stt_target_lufs: float = -16.0
    exaone_model_id: str = "mradermacher/EXAONE-4.0-1.2B-i1-GGUF"
    exaone_model_dir: Path = PROJECT_ROOT / "models" / "exaone-4.0-1.2b-iq4-xs"
    exaone_device: str = "auto"
    exaone_max_input_tokens: int = 2048
    exaone_temperature: float = 0.1
    chunk_size: int = 1200
    chunk_overlap: int = 150

    @property
    def database_path(self) -> Path:
        return self.data_dir / "lass.db"

    @property
    def vector_database_path(self) -> Path:
        return self.data_dir / "vectors.db"

    @property
    def document_dir(self) -> Path:
        return self.data_dir / "documents"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.exaone_model_dir.mkdir(parents=True, exist_ok=True)
        self.stt_model_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        """Build settings, giving process variables priority over `.env`."""
        dotenv = _read_dotenv(env_file or PROJECT_ROOT / ".env")

        def get(name: str, default: str) -> str:
            return os.environ.get(name, dotenv.get(name, default))

        project_root = Path(
            get("LASS_PROJECT_ROOT", str(PROJECT_ROOT))
        ).expanduser().resolve()

        def get_path(name: str, default: Path) -> Path:
            path = Path(get(name, str(default))).expanduser()
            if not path.is_absolute():
                path = project_root / path
            return path.resolve()

        data_dir = get_path("LASS_DATA_DIR", project_root / "data")
        prompt_dir = get_path("LASS_PROMPT_DIR", project_root / "prompts")
        exaone_model_dir = get_path(
            "LASS_EXAONE_MODEL_DIR", project_root / "models" / "exaone-4.0-1.2b-iq4-xs"
        )
        settings = cls(
            project_root=project_root,
            data_dir=data_dir,
            prompt_dir=prompt_dir,
            stt_model_dir=get_path("LASS_STT_MODEL_DIR", project_root / "models" / "moonshine"),
            stt_language=get("LASS_STT_LANGUAGE", "ko"),
            stt_target_lufs=_as_float(
                get("LASS_STT_TARGET_LUFS", "-16"), "LASS_STT_TARGET_LUFS"
            ),
            exaone_model_dir=exaone_model_dir,
            exaone_device=get("LASS_EXAONE_DEVICE", "auto"),
            exaone_temperature=_as_float(
                get("LASS_EXAONE_TEMPERATURE", "0.1"),
                "LASS_EXAONE_TEMPERATURE",
            ),
            exaone_max_input_tokens=_as_int(
                get("LASS_EXAONE_MAX_INPUT_TOKENS", "2048"), "LASS_EXAONE_MAX_INPUT_TOKENS"
            ),
            chunk_size=_as_int(get("LASS_CHUNK_SIZE", "1200"), "LASS_CHUNK_SIZE"),
            chunk_overlap=_as_int(get("LASS_CHUNK_OVERLAP", "150"), "LASS_CHUNK_OVERLAP"),
        )
        if settings.stt_language not in {"ko", "en"}:
            raise ValueError("LASS_STT_LANGUAGE must be ko or en")
        if settings.chunk_size <= 0:
            raise ValueError("LASS_CHUNK_SIZE must be greater than zero")
        if settings.chunk_overlap < 0 or settings.chunk_overlap >= settings.chunk_size:
            raise ValueError("LASS_CHUNK_OVERLAP must be between 0 and chunk size - 1")
        if not 512 <= settings.exaone_max_input_tokens <= 32768:
            raise ValueError("LASS_EXAONE_MAX_INPUT_TOKENS must be between 512 and 32768")
        if not -30.0 <= settings.stt_target_lufs <= -5.0:
            raise ValueError("LASS_STT_TARGET_LUFS must be between -30 and -5")
        if not 0 <= settings.exaone_temperature <= 2:
            raise ValueError("LASS_EXAONE_TEMPERATURE must be between 0 and 2")
        return settings
