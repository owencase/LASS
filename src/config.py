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


@dataclass(frozen=True)
class Settings:
    """Runtime settings for local models and storage."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    prompt_dir: Path = PROJECT_ROOT / "prompts"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    embedding_model: str = "BAAI/bge-m3"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    chunk_size: int = 1200
    chunk_overlap: int = 150

    @property
    def input_dir(self) -> Path:
        return self.data_dir / "inputs"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def vector_db_dir(self) -> Path:
        return self.data_dir / "vector_db"

    def ensure_directories(self) -> None:
        for path in (self.input_dir, self.output_dir, self.vector_db_dir):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        """Build settings, giving process variables priority over `.env`."""
        dotenv = _read_dotenv(env_file or PROJECT_ROOT / ".env")

        def get(name: str, default: str) -> str:
            return os.environ.get(name, dotenv.get(name, default))

        project_root = Path(
            get("LASS_PROJECT_ROOT", str(PROJECT_ROOT))
        ).expanduser().resolve()
        data_dir = Path(
            get("LASS_DATA_DIR", str(project_root / "data"))
        ).expanduser().resolve()
        prompt_dir = Path(
            get("LASS_PROMPT_DIR", str(project_root / "prompts"))
        ).expanduser().resolve()

        settings = cls(
            project_root=project_root,
            data_dir=data_dir,
            prompt_dir=prompt_dir,
            whisper_model=get("LASS_WHISPER_MODEL", "small"),
            whisper_device=get("LASS_WHISPER_DEVICE", "auto"),
            whisper_compute_type=get("LASS_WHISPER_COMPUTE_TYPE", "int8"),
            embedding_model=get("LASS_EMBEDDING_MODEL", "BAAI/bge-m3"),
            ollama_base_url=get("LASS_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=get("LASS_OLLAMA_MODEL", "llama3.1:8b"),
            chunk_size=_as_int(get("LASS_CHUNK_SIZE", "1200"), "LASS_CHUNK_SIZE"),
            chunk_overlap=_as_int(get("LASS_CHUNK_OVERLAP", "150"), "LASS_CHUNK_OVERLAP"),
        )
        if settings.chunk_size <= 0:
            raise ValueError("LASS_CHUNK_SIZE must be greater than zero")
        if settings.chunk_overlap < 0 or settings.chunk_overlap >= settings.chunk_size:
            raise ValueError("LASS_CHUNK_OVERLAP must be between 0 and chunk size - 1")
        return settings
