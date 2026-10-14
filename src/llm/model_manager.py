"""Lifecycle helpers for model assets stored inside the project."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


MANIFEST_FILENAME = "lass-model-manifest.json"
GGUF_FILENAME = 'EXAONE-4.0-1.2B.i1-IQ4_XS.gguf'
REQUIRED_FILES = (GGUF_FILENAME,)


@dataclass(frozen=True)
class ModelStatus:
    model_id: str
    path: Path
    ready: bool
    total_bytes: int
    missing_files: Tuple[str, ...]
    weight_files: int
    manifest_present: bool

    @property
    def size_label(self) -> str:
        value = float(self.total_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024 or unit == "TB":
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024
        return f"{value:.1f} TB"


class LocalModelManager:
    """Inspect and verify a Hugging Face snapshot kept under ``models/``."""

    def __init__(self, model_id: str, model_dir: Path) -> None:
        self.model_id = model_id
        self.model_dir = model_dir.expanduser().resolve()

    @property
    def manifest_path(self) -> Path:
        return self.model_dir / MANIFEST_FILENAME

    def _asset_files(self) -> Iterable[Path]:
        if not self.model_dir.is_dir():
            return ()
        return (
            path
            for path in self.model_dir.rglob("*")
            if path.is_file() and ".cache" not in path.parts
        )

    def status(self) -> ModelStatus:
        missing = [name for name in REQUIRED_FILES if not (self.model_dir / name).is_file() or not (self.model_dir / name).stat().st_size]
        weight_files = tuple(path for path in self.model_dir.glob(GGUF_FILENAME) if path.stat().st_size)
        if not weight_files:
            missing.append(GGUF_FILENAME)
        if weight_files:
            with weight_files[0].open('rb') as handle:
                if handle.read(4) != b'GGUF':
                    missing.append('valid GGUF header')
        total_bytes = sum(path.stat().st_size for path in self._asset_files())
        return ModelStatus(
            model_id=self.model_id,
            path=self.model_dir,
            ready=not missing,
            total_bytes=total_bytes,
            missing_files=tuple(missing),
            weight_files=len(weight_files),
            manifest_present=self.manifest_path.is_file(),
        )

    def require_ready(self) -> ModelStatus:
        status = self.status()
        if not status.ready:
            missing = ", ".join(status.missing_files)
            raise RuntimeError(
                f"EXAONE local model is incomplete at {status.path} "
                f"(missing: {missing}). Check the internet connection and restart LASS to prepare the model."
            )
        return status

    def write_manifest(self, revision: str, files: Dict[str, Dict[str, object]]) -> Path:
        payload = {
            "schema_version": 1,
            "model_id": self.model_id,
            "revision": revision,
            "files": files,
        }
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.manifest_path

    def verify_manifest(self) -> Tuple[bool, Tuple[str, ...]]:
        if not self.manifest_path.is_file():
            return False, (f"missing {MANIFEST_FILENAME}",)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        failures = []
        for relative_name, expected in payload.get("files", {}).items():
            path = self.model_dir / relative_name
            if not path.is_file():
                failures.append(f"missing {relative_name}")
                continue
            expected_size = expected.get("size")
            if expected_size is not None and path.stat().st_size != expected_size:
                failures.append(f"size mismatch: {relative_name}")
                continue
            expected_hash = expected.get("sha256")
            if expected_hash and self.sha256(path) != expected_hash:
                failures.append(f"checksum mismatch: {relative_name}")
        return not failures, tuple(failures)

    @staticmethod
    def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            while chunk := file_handle.read(block_size):
                digest.update(chunk)
        return digest.hexdigest()
