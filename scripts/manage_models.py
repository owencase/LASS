#!/usr/bin/env python3
"""Download, inspect, and verify project-local model assets."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings  # noqa: E402
from src.llm.model_manager import MANIFEST_FILENAME, LocalModelManager  # noqa: E402


DOWNLOAD_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.safetensors",
    "LICENSE",
    "README.md",
)


def _manager(runtime: str) -> LocalModelManager:
    settings = Settings.from_env()
    settings.ensure_directories()
    if runtime == "mlx":
        return LocalModelManager(
            settings.exaone_mlx_model_id, settings.exaone_mlx_model_dir
        )
    return LocalModelManager(settings.exaone_model_id, settings.exaone_model_dir)


def _manifest_files(manager: LocalModelManager) -> Dict[str, Dict[str, object]]:
    entries: Dict[str, Dict[str, object]] = {}
    for path in sorted(manager.model_dir.rglob("*")):
        if (
            not path.is_file()
            or ".cache" in path.parts
            or path.name == MANIFEST_FILENAME
            or path.name == ".gitkeep"
        ):
            continue
        relative_name = path.relative_to(manager.model_dir).as_posix()
        print(f"hashing {relative_name} ...", flush=True)
        entries[relative_name] = {
            "size": path.stat().st_size,
            "sha256": manager.sha256(path),
        }
    return entries


def show_status(manager: LocalModelManager) -> int:
    status = manager.status()
    payload = {
        "model_id": status.model_id,
        "path": str(status.path),
        "ready": status.ready,
        "size": status.size_label,
        "weight_files": status.weight_files,
        "manifest": status.manifest_present,
        "missing": list(status.missing_files),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status.ready else 1


def download(manager: LocalModelManager, revision: str) -> int:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required. Install with `pip install -r requirements.txt`."
        ) from exc

    manager.model_dir.mkdir(parents=True, exist_ok=True)
    model_info = HfApi().model_info(manager.model_id, revision=revision)
    resolved_revision = model_info.sha or revision
    print(f"downloading {manager.model_id}@{resolved_revision}")
    print(f"destination: {manager.model_dir}")
    snapshot_download(
        repo_id=manager.model_id,
        revision=resolved_revision,
        local_dir=manager.model_dir,
        allow_patterns=list(DOWNLOAD_PATTERNS),
    )
    manager.require_ready()
    manifest = manager.write_manifest(resolved_revision, _manifest_files(manager))
    print(f"manifest: {manifest}")
    return show_status(manager)


def verify(manager: LocalModelManager) -> int:
    manager.require_ready()
    valid, failures = manager.verify_manifest()
    if valid:
        print("OK: every managed model file matches its SHA-256 checksum.")
        return 0
    print("Model verification failed:")
    for failure in failures:
        print(f"- {failure}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_runtime = (
        "mlx" if platform.system() == "Darwin" and platform.machine() == "arm64" else "transformers"
    )
    parser.add_argument(
        "--runtime",
        choices=("mlx", "transformers"),
        default=default_runtime,
        help="model runtime to manage (Apple Silicon defaults to mlx)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="show local EXAONE asset status")
    download_parser = subparsers.add_parser("download", help="download the model snapshot")
    download_parser.add_argument("--revision", default="main")
    subparsers.add_parser("verify", help="verify sizes and SHA-256 checksums")
    args = parser.parse_args()

    manager = _manager(args.runtime)
    if args.command == "download":
        return download(manager, args.revision)
    if args.command == "verify":
        return verify(manager)
    return show_status(manager)


if __name__ == "__main__":
    raise SystemExit(main())
