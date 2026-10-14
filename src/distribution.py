"""Check that the external app sources and assets match the build input."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def asset_paths(root):
    for directory in ("src", "prompts"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                yield path.relative_to(root)
    yield Path(".streamlit/config.toml")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(source, bundle):
    files = {}
    for relative in asset_paths(source):
        expected = digest(source / relative)
        if not (bundle / relative).is_file() or digest(bundle / relative) != expected:
            raise RuntimeError(f"Bundled resource missing or stale: {relative}")
        files[relative.as_posix()] = expected
    manifest = {"built_at": datetime.now(timezone.utc).isoformat(), "files": files}
    (bundle / "build-info.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def verify_bundle(bundle):
    manifest = json.loads((bundle / "build-info.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        if not (bundle / relative).is_file() or digest(bundle / relative) != expected:
            raise RuntimeError(f"Bundled resource missing or stale: {relative}")
    return manifest["built_at"]
