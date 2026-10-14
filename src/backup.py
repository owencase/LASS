"""Timestamped source/configuration snapshots before patches and updates."""
import os
import zipfile
from datetime import datetime
from pathlib import Path


def backup_sources(root, destination=None):
    root = Path(root).resolve()
    documents = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Documents"
    if os.name == "nt":
        import ctypes
        buffer = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buffer) == 0:
            documents = Path(buffer.value)
    target = Path(destination or os.environ.get("LASS_BACKUP_DIR") or documents / "Backups")
    target.mkdir(parents=True, exist_ok=True)
    archive = target / f"LASS-before-patch-{datetime.now():%Y%m%d-%H%M%S-%f}.zip"
    paths = []
    for directory in ("src", "prompts", "scripts", ".streamlit", "tests"):
        paths.extend((root / directory).rglob("*"))
    paths.extend(root / name for name in (".env", ".env.example", "pyproject.toml", "README.md",
        "requirements.txt", "requirements-cuda.txt", "LASS.bat", "LASS.pyw"))
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as output:
        for path in paths:
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                output.write(path, path.relative_to(root))
    with zipfile.ZipFile(archive) as saved:
        if saved.testzip() is not None:
            raise RuntimeError("소스 백업 무결성 검사에 실패했습니다.")
    return archive
