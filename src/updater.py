"""Opt-in GitHub release updates with verified installers and data snapshots."""

import json
import re
import shutil
import sqlite3
import subprocess
from contextlib import closing
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src import __version__
from src.model_assets import download_file

REPOSITORY = "owencase/LASS"
RELEASES_URL = f"https://github.com/{REPOSITORY}/releases"


def version_tuple(value):
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ValueError("정식 버전 번호가 올바르지 않습니다.")
    return tuple(map(int, match.groups()))


def release_url(url):
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.netloc != "github.com"
            or not parsed.path.startswith(f"/{REPOSITORY}/releases/download/")):
        raise ValueError("공식 LASS 릴리스의 파일만 설치할 수 있습니다.")
    return url


def check_update():
    request = Request(f"https://api.github.com/repos/{REPOSITORY}/releases/latest",
                      headers={"Accept": "application/vnd.github+json", "User-Agent": "LASS-Updater"})
    try:
        with urlopen(request, timeout=15) as response:
            release = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if "moonshine" not in release.get("name", "").lower() or release.get("draft") or release.get("prerelease"):
        return None
    version = release["tag_name"].removeprefix("v")
    if version_tuple(version) <= version_tuple(__version__):
        return None
    filename = f"LASS-Setup-{version}.exe"
    assets = {asset["name"]: asset for asset in release.get("assets", [])}
    if filename not in assets or filename + ".sha256" not in assets:
        raise ValueError("새 릴리스의 설치 파일 또는 SHA-256 검증 파일이 아직 준비되지 않았습니다.")
    installer = assets[filename]
    if not isinstance(installer.get("size"), int) or installer["size"] <= 0:
        raise ValueError("설치 파일 크기가 올바르지 않습니다.")
    return {"version": version, "filename": filename, "size": installer["size"],
            "url": release_url(installer["browser_download_url"]),
            "checksum_url": release_url(assets[filename + ".sha256"]["browser_download_url"])}


def download_update(release, directory, progress=None):
    with urlopen(release_url(release["checksum_url"]), timeout=15) as response:
        checksum = response.read(1024).decode("ascii").strip().split()[0]
    if not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
        raise ValueError("SHA-256 검증 파일이 올바르지 않습니다.")
    filename = f"LASS-Setup-{release['version']}.exe"
    version_tuple(release["version"])
    path = Path(directory) / filename
    download_file(release_url(release["url"]), path, release["size"], checksum.lower(), progress)
    return path


def backup_data(data_dir):
    """SQLite's backup API includes committed WAL data; raw DB copies do not."""
    data_dir = Path(data_dir).resolve()
    backup = data_dir.parent / "backups" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup.mkdir(parents=True)
    try:
        for source in data_dir.rglob("*"):
            relative = source.relative_to(data_dir)
            if (not source.is_file() or source.is_symlink()
                    or relative.parts[0] in {"logs", "preview"}
                    or source.name.endswith(("-wal", "-shm", "-journal"))):
                continue
            target = backup / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".db":
                with closing(sqlite3.connect(str(source), timeout=30)) as original, closing(sqlite3.connect(str(target))) as snapshot:
                    original.backup(snapshot)
                    if snapshot.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise RuntimeError("데이터 백업 무결성 검사에 실패했습니다.")
            else:
                shutil.copy2(source, target)
        (backup / "backup-info.json").write_text(json.dumps(
            {"app_version": __version__, "data_dir": str(data_dir)}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # Keep incomplete snapshots for diagnosis, but never proceed to install.
        (backup / "INCOMPLETE").touch()
        raise
    return backup


def launch_installer(path):
    # Inno Setup handles closing the installed app and offers restart after setup.
    return subprocess.Popen([str(Path(path).resolve()), "/CLOSEAPPLICATIONS"],
                            creationflags=subprocess.CREATE_NO_WINDOW)
