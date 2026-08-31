"""Audio preparation helpers backed by the local FFmpeg executable."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


SUPPORTED_MEDIA_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".webm",
}


def validate_media_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Media file not found: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
        raise ValueError(f"Unsupported media format: {resolved.suffix or '(none)'}")
    return resolved


def ensure_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("FFmpeg is not installed or is not available on PATH")
    return executable


def extract_audio(source: Path, destination: Path) -> Path:
    """Convert a media file to mono 16 kHz WAV for speech recognition."""
    source = validate_media_file(source)
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ensure_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or "unknown FFmpeg error"
        raise RuntimeError(f"Could not extract audio: {message}") from exc
    return destination
