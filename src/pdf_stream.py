"""Tokenized localhost HTTP Range server for memory-safe PDF viewing."""

from __future__ import annotations

import hashlib
import secrets
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock, Thread
from typing import Dict, Optional, Tuple
from urllib.parse import urlsplit


def parse_byte_range(value: str, file_size: int) -> Optional[Tuple[int, int]]:
    """Parse one RFC 7233 byte range as an inclusive start/end pair."""
    if not value:
        return None
    if file_size < 0 or not value.startswith("bytes=") or "," in value:
        raise ValueError("Unsupported byte range")
    start_text, separator, end_text = value[6:].partition("-")
    if not separator:
        raise ValueError("Invalid byte range")
    if not start_text:
        suffix = int(end_text)
        if suffix <= 0:
            raise ValueError("Invalid suffix byte range")
        return max(0, file_size - suffix), max(0, file_size - 1)
    start = int(start_text)
    end = int(end_text) if end_text else file_size - 1
    if start < 0 or start >= file_size or end < start:
        raise ValueError("Byte range is outside the file")
    return start, min(end, file_size - 1)


class LocalMediaServer:
    """Stream registered local media via byte ranges without buffering whole files."""

    extensions = {".pdf", ".jpg", ".png", ".svg", ".aac", ".flac", ".m4a", ".mkv", ".mov", ".mp3", ".mp4", ".ogg", ".wav", ".webm"}

    def __init__(self) -> None:
        self._secret = secrets.token_bytes(32)
        self._paths: Dict[str, Path] = {}
        self._lock = RLock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self) -> None:  # noqa: N802
                self._serve(send_body=False)

            def do_GET(self) -> None:  # noqa: N802
                self._serve(send_body=True)

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self._cors_headers()
                self.end_headers()

            def _cors_headers(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Range")
                self.send_header(
                    "Access-Control-Expose-Headers", "Content-Range, Accept-Ranges"
                )

            def _serve(self, *, send_body: bool) -> None:
                token = Path(urlsplit(self.path).path).stem
                source = owner._get(token)
                if source is None or not source.is_file() or source.suffix.lower() not in owner.extensions:
                    self.send_error(404)
                    return
                file_size = source.stat().st_size
                try:
                    selected = parse_byte_range(self.headers.get("Range", ""), file_size)
                except (TypeError, ValueError):
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return
                start, end = selected if selected is not None else (0, file_size - 1)
                self.send_response(206 if selected is not None else 200)
                self.send_header("Content-Type", mimetypes.guess_type(source.name)[0] or "application/octet-stream")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(max(0, end - start + 1)))
                self.send_header("Cache-Control", "private, max-age=3600" if source.suffix.lower() in {".jpg", ".png", ".svg"} else "private, no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if selected is not None:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self._cors_headers()
                self.end_headers()
                if not send_body or file_size == 0:
                    return
                remaining = end - start + 1
                try:
                    with source.open("rb") as file_handle:
                        file_handle.seek(start)
                        while remaining > 0:
                            chunk = file_handle.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = Thread(
            target=self._server.serve_forever,
            name="lass-pdf-stream",
            daemon=True,
        )
        self._thread.start()

    def _get(self, token: str) -> Optional[Path]:
        with self._lock:
            return self._paths.get(token)

    def register(self, path: Path) -> str:
        source = path.expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in self.extensions:
            raise FileNotFoundError(f"Media file not found or unsupported: {source}")
        token = hashlib.sha256(self._secret + str(source).encode("utf-8")).hexdigest()
        with self._lock:
            self._paths[token] = source
        host, port = self._server.server_address
        return f"http://{host}:{port}/{token}{source.suffix.lower()}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class PDFStreamServer(LocalMediaServer):
    extensions = {".pdf"}
