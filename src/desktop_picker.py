"""Authenticated loopback bridge from Streamlit to the OS file chooser."""

import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlparse


class FilePickerServer:
    def __init__(self, window):
        token = secrets.token_urlsafe(32)

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = urlparse(self.path)
                if request.path != "/" + token:
                    self.send_error(404)
                    return
                try:
                    kind = parse_qs(request.query).get("kind", ["all"])[0]
                    filters = {
                        "document": ("PDF · PowerPoint (*.pdf;*.ppt;*.pptx)",),
                        "audio": ("녹음 · 영상 (*.wav;*.mp3;*.m4a;*.mp4;*.aac;*.flac;*.ogg;*.webm;*.mkv;*.mov;*.avi;*.mpeg;*.mpg)",),
                    }
                    selected = window.create_file_dialog(
                        allow_multiple=False,
                        file_types=filters.get(
                            kind,
                            ("노트 자료 (*.pdf;*.ppt;*.pptx;*.wav;*.mp3;*.m4a;*.mp4;*.aac;*.flac;*.ogg;*.webm;*.mkv;*.mov;*.avi;*.mpeg;*.mpg)",),
                        ),
                    )
                    payload = json.dumps({"path": selected[0] if selected else ""}).encode()
                    self.send_response(200)
                except Exception as exc:
                    payload = json.dumps({"error": str(exc)}).encode()
                    self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/{token}"
        Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
