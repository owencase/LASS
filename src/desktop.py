"""Window-only desktop host. AI/Streamlit run in one owned background process."""

from __future__ import annotations

import argparse
import hashlib
import html
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from urllib.request import urlopen


def find_project_root() -> Path:
    candidates = [Path(sys.executable).resolve().parent] if getattr(sys, "frozen", False) else [Path(__file__).resolve().parents[1]]
    for candidate in [*candidates, *candidates[0].parents]:
        if (candidate / "src/ui/app.py").is_file() and (candidate / "requirements.txt").is_file():
            return candidate
    raise RuntimeError("LASS 실행 파일을 src, models 폴더와 함께 보관해 주세요.")


def child_options() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {"start_new_session": True}


def runtime_python(root: Path) -> Path:
    return root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def server_command(root: Path, port: int) -> list[str]:
    return [str(runtime_python(root)), "-m", "streamlit", "run", str(root / "src/ui/app.py"),
            "--server.address", "127.0.0.1", "--server.port", str(port),
            "--server.headless", "true", "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false"]


def page(message: str, *, error: bool = False) -> str:
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8"><style>
    :root {{ color-scheme:light dark }} body {{margin:0; background:light-dark(#f3f5f8,#0b0d10);
    color:light-dark(#18212f,#f5f7fa); font:16px -apple-system,Segoe UI,sans-serif;
    min-height:100vh; display:grid; place-items:center }} main {{max-width:620px;padding:32px}}
    h1 {{letter-spacing:.15em}} p {{line-height:1.8;white-space:pre-wrap}}</style>
    <main><h1>LASS</h1><p>{html.escape(message)}</p><small>{'창을 닫고 다시 실행할 수 있습니다.' if error else '잠시만 기다려 주세요.'}</small></main></html>'''


class DesktopHost:
    def __init__(self, root: Path):
        self.root = root
        self.process = None
        self.window = None
        self.stopped = threading.Event()
        self.process_lock = threading.Lock()
        self.log_path = root / "data/logs/desktop.log"

    def _spawn(self, command, log):
        with self.process_lock:
            if self.stopped.is_set():
                raise RuntimeError("실행이 취소되었습니다.")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(self.root)
            environment["PYTHONUNBUFFERED"] = "1"
            # Only unsupported MPS operators fall back; the model remains on GPU.
            environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            self.process = subprocess.Popen(command, cwd=str(self.root), env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **child_options())
            return self.process

    def prepare_runtime(self, log):
        python = runtime_python(self.root)
        if not python.is_file():
            if not getattr(sys, "frozen", False):
                bootstrap = [sys.executable]
            elif sys.platform == "win32" and shutil.which("py"):
                bootstrap = [shutil.which("py"), "-3"]
            elif shutil.which("python3"):
                bootstrap = [shutil.which("python3")]
            else:
                raise RuntimeError("Python 3.10 이상을 설치한 뒤 다시 실행해 주세요.")
            if self._spawn([*bootstrap, "-m", "venv", str(self.root / ".venv")], log).wait() != 0:
                raise RuntimeError("실행 환경을 생성하지 못했습니다.")
        requirements = self.root / "requirements.txt"
        signature = hashlib.sha256(requirements.read_bytes()).hexdigest()
        stamp = self.root / ".venv/.lass-desktop-requirements"
        if not stamp.is_file() or stamp.read_text().strip() != signature:
            self.window.load_html(page("처음 실행에 필요한 패키지를 준비하고 있습니다.\n준비가 끝나면 노트 화면이 자동으로 열립니다."))
            command = [str(python), "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements)]
            if self._spawn(command, log).wait() != 0:
                raise RuntimeError("실행 환경 설치에 실패했습니다. 인터넷 연결과 로그를 확인해 주세요.")
            stamp.write_text(signature)

    def start_server(self):
        from src.desktop_picker import FilePickerServer
        picker = FilePickerServer(self.window)
        os.environ["LASS_DESKTOP_PICKER"] = picker.url
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep a single previous launch log; no unbounded logging growth.
        if self.log_path.is_file():
            self.log_path.replace(self.log_path.with_suffix(".previous.log"))
        try:
            with self.log_path.open("w", encoding="utf-8") as log:
                self.prepare_runtime(log)
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    port = reservation.getsockname()[1]
                process = self._spawn(server_command(self.root, port), log)
                url = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 90
                while not self.stopped.wait(.25):
                    if process.poll() is not None:
                        raise RuntimeError("노트 화면 서버가 종료되었습니다.")
                    try:
                        with urlopen(url + "/_stcore/health", timeout=1) as response:
                            ready = response.status == 200
                    except OSError:
                        ready = False
                    if ready:
                        self.window.load_url(url)
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("노트 화면 준비 시간이 초과되었습니다.")
                while not self.stopped.wait(1):
                    if process.poll() is not None:
                        raise RuntimeError("백그라운드 서버가 종료되었습니다. 다시 실행해 주세요.")
        except Exception as exc:
            if not self.stopped.is_set():
                self.window.load_html(page(f"{exc}\n\n로그: {self.log_path}", error=True))
            self._stop_process()
        finally:
            picker.close()

    def _stop_process(self):
        with self.process_lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)

    def close(self):
        self.stopped.set()
        self._stop_process()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve() if args.root else find_project_root()
    import webview
    host = DesktopHost(root)
    # A kernel-released local socket prevents duplicate AI servers and focuses
    # the existing window on subsequent icon clicks. No stale PID files.
    instance_port = 42000 + int(hashlib.sha256(str(root).encode()).hexdigest()[:6], 16) % 15000
    instance = socket.socket()
    try:
        instance.bind(("127.0.0.1", instance_port))
    except OSError:
        try:
            with socket.create_connection(("127.0.0.1", instance_port), timeout=1) as existing:
                existing.sendall(b"LASS-FOCUS")
        finally:
            instance.close()
        return
    instance.listen(2)
    instance.settimeout(1)
    host.window = webview.create_window("LASS", html=page("노트 작업공간을 준비하고 있습니다."),
                                        width=1320, height=900, min_size=(900, 620))
    host.window.events.closed += host.close
    def focus_listener():
        while not host.stopped.is_set():
            try:
                connection, _ = instance.accept()
                with connection:
                    connection.settimeout(1)
                    if connection.recv(32) == b"LASS-FOCUS":
                        host.window.restore()
                        host.window.show()
            except OSError:
                continue
    threading.Thread(target=focus_listener, daemon=True).start()
    try:
        webview.start(host.start_server, private_mode=False)
    finally:
        host.close()
        instance.close()


if __name__ == "__main__":
    main()
