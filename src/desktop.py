"""Window-only desktop host. AI/Streamlit run in one owned background process."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback
from urllib.request import urlopen


def find_project_root() -> Path:
    return Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]


def configure_paths(root: Path) -> None:
    import tempfile
    temporary = root / 'data/tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(temporary)
    tempfile.tempdir = str(temporary)
    bundle = find_project_root()
    os.environ["LASS_PROJECT_ROOT"] = str(bundle)
    os.environ["LASS_PROMPT_DIR"] = str(bundle / "prompts")
    os.environ["LASS_DATA_DIR"] = str(root / "data")
    os.environ.setdefault("LASS_STT_MODEL_DIR", str(root / "models/moonshine"))
    os.environ.setdefault("LASS_EXAONE_MODEL_DIR", str(root / "models/exaone-4.0-1.2b-iq4-xs"))
    os.environ.setdefault("HF_HOME", str(root / "models/.cache"))


def child_options() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW}


def prepare_stdio(root: Path, mode: str) -> None:
    # Windowed EXEs have no console streams. Uvicorn/tqdm need real file objects.
    if sys.stdout is None or sys.stderr is None:
        log_dir = root / "data/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stream = (log_dir / f"stdio-{mode}.log").open("w", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = stream
        if sys.stderr is None:
            sys.stderr = stream
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r")


def child_command(root: Path, *arguments: str) -> list[str]:
    entry = [] if getattr(sys, "frozen", False) else ["-m", "src.desktop"]
    return [sys.executable, *entry, "--root", str(root), *arguments]


def server_command(root: Path, port: int) -> list[str]:
    return child_command(root, "--serve", str(port))


def serve(port: int):
    from streamlit.web import bootstrap
    options = {
        "global.developmentMode": False,
        "server.address": "127.0.0.1", "server.port": port,
        "server.headless": True, "server.fileWatcherType": "none",
        "server.maxUploadSize": 2048, "server.maxMessageSize": 32,
        "browser.gatherUsageStats": False,
    }
    bootstrap.load_config_options(options)
    bootstrap.run(str(find_project_root() / "src/ui/app.py"), False, [], options)


def self_test(root: Path):
    if getattr(sys, "frozen", False):
        from src.distribution import verify_bundle
        verify_bundle(find_project_root())
    from src.llm.native_runtime import prepare_native
    prepare_native()
    import llama_cpp, moonshine_voice, langchain_core, streamlit, webview
    import av, pypdf, pymupdf, pyloudnorm, onnxruntime, rapidocr
    from src.llm.langchain_model import EXAONEChatModel
    from src.stt.moonshine import MoonshineTranscriber
    from src.ocr import OCRReader
    from src.llm.prompt_manager import PromptManager
    from src.config import Settings
    settings = Settings.from_env()
    assert (find_project_root() / "src/ui/layout_component/frontend/index.html").is_file()
    assert PromptManager(settings.prompt_dir).load("lecture_detail").name == "강의 상세 노트"
    from src.llm.model_manager import LocalModelManager
    if LocalModelManager(settings.exaone_model_id, settings.exaone_model_dir).status().ready:
        from src.llm.model_loader import LocalEXAONEClient
        from langchain_core.messages import SystemMessage, HumanMessage
        client = LocalEXAONEClient(settings.exaone_model_dir, device=settings.exaone_device)
        try:
            reply = EXAONEChatModel(client=client).invoke([
                SystemMessage(content="Reply briefly in English."),
                HumanMessage(content="What is 1 + 1? Reply with one digit."),
            ]).content
            assert reply.strip()
            (root / "llm-test.txt").write_text(reply, encoding="utf-8")
        finally:
            client.unload()
    audio = root / "test.wav"
    if audio.is_file():
        transcriber = MoonshineTranscriber(settings.stt_model_dir)
        try:
            result = transcriber.transcribe(audio)
            (root / "transcription-test.txt").write_text(result.text, encoding="utf-8")
        finally:
            transcriber.unload()
    (root / "self-test.txt").write_text("OK: bundled Python, models runtime, UI and prompt resources", encoding="utf-8")


def page(message: str, *, error: bool = False) -> str:
    loading = '' if error else '<div class="track" role="progressbar" aria-label="앱 준비 중"><div></div></div>'
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8"><style>
    :root {{ color-scheme:light dark }} body {{margin:0; background:#0b0d10;
    color:#f5f7fa; font:16px Segoe UI,sans-serif;
    min-height:100vh; display:grid; place-items:center }} main {{max-width:620px;padding:32px}}
    h1 {{letter-spacing:.04em}} p {{line-height:1.8;white-space:pre-wrap}}
    .track {{height:6px;overflow:hidden;border-radius:8px;background:#64748b33;margin:28px 0 16px}}
    .track div {{height:100%;width:35%;border-radius:8px;background:#5279ff;animation:loading 1.4s ease-in-out infinite}}
    @keyframes loading {{from {{transform:translateX(-110%)}} to {{transform:translateX(390%)}}}}
    @media(prefers-reduced-motion:reduce) {{.track div {{animation:none;width:100%;opacity:.6}}}}
    </style>
    <main><h1>Local AI Summarizer System</h1>{loading}<p id="status" role="status" aria-live="polite">{html.escape(message)}</p><small>{'창을 닫고 다시 실행할 수 있습니다.' if error else '잠시만 기다려 주세요.'}</small></main></html>'''


class DesktopHost:
    def __init__(self, root: Path):
        self.root = root
        self.process = None
        self.cover = None
        self.navigating = False
        self.window = None
        self.stopped = threading.Event()
        self.process_lock = threading.Lock()
        self.log_path = root / "data/logs/desktop.log"

    def _spawn(self, command, log):
        with self.process_lock:
            if self.stopped.is_set():
                raise RuntimeError("실행이 취소되었습니다.")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(find_project_root())
            environment["PYTHONUNBUFFERED"] = "1"
            self.process = subprocess.Popen(command, cwd=str(self.root), env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **child_options())
            return self.process

    def install_cover(self):
        """Keep a native surface above WebView until its application has painted."""
        from System.Drawing import Color, ContentAlignment, Font
        from System.Windows.Forms import Panel, Label, DockStyle, ProgressBar, ProgressBarStyle
        form = self.window.native
        panel = Panel()
        panel.Dock = DockStyle.Fill
        panel.BackColor = Color.FromArgb(11, 13, 16)
        label = Label()
        label.Dock = DockStyle.Fill
        label.Text = 'LASS\n\n노트 작업공간을 준비하고 있습니다.'
        label.TextAlign = ContentAlignment.MiddleCenter
        label.ForeColor = Color.FromArgb(245, 247, 250)
        label.Font = Font('Segoe UI', 18)
        bar = ProgressBar()
        bar.Dock = DockStyle.Bottom
        bar.Height = 6
        bar.Style = ProgressBarStyle.Marquee
        panel.Controls.Add(label)
        panel.Controls.Add(bar)
        form.Controls.Add(panel)
        panel.BringToFront()
        self.cover = panel

    def set_cover(self, visible):
        if self.cover is None or self.stopped.is_set():
            return
        from System import Action
        def update():
            self.cover.Visible = visible
            if visible:
                self.cover.BringToFront()
        form = self.window.native
        if not form.IsDisposed:
            form.Invoke(Action(update))

    def document_loaded(self):
        if not self.navigating and not self.stopped.is_set():
            self.set_cover(False)
            self.window.show()

    def wait_for_screen(self, process, timeout=90):
        deadline = time.monotonic() + timeout
        while not self.stopped.wait(.1):
            if process.poll() is not None:
                raise RuntimeError('노트 화면 서버가 종료되었습니다.')
            try:
                ready = self.window.evaluate_js('Boolean(document.getElementById("lass-screen-ready"))') is True
            except Exception:
                ready = False
            if ready:
                self.set_cover(False)
                self.window.show()
                return
            if time.monotonic() > deadline:
                raise RuntimeError('화면 렌더링 준비 시간이 초과되었습니다.')

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
                # Show the workspace before model downloads or CUDA initialization.
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
                        self.set_cover(True)
                        self.navigating = True
                        self.window.load_url(url)
                        self.wait_for_screen(process)
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("노트 화면 준비 시간이 초과되었습니다.")
                while not self.stopped.wait(1):
                    if process.poll() is not None:
                        raise RuntimeError("백그라운드 서버가 종료되었습니다. 다시 실행해 주세요.")
        except Exception as exc:
            if not self.stopped.is_set():
                self.navigating = False
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
    parser.add_argument("--serve", type=int)
    parser.add_argument("--prepare-models", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--llm-worker", action="store_true")
    args = parser.parse_args()
    if args.llm_worker:
        from src.inference_worker import main as worker_main
        worker_main()
        return
    root = args.root.resolve() if args.root else (
        Path(os.environ["LOCALAPPDATA"]) / "LASS-IQ4-Moonshine" if getattr(sys, "frozen", False) else find_project_root())
    root.mkdir(parents=True, exist_ok=True)
    prepare_stdio(root, "server" if args.serve else "models" if args.prepare_models else "check" if args.self_test else "desktop")
    configure_paths(root)
    if args.serve or args.prepare_models or args.self_test:
        try:
            if args.serve:
                serve(args.serve)
            elif args.prepare_models:
                from src.config import Settings
                from src.model_assets import prepare_models
                def progress(message):
                    print(message, flush=True)
                prepare_models(Settings.from_env(), progress=progress)
                from src.llm.native_runtime import prepare_native
                prepare_native()
                import llama_cpp
                gpu = bool(llama_cpp.llama_supports_gpu_offload())
                device = {'engine': 'llama.cpp', 'quantization': 'IQ4_XS',
                          'gpu_offload_supported': gpu}
                (root / 'data/runtime-device.json').write_text(json.dumps(device), encoding='utf-8')
                progress('GGUF 런타임 준비 완료 · 실행 시 GPU/CPU 자동 선택')
            else:
                self_test(root)
        except Exception:
            details = traceback.format_exc()
            (root / "runtime-error.log").write_text(details, encoding="utf-8")
            if sys.stderr:
                print(details, file=sys.stderr)
            raise SystemExit(1)
        return
    from src.windows_runtime import ensure_webview2
    ensure_webview2()
    import webview
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("LASS.LocalAISummarizerSystem")
    webview.settings["ALLOW_DOWNLOADS"] = True
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
                                        width=1320, height=900, min_size=(900, 620),
                                        hidden=True, background_color="#0b0d10")
    host.window.events.before_show += host.install_cover
    host.window.events.loaded += host.document_loaded
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
        webview.start(host.start_server, private_mode=False, gui="edgechromium",
                      storage_path=str(root / "data/webview"),
                      icon=str(find_project_root() / "src/assets/lass.ico"))
    finally:
        host.close()
        instance.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
