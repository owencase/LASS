"""Pipe-owned model process: exiting returns weights and native allocator memory."""

import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from queue import Queue
from threading import Thread, Event


def main():
    # A windowed PyInstaller EXE can have None streams despite inherited pipes.
    if os.name == "nt" and any(stream is None for stream in (sys.stdin, sys.stdout, sys.stderr)):
        import ctypes
        import msvcrt
        from ctypes import wintypes
        get_handle = ctypes.WinDLL("kernel32").GetStdHandle
        get_handle.argtypes = [wintypes.DWORD]
        get_handle.restype = wintypes.HANDLE
        for name, number, mode, flag in [("stdin", -10, "r", os.O_RDONLY),
                                         ("stdout", -11, "w", os.O_WRONLY),
                                         ("stderr", -12, "w", os.O_WRONLY)]:
            if getattr(sys, name) is None:
                fd = msvcrt.open_osfhandle(get_handle(number), flag | os.O_BINARY)
                setattr(sys, name, open(fd, mode, encoding="utf-8", buffering=1))
    # Frozen Windows workers can also receive existing cp949 text streams.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if os.name == "nt":
            import msvcrt
            msvcrt.setmode(stream.fileno(), os.O_BINARY)
        stream.reconfigure(encoding="utf-8", errors="strict")
    output = sys.stdout
    # Load native DLLs before a thread blocks on the Windows CRT stdin lock.
    with redirect_stdout(sys.stderr):
        import numpy
    requests = Queue()

    def read_requests():
        if os.name == "nt":
            import ctypes
            import msvcrt
            from ctypes import wintypes
            peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
            peek.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                             wintypes.LPVOID, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
            handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            pause = Event()
            while True:
                available = wintypes.DWORD()
                if not peek(handle, None, 0, None, ctypes.byref(available), None):
                    break
                if available.value:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    requests.put(line)
                else:
                    pause.wait(.05)
        else:
            for line in sys.stdin:
                requests.put(line)
        # The desktop/server may disappear during generation. Never orphan a GPU job.
        os._exit(0)

    Thread(target=read_requests, daemon=True).start()

    def send(**message):
        output.write(json.dumps(message, ensure_ascii=True) + "\n")
        output.flush()

    client = None
    while True:
        try:
            request = json.loads(requests.get())
            with redirect_stdout(sys.stderr):
                if client is None:
                    from src.llm.model_loader import LocalEXAONEClient
                    config = request["config"]
                    config["model_dir"] = Path(config["model_dir"])
                    client = LocalEXAONEClient(**config)
                client._runtime()
                send(device=client.active_device)
                for text in client.stream_messages(request["messages"]):
                    send(token=text)
            send(done=True, device=client.active_device)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            send(error=str(exc))
            return


if __name__ == "__main__":
    main()
