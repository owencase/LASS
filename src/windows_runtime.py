"""Locate or install Microsoft's per-user WebView2 runtime."""
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlretrieve
import winreg

RUNTIME_KEY = r"Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
BOOTSTRAPPER_URL = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"


def webview2_installed():
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY):
            try:
                with winreg.OpenKey(hive, RUNTIME_KEY, 0, winreg.KEY_READ | view) as key:
                    version = winreg.QueryValueEx(key, "pv")[0]
                    if version and version != "0.0.0.0":
                        return True
            except OSError:
                pass
    return False


def ensure_webview2():
    if webview2_installed():
        return
    with tempfile.TemporaryDirectory(prefix="lass-webview2-") as directory:
        installer = Path(directory) / "MicrosoftEdgeWebview2Setup.exe"
        urlretrieve(BOOTSTRAPPER_URL, installer)
        subprocess.run([str(installer), "/silent", "/install"], check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=600)
    if not webview2_installed():
        raise RuntimeError("WebView2 설치에 실패했습니다. 인터넷 연결을 확인하고 다시 실행해 주세요.")
