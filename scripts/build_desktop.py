"""Build a clickable launcher alongside the local project/runtime.

macOS: python scripts/build_desktop.py -> ./LASS.app
Windows: pip install pyinstaller pywebview; python scripts/build_desktop.py -> ./LASS.exe
The launcher and project are distributed together; model files are never baked
into a one-file executable or extracted again at every startup.
"""

from pathlib import Path
import plistlib
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def build_mac():
    import shutil
    dist = ROOT / "build/launcher"
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--windowed", "--name", "LASS", "--osx-bundle-identifier", "local.lass.workspace",
        "--distpath", str(dist), "--workpath", str(ROOT / "build/desktop"),
        "--specpath", str(ROOT / "build"), "--paths", str(ROOT),
        "--hidden-import", "src.desktop_picker", "--exclude-module", "torch",
        "--exclude-module", "transformers", "--exclude-module", "streamlit",
        str(ROOT / "src/desktop.py")], check=True, cwd=ROOT)
    source = dist / "LASS.app"
    info_path = source / "Contents/Info.plist"
    info = plistlib.loads(info_path.read_bytes())
    info.update({"CFBundleDisplayName": "LASS", "CFBundleShortVersionString": "1.2.0",
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": "노트에 음성을 녹음하기 위해 마이크를 사용합니다.",
        "NSDocumentsFolderUsageDescription": "LASS 프로젝트의 모델과 노트 자료를 읽고 저장합니다."})
    info_path.write_bytes(plistlib.dumps(info))
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(source)], check=True)
    app = ROOT / "LASS.app"
    if app.exists():
        shutil.rmtree(app)
    shutil.copytree(source, app, symlinks=True)
    print(app)


def build_windows():
    # Only the GUI host is frozen. The ML runtime stays in .venv beside it.
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile",
        "--windowed", "--name", "LASS", "--distpath", str(ROOT),
        "--workpath", str(ROOT / "build/desktop"), "--specpath", str(ROOT / "build"),
        "--exclude-module", "torch", "--exclude-module", "transformers",
        "--exclude-module", "streamlit", "--paths", str(ROOT),
        "--hidden-import", "src.desktop_picker", str(ROOT / "src/desktop.py")], check=True, cwd=ROOT)
    print(ROOT / "LASS.exe")


if __name__ == "__main__":
    if sys.platform == "darwin":
        build_mac()
    elif sys.platform == "win32":
        build_windows()
    else:
        raise SystemExit("Build the launcher on macOS or Windows.")
