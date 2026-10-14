"""Build the Windows x64 distribution with its own Python and ML runtime."""
from pathlib import Path
import argparse
import shutil
import sys
import threading
import os
import subprocess
import hashlib
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iscc", type=Path, help="Path to Inno Setup 6 ISCC.exe")
    parser.add_argument("--skip-bundle", action="store_true")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("Build LASS on Windows x64.")
    sys.path.insert(0, str(ROOT))
    from src.backup import backup_sources
    print(f"Source backup: {backup_sources(ROOT)}", flush=True)
    if not args.skip_bundle:
        from src.llm.native_runtime import prepare_native
        prepare_native()
        import llama_cpp
        print(f'Bundling llama.cpp {llama_cpp.__version__}, IQ4_XS, GPU offload: {llama_cpp.llama_supports_gpu_offload()}', flush=True)
        from PyInstaller.__main__ import run
        # Python 3.10 on Windows has a small main-thread native stack. Torch's
        # dependency graph needs both the spec's recursion budget and more stack.
        threading.stack_size(64 * 1024 * 1024)
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(run, ["--clean", "--noconfirm", "--distpath", str(ROOT / "dist"),
                                  "--workpath", str(ROOT / "build/desktop"),
                                  str(ROOT / "scripts/LASS.spec")]).result()
        sys.path.insert(0, str(ROOT))
        from src.distribution import write_manifest
        write_manifest(ROOT, ROOT / "dist/LASS/_internal")
        compiler = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
        subprocess.run([str(compiler), "/nologo", "/target:winexe", "/platform:x64",
                        "/reference:System.Windows.Forms.dll", f"/win32icon:{ROOT / 'src/assets/lass.ico'}",
                        f"/out:{ROOT / 'dist/LASS.exe'}", str(ROOT / "scripts/launcher.cs")], check=True)
        print(ROOT / "dist/LASS.exe")

    compiler = args.iscc or os.environ.get('INNO_SETUP_COMPILER') or shutil.which('ISCC.exe')
    if not compiler:
        compiler = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Inno Setup 6/ISCC.exe'
    if not Path(compiler).is_file():
        print('Portable EXE ready. Optional installer requires --skip-bundle --iscc <ISCC.exe>.')
        return
    sys.path.insert(0, str(ROOT))
    from src import __version__
    from src.distribution import write_manifest
    # --skip-bundle must not label an older bundle as the new release.
    write_manifest(ROOT, ROOT / 'dist/LASS/_internal')
    subprocess.run([str(compiler), f'/DMyAppVersion={__version__}', str(ROOT / 'scripts/installer.iss')], check=True)
    installer = ROOT / f'dist/installer/LASS-Setup-{__version__}.exe'
    with installer.open('rb') as handle:
        checksum = hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
        if checksum is None:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
            checksum = digest.hexdigest()
    installer.with_suffix('.exe.sha256').write_text(f'{checksum}  {installer.name}\n', encoding='ascii')
    print(installer)


if __name__ == '__main__':
    main()
