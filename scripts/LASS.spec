# Model weights stay outside the EXE in LOCALAPPDATA.
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata
sys.setrecursionlimit(5000)

root = Path(SPECPATH).parent
datas = [(str(path), str(path.parent.relative_to(root)))
         for path in (root / "src").rglob("*")
         if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"]
datas += [(str(root / "prompts"), "prompts"),
         (str(root / ".streamlit/config.toml"), ".streamlit")]
import importlib.util
llama_site = Path(importlib.util.find_spec('llama_cpp').origin).parents[1]
binaries = [(str(p), 'llama_cpp/lib') for folder in ('cuda_runtime', 'cublas')
            for p in (llama_site / 'nvidia' / folder / 'bin').glob('*.dll')]
datas += [(str(root / 'vendor/llama_cpu'), 'vendor/llama_cpu')]
# ui/ is a namespace directory; collect_submodules alone misses the app and
# its document/audio dependencies. Include every application module explicitly.
hiddenimports = [".".join(path.relative_to(root).with_suffix("").parts)
                 for path in (root / "src").rglob("*.py")
                 if "__pycache__" not in path.parts and path.name != "__init__.py"]
for package in ("streamlit", "webview", "moonshine_voice", "rapidocr", "langchain_core", "llama_cpp"):
    package_data, package_binaries, package_imports = collect_all(package)
    datas += package_data
    binaries += package_binaries
    hiddenimports += package_imports
for package in ('llama-cpp-python', 'langchain-core', 'moonshine-voice'):
    datas += copy_metadata(package, recursive=True)

a = Analysis([str(root / "src/desktop.py")], pathex=[str(root)],
             binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             excludes=["sherpa_onnx", "torch", "transformers", "accelerate", "whisper", "funasr", "torchaudio", "tensorflow", "jax", "flax", "numba", "llvmlite",
                       "sklearn", "matplotlib", "IPython", "pytest", "notebook", "jupyter", "librosa",
                       "webview.platforms.cocoa", "webview.platforms.gtk", "webview.platforms.qt"],
             module_collection_mode={"src": "py"},
             noarchive=False)
# Native variants belong only in their backend directories, never in bundle root.
native_names = {'ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll', 'ggml.dll', 'llama.dll', 'mtmd.dll'}
a.binaries = [entry for entry in a.binaries if entry[0] not in native_names]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="LASS",
          console=False, debug=False, strip=False, upx=False,
          manifest=str(root / "src/assets/lass.manifest"),
          icon=str(root / "src/assets/lass.ico"))
collection = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="LASS")
