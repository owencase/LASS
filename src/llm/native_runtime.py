"""Choose bundled CPU/CUDA DLLs before importing llama_cpp."""
import ctypes
import importlib.util
import os
from pathlib import Path
import sys

_handles = []
_libraries = []

def gpu_available():
    try:
        driver = ctypes.WinDLL('nvcuda.dll')
        count = ctypes.c_int()
        return driver.cuInit(0) == 0 and driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
    except OSError:
        return False

def prepare_native():
    root = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[2]
    cpu = root / 'vendor/llama_cpu'
    if not gpu_available() or os.environ.get('LASS_EXAONE_DEVICE', 'auto') == 'cpu':
        if cpu.is_dir():
            os.environ['LLAMA_CPP_LIB_PATH'] = str(cpu)
    if getattr(sys, 'frozen', False):
        directories = [root / 'llama_cpp/lib']
    else:
        spec = importlib.util.find_spec('llama_cpp')
        site = Path(spec.origin).parents[1]
        directories = [site / 'nvidia/cuda_runtime/bin', site / 'nvidia/cublas/bin', site / 'llama_cpp/lib']
    for directory in directories:
        if directory.is_dir():
            _handles.append(os.add_dll_directory(str(directory)))
            os.environ['PATH'] = str(directory) + os.pathsep + os.environ['PATH']
    # Freeze can discover both CPU and CUDA libraries with the same DLL names.
    # Preload the chosen backend in dependency order so Windows cannot mix them.
    selected = Path(os.environ.get('LLAMA_CPP_LIB_PATH', str(directories[-1])))
    for name in ('ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll', 'ggml.dll', 'llama.dll'):
        path = selected / name
        if path.is_file():
            _libraries.append(ctypes.WinDLL(str(path)))
