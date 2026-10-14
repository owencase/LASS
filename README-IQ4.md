# LASS 2.0.0 IQ4_XS

Independent copy of LASS 1.3.11. The existing project and user database are not modified.

Model: mradermacher/EXAONE-4.0-1.2B-i1-GGUF, IQ4_XS.
Runtime: llama-cpp-python 0.3.36, CUDA 12.4 wheel with CPU fallback.
Input budget remains 2048 tokens. Runtime context is max(8192, input budget + 4096).
EXAONE text chat template runs with thinking disabled.
Model files stay outside the EXE and are downloaded with fixed revision and SHA-256 checks.
Installed application: %LOCALAPPDATA%/Programs/LASS-IQ4.
Installed data/models: %LOCALAPPDATA%/LASS-IQ4.
Portable launch with project-local models/data: run LASS.bat.
Keep dist/LASS.exe and dist/LASS/ together; the EXE requires its bundled runtime.

Build: .venv/Scripts/python.exe scripts/build_desktop.py
Self test: dist/LASS/LASS.exe --root <project-directory> --self-test
