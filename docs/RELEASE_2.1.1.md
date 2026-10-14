# LASS 2.1.1

Replace the earlier application with the Windows IQ4_XS + Moonshine release.

- EXAONE 4.0 1.2B IQ4_XS GGUF inference with CUDA/CPU runtimes.
- Moonshine Korean Tiny and English Medium Streaming with explicit language selection.
- Identify model-download requests with the app User-Agent to avoid HTTP 403; keep resume and SHA-256 verification.
- Keep temporary and browser cache files in the application's own data folder.
- Preserve data during upgrades. Explicit uninstall removes this version's installed files, notes, attachments, models, logs and caches.
- Preserve other versions, source projects, backups and shared legacy pywebview caches.

Validation: 182 regression tests; frozen first-run download of all 11 STT files without cache; Korean/English transcription; isolated install/uninstall cleanup test.

Setup EXE and model weights are build/download artifacts, not Git source files.
Install Inno Setup separately to build the Setup EXE with scripts/build_desktop.py.
