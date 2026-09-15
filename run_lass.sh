#!/usr/bin/env bash

set -e

LASS_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$LASS_ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
    echo "Preparing the LASS environment for the first run."
    python3 -m venv .venv
fi

RUNTIME_VERSION="2026-09-14-bigvox-v1"
RUNTIME_STAMP=".venv/.lass-runtime-version"
if [[ ! -f "$RUNTIME_STAMP" ]] || ! grep -qx "$RUNTIME_VERSION" "$RUNTIME_STAMP"; then
    echo "Synchronizing the LASS runtime dependencies."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install --no-cache-dir -r requirements.txt
    printf '%s\n' "$RUNTIME_VERSION" > "$RUNTIME_STAMP"
fi

exec .venv/bin/python -m streamlit run src/ui/app.py --server.headless false
