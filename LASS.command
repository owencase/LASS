#!/bin/zsh

set -e

if [[ -d "${0:A:h}/LASS.app" ]]; then
    open "${0:A:h}/LASS.app"
    exit 0
fi

LASS_ROOT=${0:A:h}
cd "$LASS_ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
    echo "LASS 실행 환경을 처음 한 번만 준비합니다."
    python3 -m venv .venv
fi

RUNTIME_VERSION="2026-09-14-bigvox-v1"
RUNTIME_STAMP=".venv/.lass-runtime-version"
if [[ ! -f "$RUNTIME_STAMP" ]] || ! grep -qx "$RUNTIME_VERSION" "$RUNTIME_STAMP"; then
    echo "LASS 실행 환경의 변경된 패키지를 동기화합니다."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install --no-cache-dir -r requirements.txt
    printf '%s\n' "$RUNTIME_VERSION" > "$RUNTIME_STAMP"
fi

echo "LASS 전용 창을 엽니다. 일반 실행은 LASS.app을 사용해 주세요."
exec .venv/bin/python -m src.desktop
