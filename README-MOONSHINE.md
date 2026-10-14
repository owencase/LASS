# LASS 2.1.1 · IQ4_XS + Moonshine Voice

2.0.0 프로젝트는 그대로 보존한 독립 후속 버전입니다.

설치: `dist/installer/LASS-Setup-2.1.1.exe`.
프로젝트 모델을 바로 사용하는 실행: `LASS.bat`.
설치 경로: `%LOCALAPPDATA%/Programs/LASS-IQ4-Moonshine`.
설치본 데이터 및 모델: `%LOCALAPPDATA%/LASS-IQ4-Moonshine`.
2.0.0의 데이터나 설치 경로를 덮어쓰지 않습니다.

LLM은 기존 EXAONE 4.0 1.2B IQ4_XS GGUF를 유지합니다.
STT는 Moonshine Voice 0.1.5를 사용합니다.
한국어: Tiny 26M, 양자화된 ORT 모델, 기본 언어.
영어: Medium Streaming 245M, quantized_26_08_21 ORT 모델.
한국어 전용 스트리밍 체크포인트는 현재 제공되지 않아 기존 녹음 구간 단위 전사로 동작합니다.
두 언어는 별도 모델이므로 자동 언어 감지 대신 목록의 음성 인식 언어에서 선택합니다.
환경 변수 `LASS_STT_LANGUAGE=ko|en`으로 초기 언어를 지정할 수도 있습니다.

첫 실행에서 LLM과 한국어/영어 STT 모델을 다운로드합니다. 총 모델 크기는 약 1.09GB입니다.
다운로드는 고정 URL, 크기, SHA-256으로 검증하며 중단된 다운로드를 재개합니다.
소스 프로젝트의 모델은 다운로드와 공식 CRC32C·SHA-256 검증을 완료했습니다.
설치 프로그램에는 모델 가중치를 포함하지 않습니다.

Powered by Moonshine AI.
한국어 Tiny에는 Moonshine AI Community License가 적용됩니다. MIT 모델과 구분해야 합니다.
사용·재배포 조건 및 상업 사용 시 조건은 `src/assets/MOONSHINE-LICENSE.txt`를 확인하세요.
영어 Medium Streaming 및 Moonshine 라이브러리는 MIT입니다.
모델 가중치는 수정하지 않았습니다.

빌드: `.venv/Scripts/python.exe scripts/build_desktop.py --iscc tools/inno/app/ISCC.exe`
재현 빌드 시 .venv를 새로 만들고 requirements.txt를 설치합니다. vendor/llama_cpu를 함께 유지하세요.
이 작업의 빌드 환경은 기존 1.x 공통 Python 패키지를 읽기 참조하나 EXE에는 필요한 의존성을 포함합니다.

Windows 10 1903 이상 / Windows 11 x64가 필요합니다. EXE의 UTF-8 activeCodePage 설정으로 한글 모델 경로를 지원합니다.

## 2.1.1 수정

Moonshine 서버가 기본 Python-urllib 요청을 403(1010)으로 차단하는 문제를 앱 User-Agent 지정으로 해결했습니다.
Windows 앱 제거 시 설치 파일, 숨김 잔여물, 전용 노트·첨부파일·모델·브라우저 캐시·임시 파일·로그까지 삭제합니다.
업데이트 설치는 데이터를 유지하며, 기존 2.1.0 위에 2.1.1을 설치하면 새 제거 규칙이 적용됩니다.
이전 버전이 공용 %APPDATA%/pywebview에 남긴 캐시는 다른 앱·2.0.0과 공유할 수 있어 자동 삭제하지 않습니다. 2.1.1부터 전용 data/webview를 사용합니다.
문서 폴더의 프로젝트 및 소스 백업과 다른 버전의 데이터는 삭제하지 않습니다.
