"""Manual update checks; no background network calls or forced installation."""
import os
import sys

import streamlit as st

from src import __version__
from src.backup import backup_sources
from pathlib import Path
from src.summary_jobs import summary_jobs
from src.updater import backup_data, check_update, download_update, launch_installer


def render_app_update(settings):
    with st.expander(f"앱 업데이트 · v{__version__}"):
        st.caption("새 버전을 다운로드하고 데이터를 백업한 뒤 기존 설치 위에 업데이트합니다. 노트·요약·모델은 유지됩니다.")
        busy = summary_jobs.is_running() or bool(st.session_state.get("recording-note-id"))
        if st.button("업데이트 확인", key="check-app-update", disabled=busy):
            try:
                with st.spinner("공식 릴리스 확인 중…"):
                    st.session_state["available-app-update"] = check_update()
                st.session_state["app-update-checked"] = True
            except Exception as exc:
                st.session_state.pop("available-app-update", None)
                st.session_state["app-update-checked"] = False
                st.error(f"업데이트를 확인하지 못했습니다: {exc}")
        release = st.session_state.get("available-app-update")
        if release:
            st.info(f"새 버전 v{release['version']}을 사용할 수 있습니다.")
            desktop = bool(getattr(sys, "frozen", False)) and os.name == "nt"
            if not desktop:
                st.caption("설치형 Windows 앱에서 업데이트를 적용할 수 있습니다.")
            if st.button("백업 후 업데이트 설치", key="install-app-update", disabled=busy or not desktop):
                meter = st.progress(0.0, text="설치 파일 다운로드 중…")
                try:
                    installer = download_update(release, settings.data_dir.parent / "updates",
                        lambda count: meter.progress(count / release["size"], text="설치 파일 다운로드 및 검증 중…"))
                    meter.progress(1.0, text="기존 데이터 백업 중…")
                    transcriber = st.session_state.pop("_live_transcriber", None)
                    if transcriber is not None:
                        transcriber.unload()
                    with st.spinner("노트·요약·첨부 자료를 백업합니다…"):
                        backup_sources(Path(__file__).resolve().parents[2])
                        backup = backup_data(settings.data_dir)
                    launch_installer(installer)
                    st.success(f"백업 완료: {backup}\n설치 창에서 진행해 주세요. LASS가 종료될 수 있습니다.")
                except Exception as exc:
                    st.error(f"업데이트 준비에 실패했습니다. 기존 설치와 데이터는 유지됩니다: {exc}")
        elif st.session_state.get("app-update-checked"):
            st.caption("설치 가능한 새 정식 릴리스가 없습니다.")
        if busy:
            st.caption("요약과 녹음이 끝난 뒤 업데이트할 수 있습니다.")
