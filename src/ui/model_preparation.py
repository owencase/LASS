"""Model-specific downloads owned by the server, independent of UI reruns."""
import streamlit as st

from src.model_assets import MODELS, download_model, model_ready
from src.summary_jobs import SummaryJobs

model_jobs = SummaryJobs()


def models_ready(settings):
    return all(model_ready(settings, key) for key, _, _ in MODELS)


@st.fragment(run_every=.5)
def render_model_preparation(settings, ready=None, *, required=False):
    statuses = {key: model_ready(settings, key) for key, _, _ in MODELS}
    if required and all(statuses.values()):
        st.rerun()
    if not required and all(statuses.values()):
        return
    st.subheader('로컬 AI 모델 준비')
    st.caption('최초 한 번 다운로드합니다. 두 모델이 준비되면 작업공간으로 자동 이동합니다. 인터넷 연결이 필요합니다.'
               if required else '필요한 모델을 다운로드해 주세요. 저장된 노트는 계속 사용할 수 있습니다.')
    for key, name, description in MODELS:
        job_key = f'models:{settings.data_dir}:{key}'
        job = model_jobs.snapshot(job_key)
        running = bool(job and job.running)
        complete = statuses[key] and not running
        with st.container(border=True):
            details, action = st.columns([5, 1], vertical_alignment='center')
            with details:
                st.markdown(f'### **{name}**')
                st.caption(description)
            with action:
                if st.button('완료' if complete else '다운로드 중' if running else '이어받기' if job and job.error else '다운로드',
                             disabled=complete or running, key=f'download-{key}', use_container_width=True):
                    def work(progress, token, model_key=key):
                        download_model(settings, model_key,
                                       lambda fraction, message: progress(message, fraction))
                    model_jobs.start(job_key, work)
                    st.rerun()
            if complete:
                st.success('✓ 완료')
                st.progress(1.0, text='100%')
            elif running:
                st.progress(job.fraction, text=f'{job.fraction:.1%} · {job.progress}')
            elif job and job.error:
                st.error(f'다운로드 실패: {job.error}')
                st.caption('완료된 파일과 다운로드 조각을 보존했습니다. 이어받기로 재개할 수 있습니다.')
    st.html('<span id="lass-screen-ready" hidden></span>')
