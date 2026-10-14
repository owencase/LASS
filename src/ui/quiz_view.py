"""Single scrolling exam with a separate OMR answer card."""

import time
import json
from uuid import uuid4

import streamlit as st

from src.quiz import QuizStore, generate_exam
from src.summary_jobs import SummaryJobs


quiz_jobs = SummaryJobs()


def _begin_attempt(quiz_id, repository):
    attempt = {"quiz_id": quiz_id, "id": uuid4().hex, "started": time.time(), "answers": {}}
    st.session_state["quiz-attempt"] = attempt
    repository.set_metadata(f"quiz-attempt:{quiz_id}", json.dumps(attempt))
    st.session_state.pop("quiz-result", None)


def _remember_answer(repository, quiz_id, question_number, widget_key):
    attempt = dict(st.session_state["quiz-attempt"])
    attempt["answers"] = dict(attempt.get("answers", {}))
    attempt["answers"][str(question_number)] = st.session_state[widget_key]
    st.session_state["quiz-attempt"] = attempt
    repository.set_metadata(f"quiz-attempt:{quiz_id}", json.dumps(attempt))


@st.fragment(run_every=2)
def _generation_progress(key, folder_id):
    job = quiz_jobs.snapshot(key)
    if not job:
        return
    if job.running:
        st.info(job.progress)
    elif st.session_state.get("quiz-job-completed") != job.id:
        st.session_state["quiz-job-completed"] = job.id
        if not job.error:
            attempt = st.session_state.get("quiz-attempt")
            if attempt and attempt["quiz_id"] == st.session_state.get(f"quiz-selection-{folder_id}") and not st.session_state.get("quiz-result"):
                st.success("새 시험을 저장했습니다. 현재 응시를 마치면 시험 목록에서 선택할 수 있습니다.")
                return
            st.session_state["quiz-select-latest"] = True
        st.rerun()
    if job.error:
        st.error(job.error)


def render_quiz_view(repository, model_status, get_pipeline):
    with st.container(key="lass-quiz-view"):
        st.subheader("퀴즈 시험지")
        folders = repository.list_folders()
        if not folders:
            st.info("먼저 과목 폴더를 만들고 학습할 노트를 넣어 주세요.")
            return
        folder_id = st.selectbox("과목 / 폴더", [folder.id for folder in folders],
            format_func=lambda value: next(folder.name for folder in folders if folder.id == value), key="quiz-folder")
        st.caption("총 20문항 · O/X 10문항 + 4지선다 10문항 · 100점 만점 · 제출 시 소요 시간과 점수 자동 저장")
        store = QuizStore(repository)
        quizzes = store.list(folder_id)
        key = f"quiz:{repository.database_path}:{folder_id}"
        job = quiz_jobs.snapshot(key)
        running = bool(job and job.running)
        draft = store.draft(folder_id)
        if draft:
            st.caption(f"검증 완료 {len(draft)}/20문항 저장됨 · 실패한 다음 문항부터 이어서 생성합니다.")
        if st.button("이어서 생성" if draft else "문제 재생성" if quizzes else "문제 생성", disabled=running or not model_status.ready or not repository.folder_has_content(folder_id), key="quiz-generate"):
            pipeline = get_pipeline()
            def work(progress, token):
                questions = generate_exam(pipeline, lambda: repository.iter_by_folder(folder_id),
                    store.previous_questions(folder_id), progress=progress, store=store, folder_id=folder_id)
                store.finish(folder_id, questions)
            quiz_jobs.start(key, work)
            st.rerun()
        if not model_status.ready:
            st.caption("모델 준비가 끝나면 새 문제를 생성할 수 있습니다. 저장된 시험은 바로 응시할 수 있습니다.")
        _generation_progress(key, folder_id)
        if not quizzes:
            return
        selection_key = f"quiz-selection-{folder_id}"
        if st.session_state.pop("quiz-select-latest", False):
            st.session_state[selection_key] = quizzes[0]["id"]
        labels = {quiz["id"]: f"시험 {len(quizzes) - index} · {quiz['created_at'][:19].replace('T', ' ')} UTC" for index, quiz in enumerate(quizzes)}
        quiz_id = st.selectbox("저장된 시험 (재응시 가능)", list(labels), format_func=labels.get, key=selection_key)
        try:
            quiz = store.get(quiz_id)
        except (KeyError, ValueError, TypeError):
            st.error("저장된 시험의 형식이 올바르지 않습니다. 문제 재생성으로 새 시험을 만들어 주세요.")
            return
        attempt = st.session_state.get("quiz-attempt")
        if not attempt or attempt["quiz_id"] != quiz_id:
            try:
                attempt = json.loads(repository.get_metadata(f"quiz-attempt:{quiz_id}") or "null")
                if attempt is not None and (not isinstance(attempt, dict) or
                        attempt.get("quiz_id") != quiz_id or not isinstance(attempt.get("id"), str) or
                        not isinstance(attempt.get("started"), (int, float)) or
                        not isinstance(attempt.get("answers"), dict)):
                    attempt = None
            except (ValueError, TypeError):
                attempt = None
            if attempt:
                st.session_state["quiz-attempt"] = attempt
            st.session_state.pop("quiz-result", None)
        result = next((saved for saved in store.attempts(folder_id) if saved["id"] == attempt["id"]), None) if attempt else None
        if result:
            st.session_state["quiz-result"] = result
        if st.button("다시 응시" if attempt else "응시 시작", key="quiz-start"):
            _begin_attempt(quiz_id, repository)
            st.rerun()
        if result:
            st.success(f"점수 {result['score']} / 100 · 소요 시간 {result['elapsed_seconds']:.1f}초 · 과목 기록에 저장했습니다.")
        layout = st.radio("시험지 / OMR 배치", ["상하 배치", "좌우 배치"], horizontal=True, key="quiz-layout")
        panels = st.columns([3, 2]) if layout == "좌우 배치" else (st.container(), st.container())
        with panels[0], st.container(height=600, key="quiz-question-sheet"):
            st.markdown("#### 시험 문제")
            for question in quiz["questions"]:
                label = "O/X" if question["question_type"] == "OX" else "객관식"
                st.markdown(f"**{question['question_number']}. [{label}]**")
                st.write(question["question"])
                if question["question_type"] == "MULTIPLE_CHOICE":
                    for index, option in enumerate(question["options"], 1):
                        st.text(f"{index}. {option}")
                if result:
                    st.caption(f"정답: {question['correct_answer']} · 출처: {question['source_reference']}")
                st.divider()
        with panels[1], st.container(height=600, key="quiz-omr-sheet"):
            st.markdown("#### OMR 답안 카드")
            if attempt:
                answers = []
                for question in quiz["questions"]:
                    options = question["options"]
                    number = question["question_number"]
                    widget_key = f"omr-{attempt['id']}-{number}"
                    saved_answer = attempt.get("answers", {}).get(str(number))
                    answers.append(st.radio(f"{number}번", options,
                        index=options.index(saved_answer) if saved_answer in options else None,
                        format_func=lambda value, values=options: value if values == ["O", "X"] else str(values.index(value) + 1),
                        horizontal=True, disabled=bool(result), key=widget_key,
                        on_change=_remember_answer, args=(repository, quiz_id, number, widget_key)))
                if st.button("제출 및 채점", disabled=bool(result), key="quiz-submit"):
                    try:
                        result = store.submit(quiz_id, answers, max(0, time.time() - attempt["started"]), attempt["id"])
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["quiz-result"] = result
                        st.rerun()
            else:
                st.info("응시 시작을 누르면 시간 측정과 답안 체크가 시작됩니다.")
        with st.expander("과목 응시 기록"):
            for saved in store.attempts(folder_id):
                st.write(f"{saved['created_at'][:19]} UTC · {saved['score']}점 · {saved['elapsed_seconds']:.1f}초")
