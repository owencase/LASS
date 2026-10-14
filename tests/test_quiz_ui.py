import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.quiz import QuizStore
from src.storage import LectureRepository
from tests.test_quiz import exam


class QuizUITests(unittest.TestCase):
    def test_quiz_navigation_omr_submission_retake_and_chat(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "LASS_DATA_DIR": directory, "LASS_EXAONE_MODEL_DIR": str(Path(directory) / "no-model"),
        }):
            repository = LectureRepository(Path(directory) / "lass.db")
            folder = repository.create_folder("과목")
            note = repository.create_empty("학습 노트", folder.id)
            repository.update(note.id, summary="학습 원리 요약")
            store = QuizStore(repository)
            quiz_id = store.save(folder.id, exam())
            app = AppTest.from_file(str(Path(__file__).parents[1] / "src/ui/app.py"), default_timeout=30).run()
            app.button(key="quiz-button").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["workspace-mode"], "quiz")
            self.assertNotIn("PDF · PPT 추가", [button.label for button in app.button])
            app.button(key="quiz-start").click().run()
            attempt = app.session_state["quiz-attempt"]
            self.assertEqual(len([radio for radio in app.radio if str(radio.key).startswith("omr-")]), 20)
            first_key = f"omr-{attempt['id']}-1"
            app.radio(key=first_key).set_value("O").run()
            app.button(key="quiz-close").click().run()
            app.button(key="quiz-button").click().run()
            self.assertEqual(app.radio(key=first_key).value, "O")
            app = AppTest.from_file(str(Path(__file__).parents[1] / "src/ui/app.py"), default_timeout=30).run()
            app.button(key="quiz-button").click().run()
            self.assertEqual(app.session_state["quiz-attempt"]["id"], attempt["id"])
            self.assertEqual(app.radio(key=first_key).value, "O")
            app.radio(key="quiz-layout").set_value("좌우 배치").run()
            for question in exam():
                app.radio(key=f"omr-{attempt['id']}-{question['question_number']}").set_value(question["correct_answer"])
            store.save(folder.id, exam())
            with patch("src.ui.quiz_view.quiz_jobs.snapshot", return_value=SimpleNamespace(id="completed-generation", running=False, error="")):
                app.run()
            self.assertEqual(app.session_state[f"quiz-selection-{folder.id}"], quiz_id)
            self.assertEqual(app.session_state["quiz-attempt"]["id"], attempt["id"])
            next(button for button in app.button if button.label == "제출 및 채점").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["quiz-result"]["score"], 100)
            self.assertEqual(len(store.attempts(folder.id)), 1)
            app.button(key="quiz-start").click().run()
            self.assertNotEqual(app.session_state["quiz-attempt"]["id"], attempt["id"])
            self.assertTrue(all(radio.value is None for radio in app.radio if str(radio.key).startswith("omr-")))
            app.button(key="quiz-close").click().run()
            self.assertEqual(app.session_state["workspace-mode"], "notes")
            app.button(key="chatting-button").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_input), 1)
            self.assertEqual(store.get(quiz_id)["questions"], exam())

    def test_invalid_saved_exam_shows_recovery_message_instead_of_crashing(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "LASS_DATA_DIR": directory, "LASS_EXAONE_MODEL_DIR": str(Path(directory) / "no-model"),
        }):
            repo = LectureRepository(Path(directory) / "lass.db")
            folder = repo.create_folder("과목")
            store = QuizStore(repo)
            quiz_id = store.save(folder.id, exam())
            with repo._connect() as connection:
                connection.execute("UPDATE folder_quizzes SET questions_json = ? WHERE id = ?", ('{"broken":true}', quiz_id))
                connection.execute("DROP TABLE quiz_concepts")
            connection.close()
            app = AppTest.from_file(str(Path(__file__).parents[1] / "src/ui/app.py"), default_timeout=30).run()
            app.button(key="quiz-button").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("형식이 올바르지" in item.value for item in app.error))
            self.assertEqual(list(QuizStore(repo).previous_questions(folder.id)), [])


if __name__ == "__main__":
    unittest.main()
