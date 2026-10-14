import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.config import Settings
from src.llm.model_manager import ModelStatus
from src.pipeline import LASSPipeline
from src.storage import LectureRepository


class GlobalChatTests(unittest.TestCase):
    def test_english_is_rewritten_before_display_and_failure_never_leaks_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = LASSPipeline(Settings(data_dir=Path(directory)))
            pipeline.llm_client.chat_messages = Mock(side_effect=["This is an English answer to your question.", "한국어 답변입니다."])
            displayed = []
            answer = pipeline.answer_question([], "Answer in English", on_token=displayed.append)
            self.assertIn("한국어 답변입니다", answer)
            self.assertEqual(displayed, ["한국어 답변입니다."])
            pipeline.llm_client.chat_messages = Mock(return_value="This is an English answer to your question.")
            displayed.clear()
            with self.assertRaisesRegex(ValueError, "한국어 응답 검증"):
                pipeline.answer_question([], "Hello", on_token=displayed.append)
            self.assertEqual(displayed, [])

    def test_all_folders_unfiled_notes_cache_and_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(data_dir=Path(directory))
            repo = LectureRepository(settings.database_path)
            pipeline = LASSPipeline(settings)
            pipeline.llm_client.chat_messages = Mock(return_value="검색 답변")
            for title, folder in (("과목 A", repo.create_folder("A").id),
                                  ("과목 B", repo.create_folder("B").id), ("미분류", None)):
                note = repo.create_empty(title, folder)
                repo.complete(note.id, prompt_name="", language=None, duration=None,
                              segments=[], summary="", transcript=f"## {title}.pdf\n## 3페이지\n공통검색어 원문 {title}")
            revision = repo.notes_revision()
            answer = pipeline.answer_question(repo.iter_notes, "공통검색어", fingerprint=revision)
            for title in ("과목 A", "과목 B", "미분류"):
                self.assertIn(title + ".pdf", answer)
            self.assertIn("3페이지", answer)
            unused = Mock(side_effect=AssertionError("cached search read note bodies"))
            self.assertEqual(pipeline.answer_question(unused, "공통검색어", fingerprint=revision), answer)
            repo.delete(note.id)
            self.assertNotEqual(repo.notes_revision(), revision)
            updated = pipeline.answer_question(repo.iter_notes, "공통검색어", fingerprint=repo.notes_revision())
            self.assertNotIn("미분류.pdf", updated)

    def test_empty_library_and_no_match_allow_general_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = LASSPipeline(Settings(data_dir=Path(directory)))
            pipeline.llm_client.chat_messages = Mock(return_value="안녕하세요")
            answer = pipeline.answer_question([], "안녕", [{"role": "user", "content": "이전 대화"}])
            self.assertIn("일반 답변 · 관련 노트 출처 없음", answer)
            self.assertIn("안녕하세요", answer)
            pipeline.llm_client.chat_messages.assert_called_once()
            messages = pipeline.llm_client.chat_messages.call_args.args[1]
            self.assertEqual(messages[0]["content"], "이전 대화")
            self.assertNotIn("검색한 근거 위치", answer)

    def test_global_history_persists_separately_from_folder_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lass.db"
            repo = LectureRepository(path)
            repo.add_chat_message(None, "user", "일반 질문")
            folder = repo.create_folder("과목")
            repo.add_chat_message(folder.id, "user", "예전 폴더 질문")
            repo = LectureRepository(path)
            self.assertIsNone(repo.list_chat_messages(None)[0].folder_id)
            self.assertEqual(repo.clear_chat_messages(None), 1)
            self.assertEqual(len(repo.list_chat_messages(folder.id)), 1)
            repo.add_chat_message(None, "assistant", "일반 답변")
            repo.delete_folder(folder.id)
            self.assertEqual(repo.list_chat_messages(None)[0].content, "일반 답변")

    def test_chat_opens_and_submits_without_folders_or_notes(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "LASS_DATA_DIR": directory, "LASS_EXAONE_MODEL_DIR": str(Path(directory) / "no-model"),
        }), patch("src.ui.app.LocalModelManager.status", return_value=ModelStatus(
            "test", Path(directory), True, 1, (), 1, True,
        )):
            answer_question = Mock(return_value="일반 답변 · 관련 노트 출처 없음\n\n안녕하세요")
            app = AppTest.from_file(str(Path(__file__).parents[1] / "src/ui/app.py"), default_timeout=30).run()
            def recorder_id():
                return next(item.proto.id for item in app.get("component_instance") if "lass_live_recorder" in item.proto.component_name)
            microphone_id = recorder_id()
            app.button(key="close-right-panel").click().run()
            self.assertEqual(recorder_id(), microphone_id)
            app.button(key="chatting-button").click().run()
            self.assertEqual(recorder_id(), microphone_id)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.selectbox), 0)
            self.assertEqual(len(app.chat_input), 1)
            self.assertNotIn("toggle-library-panel", [button.key for button in app.button])
            self.assertFalse(app.chat_input[0].disabled)
            with patch.object(LASSPipeline, "answer_question", answer_question):
                app.chat_input[0].set_value("안녕").run()
            self.assertFalse(app.exception)
            answer_question.assert_called_once()
            self.assertEqual(list(answer_question.call_args.args[0]()), [])
            repo = LectureRepository(Path(directory) / "lass.db")
            self.assertEqual([message.role for message in repo.list_chat_messages(None)], ["user", "assistant"])
            app.button(key="chat-close").click().run()
            app.button(key="chatting-button").click().run()
            self.assertEqual(len(app.chat_message), 2)
            previous = repo.active_chat_session()
            app.button(key="toggle-chat-panel").click().run()
            self.assertTrue(app.session_state['chat-panel-open'])
            app.button(key="clear-global-chat").click().run()
            self.assertEqual(repo.list_chat_messages(None), [])
            self.assertEqual(len(repo.list_chat_messages(None, session_id=previous)), 2)
            repo.set_metadata("active_chat_session", previous)
            app.run()
            self.assertIn("close-chat-panel", [button.key for button in app.button])
            self.assertNotIn("toggle-chat-panel", [button.key for button in app.button])
            app.button(key="close-chat-panel").click().run()
            self.assertFalse(app.session_state["chat-panel-open"])
            self.assertEqual(len(app.chat_message), 2)


if __name__ == "__main__":
    unittest.main()
