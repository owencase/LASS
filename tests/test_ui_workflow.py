import os
import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class UIWorkflowTests(unittest.TestCase):
    def test_library_blank_space_create_events_create_folder_and_note(self):
        from src.config import Settings
        from src.ui.app import _handle_library_event

        repository = Mock()
        repository.create_empty.return_value = SimpleNamespace(id='note-1')
        state = {}
        with (
            patch('src.ui.app.st.session_state', state),
            patch('src.ui.app.st.rerun'),
        ):
            _handle_library_event(
                {'action': 'create_folder', 'name': '강의', 'nonce': 'folder-1'},
                repository,
                Settings(),
            )
            _handle_library_event(
                {'action': 'create_note', 'title': '새 강의', 'nonce': 'note-1'},
                repository,
                Settings(),
            )

        repository.create_folder.assert_called_once_with('강의')
        repository.create_empty.assert_called_once_with(
            title='새 강의', folder_id=None
        )
        self.assertEqual(state['selected_note_id'], 'note-1')

        repository.create_empty.reset_mock()
        with (
            patch('src.ui.app.st.session_state', state),
            patch('src.ui.app.st.rerun'),
        ):
            _handle_library_event(
                {
                    'action': 'create_note',
                    'title': '폴더 노트',
                    'folder_id': 'folder-1',
                    'nonce': 'note-2',
                },
                repository,
                Settings(),
            )

        repository.create_empty.assert_called_once_with(
            title='폴더 노트', folder_id='folder-1'
        )

    def test_summary_progress_is_only_in_editor_and_buttons_have_no_hover_help(self):
        source = (Path(__file__).parents[1] / 'src/ui/app.py').read_text(
            encoding='utf-8'
        )
        self.assertNotIn('summary_progress = st.empty()', source)
        self.assertIn('progress=job.progress if running else ""', source)
        tree = ast.parse(source)
        button_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {'button', 'download_button', 'form_submit_button'}
        ]
        self.assertFalse(
            any(keyword.arg == 'help' for call in button_calls for keyword in call.keywords)
        )

    def test_sidebar_omits_duplicate_brand_and_uploads_have_no_hover_help(self):
        root = Path(__file__).parents[1]
        app_source = (root / 'src/ui/app.py').read_text(encoding='utf-8')
        source_panel = (root / 'src/ui/source_panel.py').read_text(encoding='utf-8')

        self.assertNotIn('<strong>LASS</strong>', app_source)
        self.assertIn('<strong>내 노트</strong>', app_source)
        self.assertNotIn('<strong>AI Lecture Notes</strong>', app_source)
        self.assertNotIn('help=', app_source + source_panel)

    def test_library_header_has_chatting_and_quiz_actions(self):
        source = (Path(__file__).parents[1] / 'src/ui/app.py').read_text(
            encoding='utf-8'
        )

        self.assertIn('"Chatting"', source)
        self.assertIn('key="chatting-button"', source)
        self.assertIn('"Quiz"', source)
        self.assertIn('key="quiz-button"', source)
        self.assertNotIn('key="all-notes-button"', source)

    def test_interactive_summary_reads_every_window_including_middle(self):
        from src.config import Settings
        from src.pipeline import LASSPipeline

        pages = [f"페이지 {index} " + (str(index) * 1400) for index in range(40)]
        context = LASSPipeline(Settings())._all_context("\n\n".join(pages))
        text = "\n".join(context)
        for page in range(40):
            self.assertIn(f"페이지 {page} ", text)

    def test_folder_click_after_returning_from_chat_keeps_selected_note(self):
        from src.config import Settings
        from src.ui.app import _handle_library_event, _return_to_notes
        state = {"workspace-mode": "chat", "selected_note_id": "note-1",
                 "notes-panel-snapshot": {"library_open": True, "right_panel_open": False}}
        with patch("src.ui.app.st.session_state", state), patch("src.ui.app.st.rerun"):
            _return_to_notes()
            _handle_library_event({"action": "select_folder", "folder_id": "folder-1", "nonce": "new"},
                                  Mock(), Settings())
        self.assertEqual(state["workspace-mode"], "notes")
        self.assertEqual(state["selected_note_id"], "note-1")
        self.assertNotIn("selected_folder_id", state)
        self.assertTrue(state["library_open"])
        self.assertFalse(state["right_panel_open"])

    def test_uploaded_audio_is_independent_of_screen_and_clears_selection(self):
        from src.ui.app import _attach_and_process_path
        from src.config import Settings
        repo = Mock()
        repo.add_source.return_value = {"id": "source-1"}
        note = SimpleNamespace(id="note-1")
        state = {"browser-upload-open-audio-note-1": True}
        with patch("src.ui.app.st.session_state", state), patch("src.ui.app.st.rerun"), \
             patch("src.ui.app._release_live_transcriber"), patch("src.ui.app.get_pipeline"), \
             patch("src.ui.app.summary_jobs.start") as start:
            _attach_and_process_path(repo, note, Settings(), Path("sample.wav"), kind="audio", copied_upload=True)
        self.assertFalse(state["browser-upload-open-audio-note-1"])
        self.assertEqual(state["browser-upload-version-audio-note-1"], 1)
        self.assertEqual(start.call_args.args[0][0], "source")
        self.assertTrue(callable(start.call_args.args[1]))

    def test_native_document_picker_failure_opens_browser_upload_without_error(self):
        from src.ui.app import _attach_and_process_local_file

        state = {}
        with (
            patch("src.ui.app._desktop_file_path", side_effect=RuntimeError("offline")),
            patch("src.ui.app.st.session_state", state),
            patch("src.ui.app.st.rerun") as rerun,
            patch("src.ui.app.st.error") as error,
        ):
            _attach_and_process_local_file(
                Mock(), SimpleNamespace(id="note-1"), Mock(), kind="document"
            )

        self.assertTrue(state["browser-upload-open-document-note-1"])
        rerun.assert_called_once_with()
        error.assert_not_called()

    def test_sidebar_slots_do_not_move_the_workspace(self):
        source = (
            Path(__file__).parents[1] / "src" / "ui" / "app.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "library_column, workspace_column, right_column = st.columns(3, gap=None)",
            source,
        )
        self.assertNotIn("pane_names =", source)

    def test_document_navigation_keeps_page_bounds_and_zoom(self):
        import pymupdf
        from streamlit.testing.v1 import AppTest
        from src.storage import LectureRepository
        import streamlit as st
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "LASS_DATA_DIR": directory,
            "LASS_EXAONE_MODEL_DIR": str(Path(directory) / "no-model"),
        }):
            repository = LectureRepository(Path(directory) / "lass.db")
            note = repository.create_empty("페이지 이동 확인")
            source_path = Path(directory) / "pages.pdf"
            with pymupdf.open() as pdf:
                for _ in range(3):
                    pdf.new_page()
                pdf.save(source_path)
            source = repository.add_source(note.id, source_path)
            repository.complete_source(source["id"], transcript="자료")
            page_key = f"pdf-page-{note.id}-{source['id']}"
            app = AppTest.from_file(str(Path(__file__).parents[1] / "src/ui/app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            app.button(key=f"pdf-next-{note.id}-{source['id']}").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state[page_key], 2)
            app.button(key=f"pdf-next-{note.id}-{source['id']}").click().run()
            self.assertEqual(app.session_state[page_key], 3)
            self.assertTrue(app.button(key=f"pdf-next-{note.id}-{source['id']}").disabled)
            app.button(key=f"pdf-previous-{note.id}-{source['id']}").click().run()
            self.assertEqual(app.session_state[page_key], 2)
            app.button(key=f"pdf-zoom-in-{note.id}-{source['id']}").click().run()
            self.assertEqual(app.session_state[page_key], 2)
            self.assertFalse(app.exception)

    def test_three_panel_workflow_updates_on_the_first_click(self):
        from streamlit.testing.v1 import AppTest
        from src.storage import LectureRepository
        import streamlit as st
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            'LASS_DATA_DIR': directory,
            'LASS_EXAONE_MODEL_DIR': str(Path(directory) / 'no-model'),
        }):
            LectureRepository(Path(directory) / 'lass.db').create_empty()
            app = AppTest.from_file(str(Path(__file__).parents[1] / 'src/ui/app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertNotIn('toggle-theme', [button.key for button in app.button])
            self.assertNotIn('new-note-button', [button.key for button in app.button])
            self.assertNotIn('요약 형식', [widget.label for widget in app.selectbox])
            self.assertNotIn('＋ 새 노트', [button.label for button in app.button])
            self.assertNotIn('＋ 새 폴더', [button.label for button in app.button])
            self.assertIn('PDF · PPT 추가', [button.label for button in app.button])
            self.assertIn('음성 파일 업로드', [button.label for button in app.button])
            self.assertIn('Transcription', [button.label for button in app.button])
            self.assertIn('AI Summary', [button.label for button in app.button])
            labels = [button.label for button in app.button]
            layout_label = '상하 배치' if '상하 배치' in labels else '좌우 배치'
            self.assertEqual(labels[labels.index(layout_label) - 1], 'PDF · PPT 추가')
            source = (Path(__file__).parents[1] / 'src/ui/app.py').read_text(
                encoding='utf-8'
            )
            self.assertIn('EXAONE 모델을 준비하지 못했습니다.', source)
            next(b for b in app.button if b.label == 'PDF · PPT 추가').click().run()
            self.assertFalse(app.exception)
            self.assertIn('선택한 파일 추가', [button.label for button in app.button])
            next(b for b in app.button if b.label == '취소').click().run()
            next(b for b in app.button if b.label == 'AI Summary').click().run()
            self.assertFalse(app.exception)
            self.assertNotIn('요약 설정', [item.label for item in app.expander])
            self.assertIn('내 메모 포함', [item.label for item in app.checkbox])
            self.assertIn('요약하기', [button.label for button in app.button])
            self.assertIn(
                'Markdown 다운로드',
                [button.label for button in app.get('download_button')],
            )

            next(b for b in app.button if b.label == 'Transcription').click().run()
            self.assertFalse(app.exception)
            self.assertNotIn('요약 설정', [item.label for item in app.expander])

            self.assertTrue(app.session_state.get('library_open', True))
            self.assertIn('toggle-library-panel', [b.key for b in app.button])
            self.assertIn('toggle-right-panel', [b.key for b in app.button])
            next(b for b in app.button if b.label == '목록 닫기').click().run()
            self.assertFalse(app.exception)
            self.assertNotIn('＋ 새 노트', [button.label for button in app.button])
            self.assertIn('PDF · PPT 추가', [button.label for button in app.button])

            self.assertFalse(app.session_state['library_open'])
            self.assertIn('close-library-panel', [b.key for b in app.button])
            self.assertIn('toggle-library-panel', [b.key for b in app.button])
            next(b for b in app.button if b.label == '노트 목록').click().run()
            self.assertTrue(app.session_state['library_open'])
            self.assertIn('toggle-library-panel', [b.key for b in app.button])
            self.assertIn('close-library-panel', [b.key for b in app.button])
            self.assertNotIn('＋ 새 노트', [button.label for button in app.button])
            self.assertNotIn('다크', [button.label for button in app.button])
            self.assertFalse(app.exception)
