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

    def test_summary_progress_is_above_actions_and_buttons_have_no_hover_help(self):
        source = (Path(__file__).parents[1] / 'src/ui/app.py').read_text(
            encoding='utf-8'
        )
        self.assertLess(
            source.index('summary_progress = st.empty()'),
            source.index('with st.container(key="lass-summary-actions")'),
        )
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

    def test_sidebar_uses_lass_brand_and_uploads_have_no_hover_help(self):
        root = Path(__file__).parents[1]
        app_source = (root / 'src/ui/app.py').read_text(encoding='utf-8')
        source_panel = (root / 'src/ui/source_panel.py').read_text(encoding='utf-8')

        self.assertIn('<strong>LASS</strong>', app_source)
        self.assertIn('Local AI Study Studio', app_source)
        self.assertNotIn('<strong>AI Lecture Notes</strong>', app_source)
        self.assertNotIn('help=', app_source + source_panel)

    def test_library_header_has_chatting_and_quiz_placeholders(self):
        source = (Path(__file__).parents[1] / 'src/ui/app.py').read_text(
            encoding='utf-8'
        )

        self.assertIn('"Chating"', source)
        self.assertIn('key="chatting-button"', source)
        self.assertIn('"Quiz"', source)
        self.assertIn('key="quiz-button"', source)
        self.assertNotIn('key="all-notes-button"', source)

    def test_interactive_summary_bounds_long_transcripts_and_keeps_progress(self):
        from src.config import Settings
        from src.ui.app import _interactive_summary_pages

        pages = [f"페이지 {index} " + (str(index) * 1400) for index in range(40)]
        repository = Mock()
        repository.iter_transcript_pages.return_value = iter(pages)
        note = SimpleNamespace(id="note-1", transcript_chars=48_000)

        result = list(_interactive_summary_pages(repository, note, Settings()))

        self.assertEqual(len(result), 1)
        self.assertLessEqual(len(result[0]), 3_200)
        self.assertIn("전체 자료 위치 1/40", result[0])
        self.assertIn("전체 자료 위치 40/40", result[0])

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

    def test_three_panel_workflow_updates_on_the_first_click(self):
        from streamlit.testing.v1 import AppTest
        from src.storage import LectureRepository
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            'LASS_DATA_DIR': directory,
            'LASS_EXAONE_MODEL_DIR': str(Path(directory) / 'no-model'),
        }):
            LectureRepository(Path(directory) / 'lass.db').create_empty()
            app = AppTest.from_file(str(Path(__file__).parents[1] / 'src/ui/app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertNotIn('＋ 새 노트', [button.label for button in app.button])
            self.assertNotIn('＋ 새 폴더', [button.label for button in app.button])
            self.assertIn('PDF · PPT 추가', [button.label for button in app.button])
            self.assertIn('음성 파일 업로드', [button.label for button in app.button])
            self.assertIn('Transcription', [button.label for button in app.button])
            self.assertIn('AI Summary', [button.label for button in app.button])
            labels = [button.label for button in app.button]
            layout_label = '상하 배치' if '상하 배치' in labels else '좌우 배치'
            self.assertEqual(labels[labels.index(layout_label) + 1], 'PDF · PPT 추가')
            source = (Path(__file__).parents[1] / 'src/ui/app.py').read_text(
                encoding='utf-8'
            )
            self.assertIn('요약 모델 파일이 준비되지 않았습니다.', source)
            next(b for b in app.button if b.label == 'PDF · PPT 추가').click().run()
            self.assertFalse(app.exception)
            self.assertIn('선택한 파일 추가', [button.label for button in app.button])
            next(b for b in app.button if b.label == '취소').click().run()
            next(b for b in app.button if b.label == 'AI Summary').click().run()
            self.assertFalse(app.exception)
            self.assertIn('요약 설정', [item.label for item in app.expander])
            self.assertIn('요약하기', [button.label for button in app.button])
            self.assertIn(
                'Markdown 다운로드',
                [button.label for button in app.get('download_button')],
            )

            next(b for b in app.button if b.label == 'Transcription').click().run()
            self.assertFalse(app.exception)
            self.assertNotIn('요약 설정', [item.label for item in app.expander])

            next(b for b in app.button if b.label == '목록 닫기').click().run()
            self.assertFalse(app.exception)
            self.assertNotIn('＋ 새 노트', [button.label for button in app.button])
            self.assertIn('PDF · PPT 추가', [button.label for button in app.button])

            next(b for b in app.button if b.label == '노트 목록').click().run()
            self.assertNotIn('＋ 새 노트', [button.label for button in app.button])
            self.assertNotIn('다크', [button.label for button in app.button])
            self.assertFalse(app.exception)
