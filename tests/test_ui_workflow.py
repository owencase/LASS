import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class UIWorkflowTests(unittest.TestCase):
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
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            'LASS_DATA_DIR': directory,
            'LASS_EXAONE_MODEL_DIR': str(Path(directory) / 'no-model'),
        }):
            app = AppTest.from_file(str(Path(__file__).parents[1] / 'src/ui/app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            next(b for b in app.button if b.label == '＋ 새 노트').click().run()
            self.assertFalse(app.exception)
            self.assertIn('PDF · PPT 추가', [button.label for button in app.button])
            self.assertIn('음성 파일 업로드', [button.label for button in app.button])
            self.assertIn('Transcription', [button.label for button in app.button])
            self.assertIn('AI Summary', [button.label for button in app.button])
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
            self.assertIn('＋ 새 노트', [button.label for button in app.button])
            self.assertNotIn('다크', [button.label for button in app.button])
            self.assertFalse(app.exception)
