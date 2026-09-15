import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.storage import LectureRepository, LectureSegment
from src.ui.components import build_markdown_export, inject_theme, render_timeline


class UIComponentTests(unittest.TestCase):
    def test_upload_transport_has_memory_bounds(self) -> None:
        config = (
            Path(__file__).parents[1] / ".streamlit" / "config.toml"
        ).read_text(encoding="utf-8")

        self.assertIn("maxUploadSize = 200", config)
        self.assertIn("maxMessageSize = 32", config)
        self.assertIn('toolbarMode = "minimal"', config)
        self.assertIn('address = "127.0.0.1"', config)
        self.assertIn('fileWatcherType = "none"', config)

    def test_timeline_is_one_contiguous_html_block(self) -> None:
        segments = [
            LectureSegment(0, 1, "첫 문장"),
            LectureSegment(2, 3, "두 번째 문장"),
        ]
        with patch("src.ui.components.st.markdown") as markdown:
            render_timeline(segments)

        html = markdown.call_args.args[0]
        self.assertEqual(html.count('class="lass-timeline-row"'), 2)
        self.assertNotIn("\n        <div", html)
        self.assertIn("00:00:02", html)

    def test_markdown_export_contains_summary_transcript_and_memo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = LectureRepository(Path(directory) / "lass.db")
            note = repository.create_empty("내보내기 노트")
            note = repository.complete(
                note.id,
                prompt_name="기본 요약",
                language="ko",
                duration=2,
                transcript="전사 본문",
                segments=[LectureSegment(0, 2, "전사 본문")],
                summary="요약 본문",
            )
            note = repository.update(note.id, notes="메모 본문")

        exported = build_markdown_export(note)
        self.assertIn("## AI 요약\n\n요약 본문", exported)
        self.assertIn("## 전사 · 추출 텍스트\n\n전사 본문", exported)
        self.assertIn("## 내 메모\n\n메모 본문", exported)

    def test_theme_forces_native_widgets_to_dark_colors(self) -> None:
        with patch("src.ui.components.st.markdown") as markdown:
            inject_theme()

        css = markdown.call_args.args[0]
        self.assertIn("color-scheme: dark", css)
        self.assertIn('[data-baseweb="tab-list"]', css)
        self.assertIn('[data-testid="stExpander"]', css)

    def test_light_theme_uses_high_contrast_text_tokens(self) -> None:
        with patch("src.ui.components.st.markdown") as markdown:
            inject_theme("light", 110)

        css = markdown.call_args.args[0]
        self.assertIn("color-scheme: light", css)
        self.assertIn("--text: #1e293b", css)
        self.assertIn("--muted: #64748b", css)
        self.assertIn("--accent: #2563eb", css)
        self.assertIn("--primary-text: #ffffff", css)
        self.assertIn("--text-scale: 1.10", css)

    def test_theme_hides_streamlit_distribution_chrome(self) -> None:
        with patch("src.ui.components.st.markdown") as markdown:
            inject_theme()

        css = markdown.call_args.args[0]
        self.assertIn('[data-testid="stToolbar"]', css)
        self.assertIn(".stDeployButton", css)

    def test_workspace_uses_fixed_shell_and_theme_aware_hover(self) -> None:
        with patch("src.ui.components.st.markdown") as markdown:
            inject_theme("light")

        css = markdown.call_args.args[0]
        self.assertIn(".st-key-lass-layout", css)
        self.assertIn("overflow:hidden !important", css)
        self.assertIn(".st-key-lass-workspace-pane", css)
        self.assertIn(".st-key-lass-ai-pane", css)
        self.assertIn("overflow-y:auto", css)
        self.assertIn("summary:hover", css)
        self.assertIn('button[data-baseweb="tab"]:hover', css)
        self.assertIn("background:var(--hover) !important", css)
        self.assertIn(':has(input:checked)', css)
        self.assertIn("--control-border: #cbd5e1", css)
        self.assertIn("border:2px solid var(--control-border)", css)
        self.assertIn("float above the workspace", css)
        self.assertIn(".lass-center-resizer", css)
        self.assertIn(".st-key-lass-document-memo-layout", css)
        self.assertIn(".st-key-lass-document-pane", css)
        self.assertIn(".st-key-lass-recording-actions", css)
        self.assertIn('[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"]', css)
        self.assertIn(".st-key-lass-sidebar-header", css)
        self.assertIn(".st-key-toggle-library-panel", css)
        self.assertIn("height:100vh !important", css)

    def test_resizer_persists_each_layout_and_supports_keyboard(self) -> None:
        html = (
            Path(__file__).parents[1]
            / "src/ui/layout_component/frontend/index.html"
        ).read_text(encoding="utf-8")
        self.assertNotIn("lass-panel-widths", html)
        self.assertIn('event.key !== "ArrowLeft"', html)
        self.assertIn('role", "separator"', html)
        self.assertIn("pointerdown", html)
        self.assertNotIn("frame.contentDocument", html)
        self.assertNotIn('button[title="Zoom In"]', html)
        self.assertNotIn("restorePdfZoom", html)
        self.assertIn("lass-document-height-ratio-v2", html)
        self.assertIn("lass-document-width-ratio-v1", html)
        self.assertIn('querySelector(".st-key-lass-document-memo-layout")', html)
        self.assertIn('event.key !== "ArrowUp"', html)
        self.assertIn("setupTranscriptAutoscroll", html)
        self.assertIn("scrollTop = transcript.scrollHeight", html)

    def test_memo_editor_formats_and_auto_saves(self) -> None:
        html = (
            Path(__file__).parents[1]
            / "src/ui/editable_component/frontend/index.html"
        ).read_text(encoding="utf-8")
        self.assertIn('post("save_memo"', html)
        self.assertIn("setRangeText", html)
        self.assertIn("prefixLines", html)
        self.assertIn("}, 600)", html)

    def test_live_recorder_backpressures_bounded_chunks(self) -> None:
        html = (
            Path(__file__).parents[1]
            / "src/ui/live_recorder_component/frontend/index.html"
        ).read_text(encoding="utf-8")
        self.assertIn("audioBitsPerSecond:32000", html)
        self.assertIn("setTimeout", html)
        self.assertIn("ack_nonce", html)
        self.assertIn('action:"audio_chunk"', html)


if __name__ == "__main__":
    unittest.main()
