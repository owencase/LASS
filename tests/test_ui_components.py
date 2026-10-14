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

        self.assertIn("maxUploadSize = 2048", config)
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
        self.assertIn("fixed underneath the overlay drawers", css)
        self.assertIn(".lass-center-resizer", css)
        self.assertIn(".st-key-lass-document-memo-layout", css)
        self.assertIn(".st-key-lass-document-pane", css)
        self.assertIn(":has(.st-key-toggle-center-layout)", css)
        self.assertIn(":has(.st-key-lass-document-actions)", css)
        self.assertIn("flex:0 0 7.4rem", css)
        self.assertIn(".st-key-toggle-center-layout button,", css)
        self.assertIn(".st-key-lass-recording-actions", css)
        self.assertIn('[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"]', css)
        self.assertIn(".st-key-lass-sidebar-header", css)
        self.assertIn(".st-key-toggle-library-panel", css)
        self.assertIn(".st-key-toggle-right-panel", css)
        self.assertIn("place-items:center !important", css)
        self.assertIn('[data-testid="stMarkdown"] > div', css)
        self.assertIn('[role="tooltip"]', css)
        self.assertIn("visibility:hidden !important", css)
        self.assertIn("height:100dvh", css)
        self.assertIn("height:100% !important; min-height:0", css)
        self.assertIn("scrollbar-gutter:auto", css)
        self.assertIn(".st-key-lass-layout-resizer,", css)
        self.assertIn(":has(.st-key-close-library-panel)", css)
        self.assertIn("flex:0 0 3.5rem !important", css)
        self.assertIn("flex-wrap:nowrap !important", css)

    def test_resizer_persists_each_layout_and_supports_keyboard(self) -> None:
        html = (
            Path(__file__).parents[1]
            / "src/ui/layout_component/frontend/index.html"
        ).read_text(encoding="utf-8")
        self.assertNotIn("lass-panel-widths", html)
        self.assertIn('event.key !== "ArrowLeft"', html)
        self.assertIn("installPdfKeyboardNavigation", html)
        self.assertIn("suppressHoverText", html)
        self.assertIn('removeAttribute("title")', html)
        self.assertIn('input, textarea, select, [contenteditable="true"]', html)
        self.assertIn('event.key !== "ArrowLeft" && event.key !== "ArrowRight"', html)
        self.assertIn('button:not(:disabled)', html)
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
        self.assertIn('contentLayout === "side" ? width / page.naturalWidth', html)
        self.assertIn('height / page.naturalHeight', html)
        self.assertIn('pdfResizeObserver.observe(stage)', html)
        self.assertIn('pdfDomObserver.observe(parentDocument.body', html)
        self.assertIn('stage.addEventListener("wheel"', html)
        self.assertNotIn('stage.setPointerCapture(event.pointerId)', html)
        self.assertNotIn('stage.scrollLeft = drag.left', html)

    def test_pdf_viewer_stays_bounded_and_zoomable(self) -> None:
        css = (
            Path(__file__).parents[1] / "src/ui/components.py"
        ).read_text(encoding="utf-8")
        app = (Path(__file__).parents[1] / "src/ui/app.py").read_text(encoding="utf-8")
        self.assertNotIn('padding-left:calc(clamp(17rem,22vw,20rem) + .75rem)', css)
        self.assertNotIn('padding-right:calc(clamp(21rem,29vw,27rem) + .75rem)', css)
        self.assertNotIn('z-index:1; width:auto !important;', css)
        self.assertIn('position:absolute !important; inset:0; width:100% !important', css)
        self.assertIn('right:auto; z-index:60; width:var(--lass-sidebar-width,clamp(17rem,22vw,20rem))', css)
        self.assertIn('left:auto; z-index:55; width:var(--lass-sidebar-width,clamp(21rem,29vw,27rem))', css)
        self.assertIn('flex:0 0 6.5rem !important; width:6.5rem !important', css)
        self.assertIn('button[data-testid="stBaseButton-primary"]', css)
        self.assertIn('background:var(--accent-soft) !important', css)
        self.assertIn('st.columns([1, 1.5, 1], gap="small")', app)
        self.assertIn('type="primary" if fit else "secondary"', app)
        self.assertIn('.stMarkdown:has(.lass-pdf-stage) > div', css)
        self.assertIn('box-sizing:border-box; width:100%; height:100%', css)
        self.assertIn('.st-key-lass-document-pane:has(.lass-pdf-stage)', css)
        self.assertIn('padding:0; overflow:hidden;', css)
        self.assertIn('margin:0 !important; overflow:hidden;', css)
        self.assertIn('min-height:0; padding:0; overflow:hidden;', css)
        self.assertIn('max-width:none !important; max-height:none !important', css)


    def test_panel_open_close_controls_share_icon_size_and_hover_treatment(self) -> None:
        css = (Path(__file__).parents[1] / "src/ui/components.py").read_text(encoding="utf-8")
        for key in (
            "toggle-library-panel", "close-library-panel",
            "toggle-right-panel", "close-right-panel",
        ):
            self.assertIn(f'.st-key-{key} button [data-testid="stIconMaterial"]', css)
            self.assertIn(f'.st-key-{key} button:hover', css)
        self.assertIn('font-size:1.125rem !important; line-height:1 !important;', css)
        self.assertIn('border-radius:.5rem !important; background:transparent !important;', css)


    def test_document_upload_is_dialog_without_visible_upload_button(self) -> None:
        root = Path(__file__).parents[1]
        app = (root / "src/ui/app.py").read_text(encoding="utf-8")
        css = (root / "src/ui/components.py").read_text(encoding="utf-8")
        self.assertIn('@st.dialog(\n    "PDF · PPT 추가"', app)
        self.assertEqual(
            app.count('_render_browser_upload(repository, note, settings, kind="document")'),
            1,
        )
        self.assertIn(
            '[role="dialog"] [data-testid="stFileUploaderDropzone"][aria-label="PDF · PowerPoint"] > span',
            css,
        )
        self.assertIn('clip-path:inset(50%) !important;', css)

    def test_live_recorder_is_compact_and_has_no_waveform_label(self) -> None:
        html = (
            Path(__file__).parents[1]
            / "src/ui/live_recorder_component/frontend/index.html"
        ).read_text(encoding="utf-8")

        self.assertNotIn("Waveform active", html)
        self.assertIn("height:118px", html)
        self.assertIn("height:118", html)
        self.assertIn('id="elapsed"', html)

    def test_memo_editor_formats_and_auto_saves(self) -> None:
        html = (
            Path(__file__).parents[1]
            / "src/ui/editable_component/frontend/index.html"
        ).read_text(encoding="utf-8")
        self.assertIn('post("save_memo"', html)
        self.assertIn("}, 600)", html)
        self.assertIn('field.contentEditable = "true"', html)
        self.assertIn("memoEditorContent", html)
        self.assertIn("memoToMarkdown", html)
        self.assertIn('command("bold")', html)
        self.assertIn('tool("A−", "선택 글자 작게"', html)
        self.assertIn('tool("A+", "선택 글자 크게"', html)
        self.assertIn('document.execCommand("fontSize"', html)
        self.assertIn('if (node.matches("font[size]"))', html)
        self.assertIn('const view = markdownView(text, true)', html)
        self.assertIn('pane.getBoundingClientRect().bottom - frame.getBoundingClientRect().top', html)
        self.assertIn('command("insertUnorderedList")', html)
        self.assertIn('command("italic")', html)
        self.assertIn('command("underline")', html)
        self.assertIn('if (underline && !activeMarks.has("underline"))', html)
        self.assertIn('if (bold && !activeMarks.has("bold"))', html)
        self.assertIn('if (italic && !activeMarks.has("italic"))', html)
        self.assertIn('formatted = `<strong>${formatted}</strong>`', html)
        self.assertIn('formatted = `<em>${formatted}</em>`', html)
        self.assertIn('font|u|strong|em', html)
        self.assertIn('token.startsWith("****")', html)
        self.assertIn('token.startsWith("***")', html)
        self.assertNotIn('className = "memo-preview"', html)
        self.assertIn('tool("B", "굵게"', html)
        self.assertIn('tool("•", "글머리 목록"', html)
        self.assertIn('tool("I", "기울이기"', html)
        self.assertIn('tool("U", "밑줄"', html)
        self.assertNotIn('tool("☑", "체크 목록"', html)
        self.assertNotIn('tool("↗", "링크"', html)

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
