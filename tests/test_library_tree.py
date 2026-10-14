import unittest
from pathlib import Path


class LibraryTreeComponentTests(unittest.TestCase):
    def test_context_menu_exposes_folder_and_note_actions(self) -> None:
        component = (
            Path(__file__).parents[1]
            / "src"
            / "ui"
            / "library_component"
            / "frontend"
            / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn('addEventListener("contextmenu"', component)
        for action in (
            "create_folder",
            "create_note",
            "rename_folder",
            "delete_folder",
            "rename_note",
            "move_note",
            "delete_note",
        ):
            self.assertIn(f'post("{action}"', component)
        self.assertIn("노트 삭제", component)
        self.assertIn("showCreateMenu", component)
        self.assertIn('folder_id: item.id', component)
        self.assertIn("폴더에 만들 노트 이름", component)
        self.assertIn('tree.addEventListener("contextmenu"', component)
        self.assertIn("빈 공간을 우클릭해 폴더나 노트를 만드세요", component)
        self.assertIn("안의 노트는 미분류로 이동합니다", component)
        self.assertIn('post("select_folder"', component)
        self.assertIn('row.addEventListener("click", toggleFolder)', component)
        self.assertIn('row.setAttribute("aria-expanded"', component)
        self.assertNotIn('chevron.addEventListener("click"', component)
        self.assertIn('class="chevron-closed"', component)
        self.assertIn('class="chevron-open"', component)
        self.assertNotIn('chevron.textContent = "▶"', component)
        self.assertIn('icon.innerHTML = folderIcon', component)
        self.assertIn('icon.innerHTML = noteIcon', component)
        self.assertIn('min-height:40px; margin-bottom:2px', component)
        self.assertNotIn('icon.textContent = "▰"', component)
        self.assertNotIn('icon.textContent = "▤"', component)

        app_source = (
            Path(__file__).parents[1] / "src" / "ui" / "app.py"
        ).read_text(encoding="utf-8")
        self.assertIn('with st.container(key="lass-library-content")', app_source)
        self.assertIn('render_pane_title("폴더 및 노트", "목록")', app_source)
        self.assertIn("_handle_library_event(event, repository, settings)", app_source)
        self.assertIn('elif action == "create_note":', app_source)
        self.assertIn('elif action == "create_folder":', app_source)
        self.assertNotIn('"＋ 노트 추가"', app_source)
        self.assertNotIn('"＋ 새 노트"', app_source)
        self.assertNotIn('"＋ 새 폴더"', app_source)

    def test_pdf_viewer_mounts_only_the_selected_page(self) -> None:
        app_source = (
            Path(__file__).parents[1] / "src" / "ui" / "app.py"
        ).read_text(encoding="utf-8")

        preview_source = (Path(__file__).parents[1] / "src/document_preview.py").read_text(encoding="utf-8")
        self.assertIn("document.load_page(page_number - 1)", preview_source)
        self.assertNotIn('src="data:image/png;base64,', app_source)
        viewer = (Path(__file__).parents[1] / "src/ui/document_component/frontend/index.html").read_text(encoding="utf-8")
        self.assertIn('render_document_viewer(', app_source)
        self.assertIn('await next.decode()', viewer)
        self.assertLess(viewer.index('await next.decode()'), viewer.index('stage.replaceChildren(next)'))
        self.assertIn('with st.container(key="lass-document-memo-layout")', app_source)
        self.assertIn('if selected_id != source["id"]:', app_source)
        self.assertIn('st.session_state[active_key] = current_id', app_source)
        self.assertIn('with st.container(key="lass-document-empty-viewer")', app_source)
        self.assertNotIn("st.pdf(", app_source)
        self.assertIn("pdf-previous-", app_source)
        self.assertIn("pdf-next-", app_source)

    def test_inline_editor_uses_double_click_for_summary_and_transcript(self) -> None:
        editor = (
            Path(__file__).parents[1]
            / "src"
            / "ui"
            / "editable_component"
            / "frontend"
            / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn('addEventListener("dblclick"', editor)
        self.assertIn('post("save_summary"', editor)
        self.assertIn('post("save_segment"', editor)

    def test_inline_editor_preserves_edit_state_and_renders_markdown_blocks(self) -> None:
        editor = (
            Path(__file__).parents[1]
            / "src"
            / "ui"
            / "editable_component"
            / "frontend"
            / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn("nextSignature === contentSignature", editor)
        self.assertIn('document.createElement("table")', editor)
        self.assertIn('document.createElement("pre")', editor)
        self.assertIn('document.createElement("blockquote")', editor)
        self.assertIn('body.ready { opacity:1; }', editor)


if __name__ == "__main__":
    unittest.main()
