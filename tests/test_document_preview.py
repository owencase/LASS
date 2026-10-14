import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pymupdf
from PIL import Image
from src.document_preview import preview_page
from src.pdf_utils import extract_pptx_text


class DocumentPreviewTests(unittest.TestCase):
    def test_large_pdf_page_is_bounded_and_reuses_disk_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "large.pdf"
            with pymupdf.open() as pdf:
                pdf.new_page(width=20000, height=20000)
                pdf.new_page(width=400, height=300)
                pdf.save(source)
            image, count, basic = preview_page(source, folder / "cache", 1)
            with Image.open(image) as thumbnail:
                self.assertLessEqual(max(thumbnail.size), 1600)
            self.assertEqual(count, 2)
            self.assertFalse(basic)
            with patch("pymupdf.open", side_effect=AssertionError("cache miss")):
                self.assertEqual(preview_page(source, folder / "cache", 1), (image, 2, False))
            with self.assertRaises(ValueError):
                preview_page(source, folder / "cache", 3)

    def test_pptx_basic_preview_respects_slide_order_and_escapes_text(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "slides.pptx"
            ns = 'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
            def slide(text):
                return f'<p:sld {ns}><p:cSld><p:spTree><p:sp><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="10000000" cy="6000000"/></a:xfrm></p:spPr><p:txBody><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>'
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("ppt/slides/slide1.xml", slide("First"))
                archive.writestr("ppt/slides/slide2.xml", slide("Second &lt;script&gt;"))
                archive.writestr("ppt/presentation.xml", f'<p:presentation {ns} xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId r:id="r2"/><p:sldId r:id="r1"/></p:sldIdLst><p:sldSz cx="12192000" cy="6858000"/></p:presentation>')
                archive.writestr("ppt/_rels/presentation.xml.rels", '<Relationships><Relationship Id="r1" Target="slides/slide1.xml"/><Relationship Id="r2" Target="slides/slide2.xml"/></Relationships>')
            with patch("src.document_preview.convert_presentation", side_effect=RuntimeError("no Office")) as convert:
                image, count, basic = preview_page(source, folder / "cache", 1)
                self.assertTrue(basic)
                self.assertEqual(count, 2)
                content = image.read_text(encoding="utf-8")
                self.assertIn("Second &lt;script&gt;", content)
                self.assertNotIn("<script>", content)
                preview_page(source, folder / "cache", 2)
                convert.assert_called_once()
            text = extract_pptx_text(source).text
            self.assertLess(text.index("Second"), text.index("First"))


if __name__ == "__main__":
    unittest.main()
