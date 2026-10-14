import io
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from src.pdf_utils import extract_pdf_page


class PDFUtilsTests(unittest.TestCase):
    def test_embedded_text_pages_do_not_force_ocr_for_decorative_images(self) -> None:
        source = (Path(__file__).parents[1] / 'src/pdf_utils.py').read_text(
            encoding='utf-8'
        )
        self.assertIn('needs_ocr = _visible_characters(native) < 40', source)
        self.assertNotIn('has_images and _visible_characters(native)', source)

    def test_extracts_only_the_requested_page_for_slide_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "slides.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=400, height=300)
            writer.add_blank_page(width=500, height=350)
            writer.add_blank_page(width=600, height=400)
            with source.open("wb") as output:
                writer.write(output)

            page_data, page_count = extract_pdf_page(source, 2)
            preview = PdfReader(io.BytesIO(page_data))

            self.assertEqual(page_count, 3)
            self.assertEqual(len(preview.pages), 1)
            self.assertEqual(float(preview.pages[0].mediabox.width), 500)

    def test_rejects_page_outside_document(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=400, height=300)
        source = io.BytesIO()
        writer.write(source)

        with self.assertRaisesRegex(ValueError, "1에서 1 사이"):
            extract_pdf_page(source.getvalue(), 2)


if __name__ == "__main__":
    unittest.main()
