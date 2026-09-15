import unittest

from src.pdf_stream import parse_byte_range


class PDFStreamTests(unittest.TestCase):
    def test_parses_open_closed_and_suffix_ranges(self) -> None:
        self.assertEqual(parse_byte_range("bytes=10-19", 100), (10, 19))
        self.assertEqual(parse_byte_range("bytes=90-", 100), (90, 99))
        self.assertEqual(parse_byte_range("bytes=-8", 100), (92, 99))
        self.assertIsNone(parse_byte_range("", 100))

    def test_rejects_invalid_or_multi_ranges(self) -> None:
        for value in ("items=1-2", "bytes=100-101", "bytes=8-2", "bytes=1-2,4-5"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_byte_range(value, 100)


if __name__ == "__main__":
    unittest.main()
