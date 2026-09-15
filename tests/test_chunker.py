import unittest

from src.rag.chunker import iter_text_chunks, split_text


class SplitTextTests(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self) -> None:
        self.assertEqual(split_text("   "), [])

    def test_long_text_is_split_with_size_limit(self) -> None:
        text = "첫 번째 문장입니다. " * 30
        chunks = split_text(text, chunk_size=80, overlap=10)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.text for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 80 for chunk in chunks))
        self.assertEqual([chunk.index for chunk in chunks], list(range(len(chunks))))

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_text("text", chunk_size=10, overlap=10)

    def test_iterator_yields_the_same_chunks_without_a_materialized_list(self) -> None:
        text = "가나다라마바사. " * 20
        streamed = iter_text_chunks(text, chunk_size=45, overlap=5)

        self.assertFalse(isinstance(streamed, list))
        self.assertEqual(
            list(streamed),
            split_text(text, chunk_size=45, overlap=5),
        )


if __name__ == "__main__":
    unittest.main()
