import tempfile
import unittest
from pathlib import Path

from src.rag import SQLiteTextStore, TextChunk


class SQLiteTextStoreTests(unittest.TestCase):
    def test_streaming_fts_search_reuse_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "search.db"
            store = SQLiteTextStore(database, "lecture-1")
            consumed = []

            def chunks():
                for index, text in enumerate(
                    ["회귀 분석의 기본 개념", "신경망 최적화 방법", "통계적 검정 절차"]
                ):
                    consumed.append(index)
                    yield TextChunk(index, text, index * 10, index * 10 + len(text))

            store.upsert_stream(chunks(), fingerprint="v1", batch_size=1)

            self.assertEqual(consumed, [0, 1, 2])
            self.assertEqual(store.count(), 3)
            self.assertEqual(store.get_fingerprint(), "v1")
            self.assertIn("신경망", store.search("신경망", limit=1)[0].text)
            self.assertTrue(SQLiteTextStore.delete_collection(database, "lecture-1"))
            self.assertEqual(store.count(), 0)


if __name__ == "__main__":
    unittest.main()
