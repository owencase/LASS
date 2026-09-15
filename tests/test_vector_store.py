import tempfile
import unittest
from pathlib import Path

from src.rag import SQLiteVectorStore, TextChunk


class SQLiteVectorStoreTests(unittest.TestCase):
    def test_search_and_collection_delete_use_one_database_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database_path = root / "vectors.db"
            store = SQLiteVectorStore(database_path, collection_name="lecture-1")
            chunks = [
                TextChunk(0, "사과에 대한 내용", 0, 10),
                TextChunk(1, "자동차에 대한 내용", 10, 20),
            ]
            store.upsert(
                chunks,
                [[1.0, 0.0], [0.0, 1.0]],
                fingerprint="content-v1",
            )

            results = store.search([0.9, 0.1], limit=1)

            self.assertEqual(results[0].text, "사과에 대한 내용")
            self.assertEqual(store.get_fingerprint(), "content-v1")
            self.assertEqual([path.name for path in root.iterdir()], ["vectors.db"])
            self.assertTrue(
                SQLiteVectorStore.delete_collection(database_path, "lecture-1")
            )
            self.assertFalse(
                SQLiteVectorStore.delete_collection(database_path, "lecture-1")
            )

    def test_streaming_upsert_accepts_generators_and_reports_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = SQLiteVectorStore(
                Path(temporary_directory) / "vectors.db",
                collection_name="streamed",
            )
            chunks = (TextChunk(index, f"조각 {index}", index, index + 1) for index in range(5))
            embeddings = ([float(index), 1.0] for index in range(5))

            store.upsert_stream(chunks, embeddings, fingerprint="stream-v1", batch_size=2)

            self.assertEqual(store.count(), 5)
            self.assertEqual(store.get_fingerprint(), "stream-v1")


if __name__ == "__main__":
    unittest.main()
