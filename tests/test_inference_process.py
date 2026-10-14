import tempfile
import unittest
from pathlib import Path

from src.llm.model_loader import LocalEXAONEClient
from src.runtime import memory_managed


class InferenceProcessTests(unittest.TestCase):
    def test_failed_model_load_exits_worker_and_reports_error(self):
        with tempfile.TemporaryDirectory() as directory:
            client = LocalEXAONEClient(Path(directory), isolated=True)
            try:
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    list(client.stream_messages([{"role": "user", "content": "test"}]))
                self.assertIsNone(client._process)
            finally:
                client.unload()

    def test_default_releases_after_success_and_failure_but_resident_mode_keeps_model(self):
        class Pipeline:
            llm_client = type("Client", (), {"keep_loaded": False})()
            unload_count = 0

            def unload(self):
                self.unload_count += 1

            @memory_managed
            def run(self, fail=False):
                if fail:
                    raise RuntimeError("failed")
                return "done"

        pipeline = Pipeline()
        self.assertEqual(pipeline.run(), "done")
        with self.assertRaises(RuntimeError):
            pipeline.run(fail=True)
        self.assertEqual(pipeline.unload_count, 2)
        pipeline.llm_client.keep_loaded = True
        pipeline.run()
        self.assertEqual(pipeline.unload_count, 2)
