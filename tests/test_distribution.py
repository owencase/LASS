import tempfile
import unittest
from pathlib import Path
from src.distribution import write_manifest, verify_bundle


class DistributionTests(unittest.TestCase):
    def test_missing_and_stale_assets_fail_build_and_self_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, bundle = Path(temporary) / "source", Path(temporary) / "bundle"
            for root in (source, bundle):
                for relative in ("src/ui/new.js", "prompts/test.yaml", ".streamlit/config.toml"):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("original", encoding="utf-8")
            write_manifest(source, bundle)
            self.assertTrue(verify_bundle(bundle))
            (bundle / "src/ui/new.js").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing or stale"):
                write_manifest(source, bundle)
            with self.assertRaisesRegex(RuntimeError, "missing or stale"):
                verify_bundle(bundle)
            (bundle / "src/ui/new.js").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing or stale"):
                verify_bundle(bundle)
