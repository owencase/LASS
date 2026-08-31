import tempfile
import unittest
from pathlib import Path

from src.config import Settings
from src.pipeline import LASSPipeline
from src.stt import Transcription, TranscriptionSegment


class PipelineTests(unittest.TestCase):
    def test_short_transcript_context_does_not_require_vector_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                project_root=root,
                data_dir=root / "data",
                prompt_dir=root / "prompts",
                chunk_size=20,
                chunk_overlap=2,
            )
            pipeline = LASSPipeline(settings)

            context = pipeline._select_context("짧은 테스트 스크립트입니다.", "query", top_k=8)

            self.assertEqual(context, "짧은 테스트 스크립트입니다.")

    def test_report_contains_summary_and_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                project_root=root,
                data_dir=root / "data",
                prompt_dir=root / "prompts",
            )
            settings.ensure_directories()
            pipeline = LASSPipeline(settings)
            transcription = Transcription(
                text="테스트 발화",
                language="ko",
                duration=3.0,
                segments=[TranscriptionSegment(1.0, 3.0, "테스트 발화")],
            )

            output = pipeline._write_report(
                root / "sample.mp3", "요약 결과", transcription, "기본 요약"
            )
            content = output.read_text(encoding="utf-8")

            self.assertIn("요약 결과", content)
            self.assertIn("`00:00:01` 테스트 발화", content)


if __name__ == "__main__":
    unittest.main()
