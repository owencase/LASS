import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.config import Settings
from src.pdf_utils import ExtractedPDF
from src.pipeline import LASSPipeline
from src.rag import SQLiteTextStore


def structured(text):
    return "## 주제\n" + text + "\n\n## 학습 목표\n목표\n\n## 개념\n핵심\n\n## 상세 내용\n### 1. 주제\n상세\n\n## 정리\n정리\n\n## 중요 포인트\n주의"


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

    def test_collection_name_is_stable_for_the_same_transcript(self) -> None:
        first = LASSPipeline._collection_name("동일한 강의 녹취")
        second = LASSPipeline._collection_name("동일한 강의 녹취")

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("lass-"))

    def test_long_context_streams_to_sparse_disk_index_and_reuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                project_root=root,
                data_dir=root / "data",
                prompt_dir=Path(__file__).parents[1] / "prompts",
                chunk_size=24,
                chunk_overlap=2,
            )
            pipeline = LASSPipeline(settings)
            transcript = "서로 다른 긴 자료 문장입니다. " * 20

            first, collection = pipeline._select_context_and_collection(
                transcript, "핵심", top_k=2, collection_key="note-1"
            )
            second, reused_collection = pipeline._select_context_and_collection(
                transcript, "핵심", top_k=2, collection_key="note-1"
            )

            self.assertEqual(first.page_count, 2)
            self.assertEqual(second.page_count, 2)
            self.assertEqual(collection, reused_collection)
            store = SQLiteTextStore(settings.vector_database_path, collection)
            self.assertGreater(store.count(), 2)
            self.assertFalse(hasattr(pipeline, "_embedder"))

    def test_resummarize_includes_saved_memo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                project_root=root,
                data_dir=root / "data",
                prompt_dir=Path(__file__).parents[1] / "prompts",
            )
            pipeline = LASSPipeline(settings)
            pipeline.llm_client.chat = Mock(return_value=structured("수정된 전체 녹취입니다. 사용자가 강조한 메모입니다."))

            result = pipeline.resummarize(
                "수정된 전체 녹취입니다.",
                notes="사용자가 강조한 메모입니다.",
                prompt_key="lecture_detail",
            )

            self.assertIn("수정된 전체 녹취입니다.", result.summary)
            self.assertIn("사용자가 강조한 메모입니다.", result.summary)
            prompt = pipeline.llm_client.chat.call_args.args[1]
            self.assertIn("수정된 전체 녹취", prompt)
            self.assertIn("사용자가 작성한 메모", prompt)
            self.assertIn("강조한 메모", prompt)

    def test_streamed_resummary_consumes_disk_pages_without_joining_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pipeline = LASSPipeline(Settings(
                project_root=root,
                data_dir=root / "data",
                prompt_dir=Path(__file__).parents[1] / "prompts",
            ))
            seen = []
            def pages():
                for value in ("첫 조각", "둘째 조각"):
                    seen.append(value)
                    yield value
            pipeline.llm_client.chat = Mock(return_value=structured("첫 조각과 둘째 조각의 첫 자료입니다."))
            result = pipeline.resummarize_pages(pages())
            self.assertIn("첫 조각", result.summary)
            self.assertIn("둘째 조각", result.summary)
            self.assertEqual(seen, ["첫 조각", "둘째 조각"])

    def test_folder_chat_uses_summary_memo_transcript_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                project_root=root,
                data_dir=root / "data",
                prompt_dir=Path(__file__).parents[1] / "prompts",
            )
            pipeline = LASSPipeline(settings)
            pipeline.llm_client.chat_messages = Mock(return_value="폴더 답변")
            note = SimpleNamespace(
                title="RAG 노트",
                summary="검색 핵심 요약",
                notes="복습 메모",
                transcript="벡터 검색 전체 녹취",
            )

            answer = pipeline.answer_folder_question(
                "folder-1",
                "AI 공부",
                [note],
                "검색 핵심 벡터 복습은 무엇인가요?",
                [{"role": "user", "content": "이전 질문"}],
            )

            self.assertTrue(answer.startswith("폴더 답변"))
            messages = pipeline.llm_client.chat_messages.call_args.args[1]
            self.assertEqual(messages[0]["content"], "이전 질문")
            self.assertIn("RAG 노트", messages[-1]["content"])
            self.assertIn("복습 메모", messages[-1]["content"])
            self.assertIn("벡터 검색 전체 녹취", messages[-1]["content"])

    def test_folder_chat_uses_bounded_original_evidence_in_one_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pipeline = LASSPipeline(Settings(
                project_root=root,
                data_dir=root / "data",
                prompt_dir=Path(__file__).parents[1] / "prompts",
            ))
            pipeline.llm_client.chat = Mock(return_value="압축된 질문 근거")
            pipeline.llm_client.chat_messages = Mock(return_value="폴더 답변")
            note = SimpleNamespace(
                title="긴 노트",
                summary="메모리 절약 근거 " * 180,
                notes="",
                transcript="",
            )

            answer = pipeline.answer_folder_question(
                "folder-1", "AI 공부", [note], "메모리 절약 근거는?", top_k=8
            )

            self.assertTrue(answer.startswith("폴더 답변"))
            pipeline.llm_client.chat.assert_not_called()
            pipeline.llm_client.chat_messages.assert_called_once()
            final_context = pipeline.llm_client.chat_messages.call_args.args[1][-1]["content"]
            self.assertIn("메모리 절약 근거", final_context)
            self.assertIn("출처:", final_context)
            self.assertLess(len(final_context), 1500)

    def test_pdf_processing_extracts_text_and_summarizes_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                project_root=root,
                data_dir=root / "data",
                prompt_dir=Path(__file__).parents[1] / "prompts",
            )
            pipeline = LASSPipeline(settings)
            pipeline.llm_client.chat = Mock(return_value=structured("# 문서 요약\n\n## 개요\n본문 요약 (1페이지)"))
            pdf_path = root / "large.pdf"
            pdf_path.write_bytes(b"%PDF-placeholder")

            with unittest.mock.patch(
                "src.pipeline.extract_pdf_text",
                return_value=ExtractedPDF("## 1페이지\n본문", 1, 1),
            ):
                result = pipeline.process_pdf(pdf_path)

            self.assertIn("본문 요약", result.summary)
            self.assertIn("1페이지", result.summary)
            self.assertEqual(result.page_count, 1)
            self.assertIn("본문", result.transcript)

    def test_multi_page_summary_synthesizes_all_source_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                project_root=root,
                data_dir=root / "data",
                prompt_dir=Path(__file__).parents[1] / "prompts",
                chunk_size=20,
                chunk_overlap=2,
            )
            pipeline = LASSPipeline(settings)
            pipeline.llm_client.chat = Mock(return_value=structured("첫 조각과 둘째 조각의 첫 자료입니다."))

            result = pipeline.resummarize("첫 자료입니다. " * 4, top_k=8)

            self.assertIn("첫 자료입니다.", result.summary)
            self.assertEqual(pipeline.llm_client.chat.call_count, pipeline._all_context("첫 자료입니다. " * 4).page_count + 1)
            first_call = pipeline.llm_client.chat.call_args_list[0]
            self.assertNotIn("max_new_tokens", first_call.kwargs)
            self.assertNotIn("이전 자료까지의 요약본", pipeline.llm_client.chat.call_args_list[-1].args[1])

    def test_large_source_uses_130_hierarchical_disk_reduction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = LASSPipeline(Settings(
                project_root=root, data_dir=root / "data",
                prompt_dir=Path(__file__).parents[1] / "prompts", chunk_size=20, chunk_overlap=2,
            ))
            pipeline.llm_client.chat = Mock(return_value=structured("통합 요약본"))
            result = pipeline.resummarize("서로 다른 긴 자료입니다. " * 30)
            self.assertEqual(result.summary, structured("통합 요약본"))
            calls = pipeline.llm_client.chat.call_args_list
            page_count = pipeline._all_context("서로 다른 긴 자료입니다. " * 30).page_count
            expected_calls = page_count + 1
            while page_count > 4:
                page_count = (page_count + 3) // 4
                expected_calls += page_count
            self.assertEqual(len(calls), expected_calls)
            self.assertIn("[현재 자료 조각]", calls[0].args[1])
            self.assertTrue(any("병합하는 편집자" in call.args[0] for call in calls))
            self.assertIn("통합 요약본", calls[-1].args[1])
            self.assertTrue(all(len(call.args[1]) < 1920 for call in calls))


if __name__ == "__main__":
    unittest.main()
