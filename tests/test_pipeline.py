import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.config import Settings
from src.pdf_utils import ExtractedPDF
from src.pipeline import LASSPipeline
from src.rag import SQLiteTextStore


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
            self.assertIsNone(pipeline._embedder)

    def test_resummarize_includes_saved_memo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                project_root=root,
                data_dir=root / "data",
                prompt_dir=Path(__file__).parents[1] / "prompts",
            )
            pipeline = LASSPipeline(settings)
            pipeline.llm_client.chat = Mock(return_value="새 요약")

            result = pipeline.resummarize(
                "수정된 전체 녹취입니다.",
                notes="사용자가 강조한 메모입니다.",
                prompt_key="default_summary",
            )

            self.assertEqual(result.summary, "새 요약")
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
            pipeline.llm_client.chat = Mock(side_effect=["근거 1", "근거 2", "최종 요약"])
            result = pipeline.resummarize_pages(pages())
            self.assertEqual(result.summary, "최종 요약")
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
                "핵심은 무엇인가요?",
                [{"role": "user", "content": "이전 질문"}],
            )

            self.assertEqual(answer, "폴더 답변")
            messages = pipeline.llm_client.chat_messages.call_args.args[1]
            self.assertEqual(messages[0]["content"], "이전 질문")
            self.assertIn("RAG 노트", messages[-1]["content"])
            self.assertIn("복습 메모", messages[-1]["content"])
            self.assertIn("벡터 검색 전체 녹취", messages[-1]["content"])

    def test_folder_chat_condenses_large_short_rag_results_page_by_page(self) -> None:
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
                "folder-1", "AI 공부", [note], "핵심은?", top_k=8
            )

            self.assertEqual(answer, "폴더 답변")
            self.assertGreater(pipeline.llm_client.chat.call_count, 1)
            final_context = pipeline.llm_client.chat_messages.call_args.args[1][-1]["content"]
            self.assertIn("압축된 질문 근거", final_context)

    def test_pdf_processing_extracts_text_and_summarizes_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = Settings(
                project_root=root,
                data_dir=root / "data",
                prompt_dir=Path(__file__).parents[1] / "prompts",
            )
            pipeline = LASSPipeline(settings)
            pipeline.llm_client.chat = Mock(return_value="PDF 요약")
            pdf_path = root / "large.pdf"
            pdf_path.write_bytes(b"%PDF-placeholder")

            with unittest.mock.patch(
                "src.pipeline.extract_pdf_text",
                return_value=ExtractedPDF("## 1페이지\n본문", 1, 1),
            ):
                result = pipeline.process_pdf(pdf_path)

            self.assertEqual(result.summary, "PDF 요약")
            self.assertEqual(result.page_count, 1)
            self.assertIn("본문", result.transcript)

    def test_multi_page_summary_reads_pages_sequentially_before_final_synthesis(self) -> None:
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
            pipeline.llm_client.chat = Mock(side_effect=["근거 1", "근거 2", "최종 요약"])

            result = pipeline.resummarize("첫 자료입니다. " * 4, top_k=8)

            self.assertEqual(result.summary, "최종 요약")
            self.assertEqual(pipeline.llm_client.chat.call_count, 3)
            first_call = pipeline.llm_client.chat.call_args_list[0]
            self.assertEqual(first_call.kwargs["max_new_tokens"], 128)
            self.assertIn("근거 1", pipeline.llm_client.chat.call_args_list[-1].args[1])

    def test_many_pages_are_hierarchically_reduced_in_groups_of_four(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pipeline = LASSPipeline(Settings(
                project_root=root,
                data_dir=root / "data",
                prompt_dir=Path(__file__).parents[1] / "prompts",
                chunk_size=20,
                chunk_overlap=2,
            ))

            def answer(system_prompt, _user_prompt, **_kwargs):
                if "손실 적게 병합" in system_prompt:
                    return "압축 근거"
                if "메모리 절약 방식" in system_prompt:
                    return "조각 근거"
                return "최종 요약"

            pipeline.llm_client.chat = Mock(side_effect=answer)
            result = pipeline.resummarize("서로 다른 긴 자료입니다. " * 30)

            self.assertEqual(result.summary, "최종 요약")
            reduction_calls = [
                call for call in pipeline.llm_client.chat.call_args_list
                if "손실 적게 병합" in call.args[0]
            ]
            self.assertTrue(reduction_calls)
            self.assertTrue(all(call.kwargs["max_new_tokens"] == 160 for call in reduction_calls))
            self.assertLessEqual(
                pipeline.llm_client.chat.call_args_list[-1].args[1].count("압축 근거"),
                4,
            )


if __name__ == "__main__":
    unittest.main()
