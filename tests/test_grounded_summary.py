"""Regression checks for the workflow restored from Setup 1.3.0."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from src.llm import PromptManager, Summarizer


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.template = PromptManager(Path(__file__).parents[1] / "prompts").load("lecture_detail")

    def test_single_page_returns_model_output_without_reformatting(self):
        raw = "1. 강의 주제와 학습 목표\n원문 그대로"
        client = SimpleNamespace(chat=Mock(return_value=raw))
        tokens = []
        result = Summarizer(client).summarize_pages(["", "원문"], self.template, on_token=tokens.append)
        self.assertEqual(result, raw)
        client.chat.assert_called_once_with(self.template.system_prompt, self.template.render("원문"), on_token=tokens.append)

    def test_pages_are_read_serially_then_notes_are_reduced_in_groups_of_four(self):
        read = []
        def pages():
            for number in range(17):
                read.append(number)
                yield f"원문 {number}"

        def chat(system, prompt, **kwargs):
            if "[현재 자료 조각]" in prompt:
                number = int(prompt.rsplit("원문 ", 1)[1])
                self.assertEqual(len(read), max(2, number + 1))
                return f"근거 {number}"
            if "병합하는 편집자" in system:
                self.assertLessEqual(prompt.count("근거 "), 4)
                return "병합 메모"
            return "최종 결과"

        client = SimpleNamespace(chat=Mock(side_effect=chat))
        self.assertEqual(Summarizer(client).summarize_pages(pages(), self.template), "최종 결과")
        self.assertEqual(read, list(range(17)))
        self.assertEqual(client.chat.call_count, 25)

    def test_empty_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            Summarizer(SimpleNamespace()).summarize_pages(["", "  "], self.template)
