import unittest

from src.llm.prompt_manager import PromptTemplate


class PromptTemplateTests(unittest.TestCase):
    def test_context_is_rendered_without_formatting_other_braces(self) -> None:
        template = PromptTemplate(
            key="test",
            name="Test",
            description="Test prompt",
            system_prompt="System",
            user_prompt="Context: {context}\nJSON example: {\"key\": true}",
        )

        rendered = template.render("transcript")

        self.assertIn("Context: transcript", rendered)
        self.assertIn('{"key": true}', rendered)

    def test_missing_context_placeholder_is_rejected(self) -> None:
        template = PromptTemplate("test", "Test", "Test", "System", "No placeholder")
        with self.assertRaises(ValueError):
            template.render("transcript")


if __name__ == "__main__":
    unittest.main()
