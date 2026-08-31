"""Prompt-aware summary generation."""

from __future__ import annotations

from .model_loader import OllamaClient
from .prompt_manager import PromptTemplate


class Summarizer:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    def summarize(self, context: str, template: PromptTemplate) -> str:
        if not context.strip():
            raise ValueError("Summary context must not be empty")
        return self.client.chat(template.system_prompt, template.render(context))
