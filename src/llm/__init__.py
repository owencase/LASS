"""Local language-model components."""

from .model_loader import OllamaClient
from .prompt_manager import PromptManager, PromptTemplate
from .summarizer import Summarizer

__all__ = ["OllamaClient", "PromptManager", "PromptTemplate", "Summarizer"]
