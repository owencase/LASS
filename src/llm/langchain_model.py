"""LangChain chat interface for the single offline EXAONE runtime."""

from typing import Any, Iterator
from contextlib import closing

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field
from .text_input import text_input


class EXAONEChatModel(BaseChatModel):
    client: Any = Field(exclude=True)

    @property
    def _llm_type(self) -> str:
        return "local-exaone-4.0-1.2b"

    def _stream(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        if stop:
            raise ValueError("EXAONE completes on EOS; custom stop strings are unsupported")
        conversation = [
            {"role": {"human": "user", "ai": "assistant", "system": "system"}[message.type],
             "content": text_input(message.content)}
            for message in messages
        ]
        with closing(self.client.stream_messages(conversation)) as stream:
            for text in stream:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager:
                    run_manager.on_llm_new_token(text, chunk=chunk)
                yield chunk

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = "".join(chunk.text for chunk in self._stream(messages, stop, run_manager, **kwargs))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
