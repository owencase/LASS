import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from src.llm.text_input import text_input
from src.llm.model_loader import LocalEXAONEClient


class TextInputTests(unittest.TestCase):
    def test_documents_blocks_and_unicode_are_normalized_without_object_repr(self):
        content = ["첫 문장", {"type": "text", "text": "둘째 문장"},
                   SimpleNamespace(page_content="셋째 문장"), ("넷째", "다섯째\ud800")]
        result = text_input(content)
        self.assertEqual(result, "첫 문장\n둘째 문장\n셋째 문장\n넷째\n다섯째?")
        with self.assertRaises(ValueError):
            text_input({"irrelevant": "not text"})
        with self.assertRaises(ValueError):
            text_input({"type": "image", "text": "not text"})

    def test_only_one_clean_string_reaches_tokenizer_even_for_long_blocks(self):
        from tests.gguf_fakes import Tokenizer
        tokenizer = Tokenizer()
        ids = LocalEXAONEClient(Path('.'), max_input_tokens=256)._bounded_input(tokenizer, [
            {'role': 'system', 'content': [{'type': 'text', 'text': '지시'}]},
            {'role': 'user', 'content': [{'text': '원문\ud800' * 1000}, {'text': '[질문] 핵심은?'}]},
        ])
        self.assertIn("지시", tokenizer.last)
        self.assertLessEqual(len(ids), 256)

    def test_invalid_input_fails_before_starting_model_worker(self):
        client = LocalEXAONEClient(Path('.'), isolated=True)
        client._stream_process = Mock()
        with self.assertRaises(ValueError):
            list(client.stream_messages([{'role': 'user', 'content': {'number': 1}}]))
        client._stream_process.assert_not_called()

    def test_langchain_blocks_are_flattened_at_adapter_boundary(self):
        from langchain_core.messages import HumanMessage
        from src.llm.langchain_model import EXAONEChatModel
        client = Mock()
        client.stream_messages.return_value = (text for text in ['답변'])
        reply = EXAONEChatModel(client=client).invoke([HumanMessage(content=[
            {'type': 'text', 'text': '첫 문장'}, {'type': 'text', 'text': '둘째 문장'},
        ])])
        self.assertEqual(reply.content, '답변')
        self.assertEqual(client.stream_messages.call_args.args[0][0]['content'], '첫 문장\n둘째 문장')

    def test_prompt_chain_preserves_roles_literal_braces_and_incremental_tokens(self):
        client = LocalEXAONEClient(Path('.'))
        chunks = []
        def stream(messages):
            self.assertEqual([message['role'] for message in messages], ['system', 'user', 'assistant', 'user'])
            self.assertEqual(messages[0]['content'], 'JSON {schema} 지시')
            yield '첫 토큰 '
            self.assertEqual(chunks, ['첫 토큰 '])
            yield '마지막 토큰'
        client.stream_messages = stream
        result = client.chat_messages('JSON {schema} 지시', [
            {'role':'user', 'content':'이전 질문'}, {'role':'assistant', 'content':'이전 답변'},
            {'role':'user', 'content':'현재 {질문}'},
        ], on_token=chunks.append)
        self.assertEqual(result, ''.join(chunks))
