import tempfile
import unittest
from threading import Thread, current_thread
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

from src.config import Settings
from src.pipeline import LASSPipeline
from src.runtime import resolve_device
from src.stt.transcriber import Transcription
from src.llm.model_loader import LocalEXAONEClient


class RuntimeMemoryTests(unittest.TestCase):
    def pipeline(self, root):
        return LASSPipeline(Settings(data_dir=Path(root), prompt_dir=Path(__file__).parents[1] / 'prompts'))

    def test_transcriber_and_embedding_are_unloaded_before_llm(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline = self.pipeline(root)
            stt, embedding = Mock(), Mock()
            stt.transcribe.return_value = Transcription('내용', 'ko', 1, [])
            def context(*args, **kwargs):
                stt.unload.assert_called_once()
                pipeline._embedder = embedding
                return '내용'
            pipeline._all_context = context
            def chat(*args):
                embedding.unload.assert_called_once()
                return '요약'
            pipeline.llm_client.chat = chat
            pipeline.llm_client.unload = Mock()
            with patch('src.pipeline.create_transcriber', return_value=stt):
                self.assertEqual(pipeline.process(Path(root) / 'audio.wav').summary, '요약')
            self.assertIsNone(pipeline._embedder)
            self.assertTrue(pipeline.llm_client.unload.called)

    def test_cleanup_also_runs_after_transcription_failure(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline = self.pipeline(root)
            stt = Mock()
            stt.transcribe.side_effect = RuntimeError('STT failed')
            with patch('src.pipeline.create_transcriber', return_value=stt):
                with self.assertRaisesRegex(RuntimeError, 'STT failed'):
                    pipeline.transcribe(Path(root) / 'audio.wav')
            stt.unload.assert_called_once()
            self.assertIsNone(pipeline.llm_client._model)

    def test_resolve_apple_gpu_and_report_unavailable_explicit_choice(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
        self.assertEqual(resolve_device(torch), 'mps')
        torch.backends.mps.is_available = lambda: False
        self.assertEqual(resolve_device(torch), 'cpu')
        with self.assertRaisesRegex(RuntimeError, 'Apple GPU'):
            resolve_device(torch, 'mps')

    def test_context_budget_keeps_system_and_latest_question(self):
        class Tokenizer:
            def apply_chat_template(self, conversation, **kwargs):
                self.conversation = conversation
                return SimpleNamespace(shape=(1, sum(len(m['content']) for m in conversation) + 20))
            def encode(self, value, **kwargs):
                return list(value)
            def decode(self, tokens):
                return ''.join(tokens)
        tokenizer = Tokenizer()
        client = LocalEXAONEClient(Path('.'), max_input_tokens=512)
        ids = client._bounded_input(tokenizer, [
            {'role':'system', 'content':'자료로만 답하기'},
            {'role':'user', 'content':'이전 질문' * 300},
            {'role':'assistant', 'content':'이전 답변' * 300},
            {'role':'user', 'content':'참고 자료' * 300 + '[질문] 핵심은?'}])
        self.assertLessEqual(ids.shape[-1], 512)
        self.assertEqual(tokenizer.conversation[0]['content'], '자료로만 답하기')
        self.assertIn('[질문] 핵심은?', tokenizer.conversation[-1]['content'])

    def test_mlx_context_budget_uses_token_lists_without_torch_tensors(self):
        class Tokenizer:
            def apply_chat_template(self, conversation, **kwargs):
                self.conversation = conversation
                return list(range(sum(len(m['content']) for m in conversation) + 20))
            def encode(self, value, **kwargs):
                return list(value)
            def decode(self, tokens):
                return ''.join(tokens)

        tokenizer = Tokenizer()
        client = LocalEXAONEClient(Path('.'), backend='mlx', max_input_tokens=512)
        ids = client._bounded_mlx_input(tokenizer, [
            {'role':'system', 'content':'자료로만 답하기'},
            {'role':'user', 'content':'이전 질문' * 300},
            {'role':'assistant', 'content':'이전 답변' * 300},
            {'role':'user', 'content':'참고 자료' * 300 + '[질문] 핵심은?'}])

        self.assertLessEqual(len(ids), 512)
        self.assertEqual(tokenizer.conversation[0]['content'], '자료로만 답하기')
        self.assertIn('[질문] 핵심은?', tokenizer.conversation[-1]['content'])

    def test_mlx_requests_from_new_threads_share_one_inference_thread(self):
        client = LocalEXAONEClient(Path('.'), backend='mlx')
        seen = []
        client._chat_mlx_on_thread = lambda conversation, **kwargs: seen.append(current_thread().name) or '요약'

        for _ in range(3):
            request = Thread(target=client.chat, args=('지시', '다시 요약'))
            request.start()
            request.join()

        self.assertEqual(len(seen), 3)
        self.assertEqual(len(set(seen)), 1)
        self.assertNotEqual(seen[0], current_thread().name)
