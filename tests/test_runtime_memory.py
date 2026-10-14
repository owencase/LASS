import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

from src.config import Settings
from src.pipeline import LASSPipeline
from src.runtime import resolve_device
from src.stt.transcriber import Transcription
from src.llm.model_loader import LocalEXAONEClient


class RuntimeMemoryTests(unittest.TestCase):
    def test_gguf_device_selection_and_native_close(self):
        from src.llm.native_runtime import prepare_native
        prepare_native()
        for device, gpu, layers in [('auto', True, -1), ('auto', False, 0), ('cpu', True, 0)]:
            client = LocalEXAONEClient(Path('.'), device=device)
            client.manager.require_ready = Mock()
            model = Mock(metadata={'general.file_type': '30'})
            with patch('llama_cpp.Llama', return_value=model) as load, \
                 patch('llama_cpp.llama_supports_gpu_offload', return_value=True), \
                 patch('src.llm.native_runtime.gpu_available', return_value=gpu):
                self.assertIs(client._runtime(), model)
            self.assertEqual(load.call_args.kwargs['n_gpu_layers'], layers)
            client.unload()
            model.close.assert_called_once()

    def pipeline(self, root):
        return LASSPipeline(Settings(data_dir=Path(root), prompt_dir=Path(__file__).parents[1] / 'prompts'))

    def test_transcriber_is_unloaded_before_llm(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline = self.pipeline(root)
            stt = Mock()
            stt.transcribe.return_value = Transcription('내용', 'ko', 1, [])
            def context(*args, **kwargs):
                stt.unload.assert_called_once()
                return '내용'
            pipeline._all_context = context
            def chat(*args, **kwargs):
                stt.unload.assert_called_once()
                from tests.test_pipeline import structured
                return structured('내용을 정리한 요약본')
            pipeline.llm_client.chat = chat
            pipeline.llm_client.prompt_token_count = lambda system, user: len(user)
            pipeline.llm_client.unload = Mock()
            with patch('src.pipeline.create_transcriber', return_value=stt):
                self.assertIn('내용을 정리한 요약본', pipeline.process(Path(root) / 'audio.wav').summary)
            self.assertFalse(hasattr(pipeline, "_embedder"))
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

    def test_resolve_windows_cpu_and_cuda(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        self.assertEqual(resolve_device(torch), 'cpu')
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            resolve_device(torch, 'cuda')
        torch.cuda.is_available = lambda: True
        self.assertEqual(resolve_device(torch), 'cuda')

    def test_context_budget_keeps_system_and_latest_question(self):
        from tests.gguf_fakes import Tokenizer
        tokenizer = Tokenizer()
        client = LocalEXAONEClient(Path('.'), max_input_tokens=512)
        ids = client._bounded_input(tokenizer, [
            {'role':'system', 'content':'자료로만 답하기'},
            {'role':'user', 'content':'이전 질문' * 300},
            {'role':'assistant', 'content':'이전 답변' * 300},
            {'role':'user', 'content':'참고 자료' * 300 + '[질문] 핵심은?'}])
        self.assertLessEqual(len(ids), 512)
        self.assertIn('자료로만 답하기', tokenizer.last)
        self.assertIn('[질문] 핵심은?', tokenizer.last)
