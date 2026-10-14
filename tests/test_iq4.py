import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from src.llm.model_loader import LocalEXAONEClient
from src.llm.native_runtime import prepare_native

class IQ4Tests(unittest.TestCase):
    def test_template_has_non_thinking_assistant_and_turn_boundaries(self):
        prompt=LocalEXAONEClient._prompt([{'role':'system','content':'지시'},
            {'role':'user','content':'질문'}, {'role':'assistant','content':'<think>비공개</think>답변'},
            {'role':'user','content':'다음 질문'}])
        self.assertNotIn('비공개',prompt)
        self.assertEqual(prompt.count('[|endofturn|]'),4)
        self.assertTrue(prompt.endswith('[|assistant|]\n<think>\n\n</think>\n\n'))

    def test_rejects_other_quantization(self):
        prepare_native()
        client=LocalEXAONEClient(Path('.'),device='cpu')
        client.manager.require_ready=Mock()
        model=Mock(metadata={'general.file_type':'15'})
        with patch('llama_cpp.Llama',return_value=model):
            with self.assertRaisesRegex(ValueError,'IQ4_XS'):
                client._runtime()
        model.close.assert_called_once()

    def test_update_feed_ignores_unrelated_release(self):
        import io,json
        from src.updater import check_update
        with patch('src.updater.urlopen',return_value=io.BytesIO(json.dumps({'name':'LASS BF16','tag_name':'v9.0.0'}).encode())):
            self.assertIsNone(check_update())
