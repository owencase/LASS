import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.llm.model_loader import LocalEXAONEClient


class InferenceEncodingTests(unittest.TestCase):
    def test_wire_is_ascii_json_and_preserves_korean_and_emoji(self):
        answer = '한국어 응답 · 문서.pdf · 3페이지 😊'
        process = Mock()
        process.poll.return_value = None
        process.stdin = io.BytesIO()
        process.stdout = io.BytesIO((json.dumps({'token':answer}, ensure_ascii=True)+'\n'+json.dumps({'done':True,'device':'cpu'})+'\n').encode('ascii'))
        client = LocalEXAONEClient(Path('한글 경로'), isolated=True)
        client._process = process
        self.assertEqual(''.join(client.stream_messages([{'role':'user','content':'한글 질문 😊'}])), answer)
        request = process.stdin.getvalue()
        request.decode('ascii')
        self.assertEqual(json.loads(request)['messages'][0]['content'], '한글 질문 😊')

    def test_real_worker_preserves_unicode_path_and_writes_utf8_errors(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            os.environ['LASS_DATA_DIR'] = directory
            path = Path(directory)/'한글 모델 경로'
            client = LocalEXAONEClient(path, isolated=True)
            with self.assertRaises(RuntimeError) as caught:
                list(client.stream_messages([{'role':'user','content':'한국어 질문'}]))
            self.assertIn(str(path.resolve()), str(caught.exception))
            self.assertIsNone(client._process)


if __name__ == '__main__':
    unittest.main()
