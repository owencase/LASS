import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from src.model_assets import download_file, download_model, model_ready
from src.config import Settings


class Response(io.BytesIO):
    status = 200
    headers = {}


class ModelPreparationTests(unittest.TestCase):
    def test_download_progress_atomic_publish_checksum_and_cached_reuse(self):
        data = b'model-weights' * 100
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model'
            updates = []
            checksum = hashlib.sha256(data).hexdigest()
            with patch('src.model_assets.urlopen', return_value=Response(data)):
                download_file('https://example.test/model', path, len(data), checksum, updates.append)
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(updates[-1], len(data))
            with patch('src.model_assets.urlopen') as request:
                download_file('https://example.test/model', path, len(data), checksum)
                request.assert_not_called()

    def test_resume_when_server_accepts_range_and_restart_if_ignored(self):
        for accepts in (True, False):
            with self.subTest(accepts=accepts), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model'
                path.with_name('model.partial').write_bytes(b'abc')
                response = Response(b'def' if accepts else b'abcdef')
                response.status = 206 if accepts else 200
                response.headers = {'Content-Range':'bytes 3-5/6'} if accepts else {}
                with patch('src.model_assets.urlopen', return_value=response) as request:
                    download_file('https://example.test/model', path, 6)
                    self.assertEqual(request.call_args.args[0].headers['Range'], 'bytes=3-')
                self.assertEqual(path.read_bytes(), b'abcdef')

    def test_failure_never_publishes_corrupt_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model'
            with patch('src.model_assets.urlopen', return_value=Response(b'bad')):
                with self.assertRaises(OSError):
                    download_file('https://example.test/model', path, 3, '0' * 64)
            self.assertFalse(path.exists())

    def test_model_requests_identify_the_app_on_fresh_and_resumed_downloads(self):
        from src import __version__
        for prefix in (b'', b'abc'):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'model'
                if prefix:
                    path.with_name('model.partial').write_bytes(prefix)
                response=Response(b'def' if prefix else b'abcdef')
                response.status=206 if prefix else 200
                response.headers={'Content-Range':'bytes 3-5/6'} if prefix else {}
                with patch('src.model_assets.urlopen',return_value=response) as request:
                    download_file('https://example.test/model',path,6)
                headers=request.call_args.args[0].headers
                self.assertEqual(headers['User-agent'],f'LASS/{__version__} (local model downloader)')
                self.assertEqual(path.read_bytes(),b'abcdef')
                if prefix:self.assertEqual(headers['Range'],'bytes=3-')

    def test_aggregate_progress_and_hash_pinned_moonshine_files(self):
        import hashlib
        checksum=hashlib.sha256(b'abc').hexdigest()
        catalog={code:{'model_arch':0,'files':[{'name':'model.ort','size':3,'sha256':checksum,'url':'https://example.test/'+code}]} for code in ('ko','en')}
        with tempfile.TemporaryDirectory() as directory:
            settings=Settings(stt_model_dir=Path(directory))
            def transfer(url,path,size,expected,progress,cancelled):
                self.assertEqual(expected,checksum)
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'abc');progress(3)
            updates=[]
            with patch('src.stt.moonshine.CATALOG',catalog),patch('src.model_assets.download_file',side_effect=transfer):
                download_model(settings,'stt',lambda f,m:updates.append(f))
                self.assertTrue(model_ready(settings,'stt'))
            self.assertEqual(updates[-1],1)
            self.assertEqual(updates[0],.5)
