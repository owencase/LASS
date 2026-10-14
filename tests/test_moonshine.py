import hashlib
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from src.stt.moonshine import MoonshineTranscriber,ensure_model,model_ready

class MoonshineTests(unittest.TestCase):
    def test_only_korean_and_english_can_be_requested(self):
        with self.assertRaisesRegex(ValueError,'한국어와 영어'):
            MoonshineTranscriber(Path('unused')).transcribe(Path('unused'),'ja')

    def test_chunks_include_final_partial_chunk_and_preserve_native_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audio.wav'
            with wave.open(str(path),'wb') as audio:
                audio.setparams((1,2,16000,0,'NONE','not compressed'))
                audio.writeframes(b'\x00\x01'*(21*16000))
            model=Mock()
            model.transcribe_without_streaming.return_value=SimpleNamespace(lines=[SimpleNamespace(text='수업 내용',start_time=0,duration=30)])
            transcriber=MoonshineTranscriber(Path(directory))
            callback=Mock()
            with patch.object(transcriber,'_get_model',return_value=model),patch('src.stt.moonshine.prepared_speech_audio') as prepare:
                prepare.return_value.__enter__.return_value.path=path
                result=transcriber.transcribe(path,on_segment=callback)
            self.assertEqual([(s.start,s.end) for s in result.segments],[(0,20),(20,21)])
            self.assertEqual(callback.call_count,2)
            self.assertEqual(result.language,'ko')
            self.assertEqual(model.transcribe_without_streaming.call_count,2)

    def test_close_is_called_when_language_changes_and_on_unload(self):
        import moonshine_voice
        first,second=Mock(),Mock()
        load=Mock(side_effect=[first,second])
        transcriber=MoonshineTranscriber(Path('.'))
        with patch('src.stt.moonshine.model_ready',return_value=True),patch.dict(vars(moonshine_voice),{'Transcriber':load}):
            self.assertIs(transcriber._get_model('ko'),first)
            self.assertEqual(load.call_args.kwargs['options']['max_tokens_per_second'],13)
            self.assertIs(transcriber._get_model('en'),second)
            first.close.assert_called_once()
            transcriber.unload()
            second.close.assert_called_once()

    def test_missing_assets_never_trigger_implicit_network_download(self):
        with tempfile.TemporaryDirectory() as directory,patch('src.stt.moonshine.ensure_model') as download:
            with self.assertRaisesRegex(RuntimeError,'다운로드'):
                MoonshineTranscriber(Path(directory))._get_model('ko')
            download.assert_not_called()

    def test_existing_assets_are_sha_verified_without_network(self):
        payload=b'asset'
        catalog={'ko':{'model_arch':0,'files':[{'name':'encoder.ort','url':'https://example.test/model','size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}]}}
        with tempfile.TemporaryDirectory() as directory,patch('src.stt.moonshine.CATALOG',catalog),patch('src.model_assets.urlopen') as request:
            folder=Path(directory)/'ko';folder.mkdir();(folder/'encoder.ort').write_bytes(payload)
            ensure_model(Path(directory))
            request.assert_not_called()
            self.assertTrue(model_ready(Path(directory)))

    def test_ui_language_change_updates_future_jobs_and_releases_live_model(self):
        import os
        from src.ui.app import _change_stt_language
        repo=Mock()
        with patch('src.ui.app.st.session_state',{'moonshine-language':'en'}),patch('src.ui.app._release_live_transcriber') as release,patch.dict(os.environ,{},clear=False):
            _change_stt_language(repo)
            self.assertEqual(os.environ['LASS_STT_LANGUAGE'],'en')
            repo.set_metadata.assert_called_once_with('stt_language','en')
            release.assert_called_once()
