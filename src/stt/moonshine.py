"""Moonshine Voice recognition with local, hash-pinned language models."""
from pathlib import Path
import json
import wave
from src.runtime import release_memory
from .audio_utils import prepared_speech_audio, validate_media_file
from .transcriber import Transcription, TranscriptionSegment, clean_transcription_segments

CATALOG = json.loads(Path(__file__).with_name('moonshine-models.json').read_text(encoding='utf-8'))

def model_ready(model_dir, language=None):
    languages = (language,) if language else tuple(CATALOG)
    return all((Path(model_dir) / code / f['name']).is_file()
               and (Path(model_dir) / code / f['name']).stat().st_size == f['size']
               for code in languages for f in CATALOG[code]['files'])

def ensure_model(model_dir, progress=None, cancelled=None):
    from src.model_assets import download_file
    notify = progress or (lambda fraction, message: None)
    total = sum(f['size'] for model in CATALOG.values() for f in model['files'])
    completed = 0
    for language, model in CATALOG.items():
        for file in model['files']:
            if cancelled and cancelled.is_set():
                raise InterruptedError('다운로드가 취소되었습니다.')
            target = Path(model_dir) / language / file['name']
            download_file(file['url'], target, file['size'], file['sha256'],
                lambda done: notify(min((completed + done) / total, .999), language + '/' + file['name']), cancelled)
            completed += file['size']
    notify(1, '완료')

class MoonshineTranscriber:
    def __init__(self, model_dir, target_lufs=-16.0, default_language=None):
        from src.config import Settings
        self.model_dir = Path(model_dir)
        self.target_lufs = target_lufs
        self.default_language = default_language or Settings.from_env().stt_language
        self._model = None
        self._language = None

    def _get_model(self, language):
        if self._model is None or self._language != language:
            self.unload()
            if not model_ready(self.model_dir, language):
                raise RuntimeError('Moonshine 모델이 준비되지 않았습니다. 모델 다운로드를 완료해 주세요.')
            from moonshine_voice import Transcriber, ModelArch
            self._model = Transcriber(model_path=str(self.model_dir / language),
                model_arch=ModelArch(CATALOG[language]['model_arch']),
                options={'max_tokens_per_second': 13.0} if language == 'ko' else None)
            self._language = language
        return self._model

    def unload(self):
        if self._model is not None:
            self._model.close()
        self._model = None
        self._language = None
        release_memory()

    def transcribe(self, media_path, language=None, on_segment=None):
        language = language or self.default_language
        if language not in ('ko', 'en'):
            raise ValueError('한국어와 영어만 지원합니다.')
        import numpy as np
        source = validate_media_file(Path(media_path))
        segments = []
        with prepared_speech_audio(source, target_lufs=self.target_lufs) as prepared:
            with wave.open(str(prepared.path), 'rb') as audio:
                rate = audio.getframerate()
                duration = audio.getnframes() / rate
                if rate != 16000 or audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                    raise ValueError('Prepared speech audio must be mono 16 kHz PCM16')
                while True:
                    start = audio.tell() / rate
                    pcm = audio.readframes(rate * 20)
                    if not pcm:
                        break
                    samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
                    if not np.any(samples):
                        continue
                    transcript = self._get_model(language).transcribe_without_streaming(samples.tolist(), sample_rate=rate)
                    end = audio.tell() / rate
                    for line in transcript.lines:
                        text = line.text.strip()
                        if not text:
                            continue
                        begin = max(start, min(end, start + float(line.start_time)))
                        finish = max(begin, min(end, begin + float(line.duration)))
                        segment = TranscriptionSegment(begin, finish, text)
                        segments.append(segment)
                        if on_segment:
                            on_segment(segment)
        segments = clean_transcription_segments(segments)
        if not segments:
            raise RuntimeError('음성을 찾지 못했습니다. 녹음 음량과 입력 파일을 확인해 주세요.')
        return Transcription(' '.join(s.text for s in segments), language, duration, segments)
