"""Pinned, resumable model downloads with byte progress and SHA-256 validation."""
from fnmatch import fnmatch
from src import __version__
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

EXAONE_REVISION = '10e7d7d52ff49b7c95dd16275bc9c290e2ddad4a'
MODELS = (
    ('llm', 'EXAONE 4.0 1.2B IQ4_XS', '노트 요약·질의응답·퀴즈 생성을 위한 로컬 언어 모델'),
    ('stt', 'Moonshine Voice · 한국어 / 영어', '한국어 Tiny / 영어 Medium Streaming · Powered by Moonshine AI'),
)


def model_ready(settings, key):
    if key == 'llm':
        from src.llm.model_manager import LocalModelManager
        return LocalModelManager(settings.exaone_model_id, settings.exaone_model_dir).status().ready
    from src.stt.moonshine import model_ready as stt_ready
    return stt_ready(settings.stt_model_dir)



def download_file(url, path, size, checksum=None, progress=None, cancelled=None):
    """Publish a file only after completion; keep partial data for the next run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    notify = progress or (lambda done: None)
    def verified(candidate):
        if not candidate.is_file() or candidate.stat().st_size != size:
            return False
        if checksum:
            digest = hashlib.sha256()
            with candidate.open('rb') as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
            return digest.hexdigest() == checksum
        return True
    if verified(path):
        notify(size)
        return
    partial = path.with_name(path.name + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        partial.unlink()
        offset = 0
    if offset < size:
        headers = {'User-Agent': f'LASS/{__version__} (local model downloader)'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            if offset and response.status == 206:
                if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise OSError('서버의 다운로드 재개 위치가 올바르지 않습니다.')
            else:
                offset = 0
            with partial.open('ab' if offset else 'wb') as output:
                notify(offset)
                while block := response.read(1024 * 1024):
                    if cancelled and cancelled.is_set():
                        raise InterruptedError('다운로드가 취소되었습니다.')
                    output.write(block)
                    offset += len(block)
                    notify(min(offset, size))
    if not verified(partial):
        if partial.is_file() and partial.stat().st_size >= size:
            partial.unlink()
        raise OSError(f'{path.name}: 다운로드 크기 또는 체크섬이 일치하지 않습니다.')
    partial.replace(path)
    notify(size)


def download_model(settings, key, progress=None, cancelled=None):
    if key == 'stt':
        from src.stt.moonshine import ensure_model
        return ensure_model(settings.stt_model_dir, progress, cancelled)
    from huggingface_hub import HfApi, hf_hub_url
    notify = progress or (lambda fraction, message: None)
    if key == 'llm':
        repo, revision, destination = settings.exaone_model_id, EXAONE_REVISION, settings.exaone_model_dir
        from src.llm.model_manager import GGUF_FILENAME
        patterns = (GGUF_FILENAME,)
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    files = [f for f in info.siblings if any(fnmatch(f.rfilename, p) for p in patterns)]
    total = sum(f.size or 0 for f in files)
    if not files or not total:
        raise OSError('모델 파일 목록과 크기를 확인할 수 없습니다.')
    completed = 0
    for file in files:
        # Repository paths are external input. Never write outside the model directory.
        target = (destination / file.rfilename).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError('잘못된 모델 파일 경로입니다.')
        checksum = file.lfs.get('sha256') if isinstance(file.lfs, dict) else getattr(file.lfs, 'sha256', None)
        for attempt in range(3):
            if cancelled and cancelled.is_set():
                raise InterruptedError('다운로드가 취소되었습니다.')
            try:
                download_file(hf_hub_url(repo, file.rfilename, revision=revision), target, file.size,
                              checksum, lambda done: notify(min((completed + done) / total, .999), file.rfilename), cancelled)
                break
            except OSError:
                if attempt == 2:
                    raise
                notify(completed / total, f'{file.rfilename} · 자동 재시도 {attempt + 1}/2')
                if cancelled and cancelled.wait(attempt + 1):
                    raise InterruptedError('다운로드가 취소되었습니다.')
        completed += file.size
    if not model_ready(settings, key):
        raise OSError('필수 모델 파일이 누락되었습니다.')
    notify(1, '완료')


def prepare_models(settings, progress=None):
    settings.ensure_directories()
    for key, name, _ in MODELS:
        if not model_ready(settings, key):
            download_model(settings, key, lambda fraction, message: progress and progress(f'{name} {fraction:.0%} · {message}'))
