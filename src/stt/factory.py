"""The single supported STT runtime."""
from pathlib import Path
from .moonshine import MoonshineTranscriber

def create_transcriber(*, model_dir: Path, target_lufs: float = -16.0) -> MoonshineTranscriber:
    return MoonshineTranscriber(model_dir, target_lufs)
