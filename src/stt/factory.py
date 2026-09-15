"""STT backend selection without eagerly importing model runtimes."""

from __future__ import annotations

from typing import Union

from .bigvox import BigvoxTranscriber
from .sensevoice import SenseVoiceTranscriber
from .transcriber import WhisperTranscriber


Transcriber = Union[WhisperTranscriber, SenseVoiceTranscriber, BigvoxTranscriber]


def create_transcriber(
    *,
    backend: str,
    model_name: str,
    device: str,
    compute_type: str,
    target_lufs: float,
) -> Transcriber:
    normalized = backend.strip().casefold()
    if normalized == "sensevoice":
        return SenseVoiceTranscriber(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            target_lufs=target_lufs,
        )
    if normalized == "faster-whisper":
        return WhisperTranscriber(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            target_lufs=target_lufs,
        )
    if normalized == "bigvox":
        return BigvoxTranscriber(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            target_lufs=target_lufs,
        )
    raise ValueError(f"지원하지 않는 STT 실행 엔진입니다: {backend}")
