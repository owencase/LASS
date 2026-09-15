"""Bigvox EXAONE 4 Audio transcription adapter.

The heavyweight PyTorch/Transformers implementation is imported only when the
model is first used so opening the application does not allocate model memory.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from src.runtime import release_memory

from .audio_utils import prepared_speech_audio, validate_media_file
from .transcriber import (
    Transcription,
    TranscriptionSegment,
    clean_transcription_segments,
)


BIGVOX_MODEL_ID = "bigdefence/Bigvox-Exaone4-Audio"
BIGVOX_SPEECH_ENCODER_ID = "openai/whisper-large-v3"


@dataclass(frozen=True)
class _AudioChunk:
    samples: Any
    start: float
    end: float


class BigvoxTranscriber:
    """Transcribe Korean audio with Bigvox and its Whisper-large-v3 encoder."""

    # The upstream non-streaming model pads or trims every request to 30 seconds.
    CHUNK_SECONDS = 30

    def __init__(
        self,
        model_name: str = BIGVOX_MODEL_ID,
        device: str = "auto",
        compute_type: str = "float16",
        target_lufs: float = -16.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.target_lufs = target_lufs
        self._runtime: Any = None

    def _get_model(self) -> Any:
        if self._runtime is None:
            try:
                from .bigvox_runtime import load_bigvox_runtime
            except ImportError as exc:
                raise RuntimeError(
                    "Bigvox 실행에는 PyTorch와 Transformers가 필요합니다. "
                    "`pip install -r requirements.txt`로 의존성을 설치해 주세요."
                ) from exc
            self._runtime = load_bigvox_runtime(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                speech_encoder_id=BIGVOX_SPEECH_ENCODER_ID,
            )
        return self._runtime

    def unload(self) -> None:
        self._runtime = None
        release_memory()

    @staticmethod
    def _duration(path: Path) -> Optional[float]:
        try:
            with wave.open(str(path), "rb") as audio:
                return audio.getnframes() / audio.getframerate()
        except (OSError, EOFError, wave.Error):
            return None

    def _audio_chunks(self, path: Path) -> Iterator[_AudioChunk]:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Bigvox 오디오 처리에는 NumPy가 필요합니다.") from exc

        with wave.open(str(path), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != 16_000
            ):
                raise RuntimeError(
                    "Bigvox 입력은 16 kHz 모노 16-bit PCM으로 준비되어야 합니다."
                )
            rate = audio.getframerate()
            total_frames = audio.getnframes()
            chunk_frames = self.CHUNK_SECONDS * rate
            frame_start = 0
            while frame_start < total_frames:
                frame_count = min(chunk_frames, total_frames - frame_start)
                pcm = audio.readframes(frame_count)
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
                samples /= 32768.0
                yield _AudioChunk(
                    samples=samples,
                    start=frame_start / rate,
                    end=(frame_start + frame_count) / rate,
                )
                frame_start += frame_count

    def transcribe(
        self,
        media_path: Path,
        language: Optional[str] = None,
        beam_size: int = 5,
    ) -> Transcription:
        del beam_size  # The published Bigvox recipe uses deterministic greedy decoding.
        if language and language.casefold() not in {"ko", "korean"}:
            raise ValueError("Bigvox는 한국어 음성 인식 전용 모델입니다.")

        source = validate_media_file(media_path)
        runtime = self._get_model()
        with prepared_speech_audio(source, target_lufs=self.target_lufs) as prepared:
            duration = self._duration(prepared.path)
            segments = []
            for chunk in self._audio_chunks(prepared.path):
                text = str(runtime.transcribe(chunk.samples)).strip()
                if text:
                    segments.append(
                        TranscriptionSegment(chunk.start, chunk.end, text)
                    )
            segments = clean_transcription_segments(segments)
            if not segments:
                raise RuntimeError(
                    "Bigvox가 녹음에서 음성을 찾지 못했습니다. "
                    "입력 장치와 음량을 확인해 주세요."
                )

        return Transcription(
            text=" ".join(segment.text for segment in segments),
            language="ko",
            duration=duration,
            segments=segments,
        )
