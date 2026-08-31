"""Lazy-loaded local Whisper transcription."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from .audio_utils import validate_media_file


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcription:
    text: str
    language: Optional[str]
    duration: Optional[float]
    segments: List[TranscriptionSegment]


class WhisperTranscriber:
    """Transcribe media locally with faster-whisper.

    The model is loaded only on the first call so importing the app stays fast.
    """

    def __init__(
        self,
        model_name: str = "small",
        device: str = "auto",
        compute_type: str = "int8",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is required for transcription. "
                    "Install dependencies with `pip install -r requirements.txt`."
                ) from exc
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(
        self,
        media_path: Path,
        language: Optional[str] = None,
        beam_size: int = 5,
    ) -> Transcription:
        source = validate_media_file(media_path)
        model = self._get_model()
        raw_segments, info = model.transcribe(
            str(source),
            language=language,
            beam_size=beam_size,
            vad_filter=True,
        )
        segments = [
            TranscriptionSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
            )
            for segment in raw_segments
            if segment.text.strip()
        ]
        return Transcription(
            text=" ".join(segment.text for segment in segments),
            language=getattr(info, "language", language),
            duration=getattr(info, "duration", None),
            segments=segments,
        )
