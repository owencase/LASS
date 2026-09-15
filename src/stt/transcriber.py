"""Lazy-loaded local Whisper transcription."""

from __future__ import annotations

import re
import wave
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .audio_utils import prepared_speech_audio, validate_media_file
from src.runtime import release_memory


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


@dataclass(frozen=True)
class TranscriptionQuality:
    """Lightweight signals that a transcript should be reviewed."""

    warnings: List[str]
    repeated_segment_ratio: float
    sparse_long_segments: int

    @property
    def suspicious(self) -> bool:
        return bool(self.warnings)


def _normalized_text(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", " ", text.casefold()).strip()


def _collapse_repeated_phrases(text: str) -> str:
    """Collapse three-or-more consecutive copies of a multi-word phrase."""
    words = text.split()
    if len(words) < 6:
        return text.strip()
    result: List[str] = []
    index = 0
    while index < len(words):
        best_period = 0
        best_count = 0
        max_period = min(20, (len(words) - index) // 3)
        for period in range(2, max_period + 1):
            phrase = words[index : index + period]
            count = 1
            while (
                index + (count + 1) * period <= len(words)
                and words[index + count * period : index + (count + 1) * period]
                == phrase
            ):
                count += 1
            if count >= 3 and period * count > best_period * best_count:
                best_period = period
                best_count = count
        if best_count:
            result.extend(words[index : index + best_period])
            index += best_period * best_count
        else:
            result.append(words[index])
            index += 1
    return " ".join(result)


def clean_transcription_segments(
    segments: Iterable[TranscriptionSegment],
) -> List[TranscriptionSegment]:
    """Remove runaway Whisper repetition while preserving a repeated cue twice."""
    cleaned: List[TranscriptionSegment] = []
    previous = ""
    repeated_run = 0
    for segment in segments:
        text = _collapse_repeated_phrases(segment.text)
        normalized = _normalized_text(text)
        if not normalized:
            continue
        similar = bool(previous) and (
            normalized == previous
            or (
                min(len(normalized), len(previous)) >= 12
                and SequenceMatcher(None, normalized, previous).ratio() >= 0.92
            )
        )
        repeated_run = repeated_run + 1 if similar else 1
        if repeated_run <= 2:
            cleaned.append(
                TranscriptionSegment(segment.start, segment.end, text.strip())
            )
        previous = normalized
    return cleaned


def assess_transcription_quality(
    segments: Iterable[Any], duration: Optional[float]
) -> TranscriptionQuality:
    """Detect repetition and implausibly sparse timestamp spans."""
    items = list(segments)
    normalized = [_normalized_text(str(item.text)) for item in items]
    normalized = [text for text in normalized if text]
    repeated_ratio = 0.0
    if normalized:
        repeated_ratio = max(Counter(normalized).values()) / len(normalized)
    sparse_long = sum(
        1
        for item in items
        if float(item.end) - float(item.start) >= 60
        and len(_normalized_text(str(item.text)))
        / max(1.0, float(item.end) - float(item.start))
        < 0.5
    )
    warnings: List[str] = []
    if len(normalized) >= 6 and repeated_ratio >= 0.3:
        warnings.append("같은 문장이 녹취 구간의 30% 이상에서 반복됩니다.")
    if sparse_long >= 2:
        warnings.append("긴 시간 구간에 비해 인식된 문장이 지나치게 적습니다.")
    if duration and duration >= 600:
        character_count = sum(len(text) for text in normalized)
        if character_count / duration < 0.25:
            warnings.append("강의 길이에 비해 인식된 텍스트 양이 매우 적습니다.")
    return TranscriptionQuality(warnings, repeated_ratio, sparse_long)


class WhisperTranscriber:
    """Transcribe media locally with faster-whisper.

    The model is loaded only on the first call so importing the app stays fast.
    """

    LONG_AUDIO_SECONDS = 10 * 60
    CHUNK_SECONDS = 5 * 60
    CHUNK_OVERLAP_SECONDS = 1.0

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "auto",
        compute_type: str = "int8",
        target_lufs: float = -16.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.target_lufs = target_lufs
        self._model: Any = None

    def _get_model(self) -> Any:
        if self.device == "mps":
            raise ValueError("faster-whisper는 Apple GPU를 지원하지 않습니다. SenseVoice 또는 CPU를 선택해 주세요.")
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

    def unload(self) -> None:
        self._model = None
        release_memory()

    def transcribe(
        self,
        media_path: Path,
        language: Optional[str] = None,
        beam_size: int = 5,
    ) -> Transcription:
        source = validate_media_file(media_path)
        model = self._get_model()
        recognition_options = {
            "language": language,
            "beam_size": max(beam_size, 8),
            "patience": 1.2,
            "temperature": (0.0, 0.2, 0.4),
            "compression_ratio_threshold": 2.2,
            "log_prob_threshold": -0.8,
            "condition_on_previous_text": False,
            "prompt_reset_on_temperature": 0.2,
            "repetition_penalty": 1.12,
            "no_repeat_ngram_size": 3,
            "no_speech_threshold": 0.5,
            "word_timestamps": True,
            "hallucination_silence_threshold": 2.0,
            "vad_filter": True,
            "vad_parameters": {
                "threshold": 0.35,
                "min_speech_duration_ms": 100,
                "min_silence_duration_ms": 350,
                "speech_pad_ms": 400,
            },
        }
        # LUFS normalization intentionally runs before VAD so quiet speech is
        # not discarded by the speech/no-speech gate.
        with prepared_speech_audio(source, target_lufs=self.target_lufs) as prepared:
            prepared_duration = self._wav_duration(prepared.path)
            if (
                prepared_duration is not None
                and prepared_duration > self.LONG_AUDIO_SECONDS
            ):
                segments, detected_language = self._transcribe_long_audio(
                    model, prepared.path, recognition_options
                )
                info = None
            else:
                raw_segments, info = model.transcribe(
                    str(prepared.path), **recognition_options
                )
                segments = self._segments(raw_segments)
                detected_language = getattr(info, "language", language)
            if not segments and (
                prepared_duration is None
                or prepared_duration <= self.LONG_AUDIO_SECONDS
            ):
                retry_options = dict(recognition_options)
                retry_options["vad_filter"] = False
                retry_options.pop("vad_parameters", None)
                retry_options["hallucination_silence_threshold"] = None
                raw_segments, info = model.transcribe(
                    str(prepared.path), **retry_options
                )
                segments = self._segments(raw_segments)
                detected_language = getattr(info, "language", language)
            if not segments:
                raise RuntimeError(
                    "긴 녹음에서 음성을 찾지 못했습니다. "
                    "입력 장치와 실제 음량을 확인한 뒤 다시 처리해 주세요."
                )
            segments = clean_transcription_segments(segments)
        return Transcription(
            text=" ".join(segment.text for segment in segments),
            language=detected_language,
            duration=(
                prepared_duration
                if prepared_duration is not None
                else getattr(info, "duration", None)
            ),
            segments=segments,
        )

    @staticmethod
    def _wav_duration(path: Path) -> Optional[float]:
        try:
            with wave.open(str(path), "rb") as audio:
                return audio.getnframes() / audio.getframerate()
        except (OSError, EOFError, wave.Error):
            return None

    def _transcribe_long_audio(
        self,
        model: Any,
        path: Path,
        recognition_options: dict[str, Any],
    ) -> tuple[List[TranscriptionSegment], Optional[str]]:
        """Reset Whisper context at fixed intervals to contain long-form drift."""
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy is required for long recording transcription") from exc

        segments: List[TranscriptionSegment] = []
        detected_language = recognition_options.get("language")
        with wave.open(str(path), "rb") as audio:
            sample_rate = audio.getframerate()
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise RuntimeError("Prepared speech audio must be mono 16-bit PCM")
            total_frames = audio.getnframes()
            core_frames = max(1, round(self.CHUNK_SECONDS * sample_rate))
            overlap_frames = max(
                0, round(self.CHUNK_OVERLAP_SECONDS * sample_rate)
            )
            core_start = 0
            while core_start < total_frames:
                core_end = min(total_frames, core_start + core_frames)
                read_start = max(0, core_start - overlap_frames)
                read_end = min(total_frames, core_end + overlap_frames)
                audio.setpos(read_start)
                pcm = audio.readframes(read_end - read_start)
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
                samples /= 32768.0
                raw_segments, info = model.transcribe(
                    samples, **recognition_options
                )
                if detected_language is None:
                    detected_language = getattr(info, "language", None)
                offset = read_start / sample_rate
                core_start_seconds = core_start / sample_rate
                core_end_seconds = core_end / sample_rate
                for segment in self._segments(raw_segments, offset=offset):
                    midpoint = (segment.start + segment.end) / 2
                    if core_start_seconds <= midpoint < core_end_seconds or (
                        core_end == total_frames and midpoint <= core_end_seconds
                    ):
                        segments.append(segment)
                core_start = core_end
        return segments, detected_language

    @staticmethod
    def _segments(
        raw_segments: Any, *, offset: float = 0.0
    ) -> List[TranscriptionSegment]:
        return [
            TranscriptionSegment(
                start=float(segment.start) + offset,
                end=float(segment.end) + offset,
                text=segment.text.strip(),
            )
            for segment in raw_segments
            if segment.text.strip()
        ]
