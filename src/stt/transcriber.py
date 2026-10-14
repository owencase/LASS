"""Shared speech transcription types and quality checks."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, List, Optional



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
