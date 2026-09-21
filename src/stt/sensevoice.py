"""Lazy-loaded SenseVoice Small transcription through FunASR."""

from __future__ import annotations

import re
import wave
import tempfile
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
from threading import RLock
from typing import Any, Callable, List, Optional

from .audio_utils import prepared_speech_audio, validate_media_file
from src.runtime import release_memory, resolve_device
from .transcriber import (
    Transcription,
    TranscriptionSegment,
    clean_transcription_segments,
)


_LANGUAGE_TAG = re.compile(r"<\|(zh|en|yue|ja|ko)\|>", re.IGNORECASE)
_RICH_TAG = re.compile(r"<\|[^|>]+\|>")
_MODEL_INITIALIZATION_LOCK = RLock()


class SenseVoiceTranscriber:
    """Transcribe arbitrary-length media with SenseVoice Small and FSMN-VAD."""

    def __init__(
        self,
        model_name: str = "FunAudioLLM/SenseVoiceSmall",
        device: str = "auto",
        compute_type: str = "float32",
        target_lufs: float = -16.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.target_lufs = target_lufs
        self._model: Any = None
        self._postprocess: Optional[Callable[[str], str]] = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        try:
            import torch
        except ImportError:
            if device not in {"auto", "cpu"}:
                raise RuntimeError("GPU 사용에는 PyTorch가 필요합니다.")
            return "cpu"
        selected = resolve_device(torch, device)
        return "cuda:0" if selected == "cuda" else selected

    def unload(self) -> None:
        self._model = self._postprocess = None
        release_memory()

    def _get_model(self) -> Any:
        if self._model is None:
            # FunASR performs module auto-registration and model downloads while
            # constructing AutoModel. Those global operations are not thread-safe;
            # concurrent Streamlit sessions can otherwise leave the registry only
            # partially initialized or trip tqdm's global download lock.
            with _MODEL_INITIALIZATION_LOCK:
                if self._model is None:
                    try:
                        from funasr import AutoModel
                        from funasr.utils.postprocess_utils import (
                            rich_transcription_postprocess,
                        )
                    except ImportError as exc:
                        raise RuntimeError(
                            "SenseVoice Small 실행에는 FunASR가 필요합니다. "
                            "`pip install -r requirements.txt`로 의존성을 "
                            "설치해 주세요."
                        ) from exc
                    device = self._resolve_device(self.device)
                    options = {
                        "model": self.model_name,
                        "vad_model": "fsmn-vad",
                        "vad_kwargs": {"max_single_segment_time": 30_000},
                        "device": device,
                        "hub": "hf",
                        "disable_update": True,
                        "disable_pbar": True,
                    }
                    if self.compute_type == "float16" and device.startswith("cuda"):
                        options["fp16"] = True
                    self._model = AutoModel(**options)
                    self._postprocess = rich_transcription_postprocess
        return self._model

    @staticmethod
    def _generate_without_vad(
        model: Any, chunk_path: Path, language: Optional[str]
    ) -> list[dict[str, Any]]:
        """Retry audible input directly when the external VAD misses all speech."""
        inference = getattr(model, "inference", None)
        if not callable(inference):
            return []
        results = inference(
            input=str(chunk_path),
            cache={},
            language=language or "auto",
            use_itn=True,
            batch_size_s=15,
            sentence_timestamp=True,
        )
        if not isinstance(results, list):
            return []
        return [item for item in results if isinstance(item, dict)]

    def _clean_text(self, text: object) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        if self._postprocess is not None:
            return str(self._postprocess(raw)).strip()
        return _RICH_TAG.sub("", raw).strip()

    @staticmethod
    def _duration(path: Path) -> Optional[float]:
        try:
            with wave.open(str(path), "rb") as audio:
                return audio.getnframes() / audio.getframerate()
        except (OSError, EOFError, wave.Error):
            return None

    @staticmethod
    def _millisecond(value: object, fallback: float) -> float:
        try:
            return max(0.0, float(value) / 1000.0)
        except (TypeError, ValueError):
            return fallback

    def _segments(
        self, result: dict[str, Any], duration: Optional[float]
    ) -> List[TranscriptionSegment]:
        segments: List[TranscriptionSegment] = []
        sentence_info = result.get("sentence_info")
        if isinstance(sentence_info, list):
            for item in sentence_info:
                if not isinstance(item, dict):
                    continue
                text = self._clean_text(item.get("text"))
                if not text:
                    continue
                start = self._millisecond(item.get("start"), 0.0)
                end = self._millisecond(item.get("end"), start)
                if end <= start:
                    end = duration if duration is not None else start
                segments.append(TranscriptionSegment(start, end, text))
        if segments:
            return clean_transcription_segments(segments)

        text = self._clean_text(result.get("text"))
        if not text:
            return []
        timestamps = result.get("timestamp")
        start = 0.0
        end = duration or 0.0
        if isinstance(timestamps, list) and timestamps:
            first = timestamps[0]
            last = timestamps[-1]
            if isinstance(first, (list, tuple)) and first:
                start = self._millisecond(first[0], 0.0)
            if isinstance(last, (list, tuple)) and len(last) >= 2:
                end = self._millisecond(last[1], end)
        return [TranscriptionSegment(start, max(start, end), text)]

    @staticmethod
    def _language(
        result: dict[str, Any], raw_text: str, requested: Optional[str]
    ) -> Optional[str]:
        reported = str(result.get("language", "")).strip().lower()
        if reported in {"zh", "en", "yue", "ja", "ko"}:
            return reported
        tags = [tag.lower() for tag in _LANGUAGE_TAG.findall(raw_text)]
        if tags:
            return Counter(tags).most_common(1)[0][0]
        return requested

    CHUNK_SECONDS = 60
    OVERLAP_SECONDS = 1

    @contextmanager
    def _audio_chunks(self, path: Path, duration: Optional[float]):
        if duration is None or duration <= self.CHUNK_SECONDS:
            yield iter([(path, 0.0, 0.0, duration or float("inf"))])
            return
        with tempfile.TemporaryDirectory(prefix="lass-sensevoice-") as directory:
            def chunks():
                with wave.open(str(path), "rb") as audio:
                    rate, total = audio.getframerate(), audio.getnframes()
                    core = max(1, round(self.CHUNK_SECONDS * rate))
                    overlap = round(self.OVERLAP_SECONDS * rate)
                    for start in range(0, total, core):
                        end = min(total, start + core)
                        left, right = max(0, start-overlap), min(total, end+overlap)
                        audio.setpos(left)
                        target = Path(directory) / "chunk.wav"
                        with wave.open(str(target), "wb") as output:
                            output.setparams(audio.getparams())
                            output.writeframes(audio.readframes(right-left))
                        yield target, left/rate, start/rate, end/rate
            yield chunks()

    def transcribe(
        self,
        media_path: Path,
        language: Optional[str] = None,
        beam_size: int = 5,
    ) -> Transcription:
        del beam_size  # SenseVoice is non-autoregressive and does not use beam search.
        source = validate_media_file(media_path)
        model = self._get_model()
        with prepared_speech_audio(source, target_lufs=self.target_lufs) as prepared:
            duration = self._duration(prepared.path)
            segments = []
            languages = []
            with self._audio_chunks(prepared.path, duration) as chunks:
                for chunk_path, offset, core_start, core_end in chunks:
                    results = model.generate(
                        input=str(chunk_path), cache={}, language=language or "auto",
                        use_itn=True, batch_size_s=15, merge_vad=True,
                        merge_length_s=10, sentence_timestamp=True,
                    )
                    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
                        raise RuntimeError("SenseVoice가 올바른 음성 인식 결과를 반환하지 않았습니다.")
                    result = results[0]
                    chunk_duration = self._duration(chunk_path)
                    chunk_segments = self._segments(result, chunk_duration)
                    if not chunk_segments and prepared.source_peak > 0:
                        fallback_results = self._generate_without_vad(
                            model, chunk_path, language
                        )
                        if fallback_results:
                            result = fallback_results[0]
                            chunk_segments = self._segments(result, chunk_duration)
                    detected = self._language(result, str(result.get("text", "")), language)
                    if detected:
                        languages.append(detected)
                    for item in chunk_segments:
                        start, end = item.start + offset, item.end + offset
                        midpoint = (start + end) / 2
                        if core_start <= midpoint < core_end or (core_end == duration and midpoint == core_end):
                            segments.append(TranscriptionSegment(start, end, item.text))
            segments = clean_transcription_segments(segments)
            if not segments:
                raise RuntimeError("녹음에서 음성을 찾지 못했습니다. 입력 장치와 음량을 확인해 주세요.")
        return Transcription(
            text=" ".join(segment.text for segment in segments),
            language=Counter(languages).most_common(1)[0][0] if languages else language,
            duration=duration, segments=segments,
        )
