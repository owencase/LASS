"""Audio validation and speech-oriented loudness normalization helpers."""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
import wave
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


SUPPORTED_MEDIA_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".webm",
}


def validate_media_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Media file not found: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
        raise ValueError(f"Unsupported media format: {resolved.suffix or '(none)'}")
    return resolved


def ensure_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("FFmpeg is not installed or is not available on PATH")
    return executable


def extract_audio(source: Path, destination: Path) -> Path:
    """Convert a media file to mono 16 kHz WAV for speech recognition."""
    source = validate_media_file(source)
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ensure_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or "unknown FFmpeg error"
        raise RuntimeError(f"Could not extract audio: {message}") from exc
    return destination


@dataclass(frozen=True)
class AudioEnhancement:
    """Information about the speech WAV produced for recognition."""

    path: Path
    gain: float
    source_lufs: float
    output_lufs: float
    source_peak: int


def _decode_pcm16(source: Path, destination: Path) -> None:
    """Decode arbitrary supported media to a streaming 16 kHz mono WAV."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required for speech audio preparation. "
            "It is installed with faster-whisper."
        ) from exc
    av_error = getattr(av, "FFmpegError", getattr(av, "AVError", Exception))

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with av.open(str(source)) as container, wave.open(str(destination), "wb") as output:
            audio_streams = container.streams.audio
            if not audio_streams:
                raise ValueError("The media file does not contain an audio stream")
            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=16000
            )
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            for frame in container.decode(audio_streams[0]):
                converted = resampler.resample(frame)
                converted_frames = converted if isinstance(converted, list) else [converted]
                for converted_frame in converted_frames:
                    if converted_frame is not None:
                        output.writeframesraw(converted_frame.to_ndarray().tobytes())
            flushed = resampler.resample(None)
            flushed_frames = flushed if isinstance(flushed, list) else [flushed]
            for converted_frame in flushed_frames:
                if converted_frame is not None:
                    output.writeframesraw(converted_frame.to_ndarray().tobytes())
    except (ValueError, av_error) as exc:
        raise RuntimeError(f"Could not decode speech audio: {exc}") from exc


def _measure_loudness(path: Path) -> tuple[float, int]:
    """BS.1770 mono gating with stateful filters and 400 ms of PCM in memory.

    Store one scalar per 100 ms, never a copy of the complete decoded waveform.
    Coefficients come from the same pyloudnorm K-weighting meter as before.
    """
    import numpy as np
    import pyloudnorm as pyln
    from scipy.signal import lfilter
    from collections import deque

    with wave.open(str(path), "rb") as audio:
        rate, count = audio.getframerate(), audio.getnframes()
        if count <= 0:
            raise ValueError("The media file contains no decodable audio samples")
        meter = pyln.Meter(rate)
        hop, block = round(rate * .1), round(rate * .4)
        if count < block:
            samples = np.frombuffer(audio.readframes(count), dtype="<i2").astype(np.float64)
            peak = int(np.max(np.abs(samples)))
            samples = np.tile(samples / 32768.0, math.ceil(block / count))[:block]
            return float(meter.integrated_loudness(samples)), peak
        filters = [(f.b, f.a, f.passband_gain, np.zeros(max(len(f.a), len(f.b))-1))
                   for f in meter._filters.values()]
        energy, recent, peak = [], deque(maxlen=4), 0
        # Match pyloudnorm's rounded final (possibly partial) gating block.
        blocks = int(np.round((count - block) / hop)) + 1
        for step in range(blocks + 3):
            pcm = np.frombuffer(audio.readframes(hop), dtype="<i2").astype(np.float64)
            if len(pcm):
                peak = max(peak, int(np.max(np.abs(pcm))))
                samples = pcm / 32768.0
                for index, (b, a, gain, state) in enumerate(filters):
                    samples, state = lfilter(b, a, samples, zi=state)
                    samples *= gain
                    filters[index] = (b, a, gain, state)
                recent.append(float(np.dot(samples, samples)))
            else:
                recent.append(0.0)
            if step >= 3:
                energy.append(sum(recent) / block)
        # Peak safety also covers samples beyond the final loudness gate.
        remaining = np.frombuffer(audio.readframes(hop), dtype="<i2").astype(np.int32)
        if len(remaining):
            peak = max(peak, int(np.max(np.abs(remaining))))
    z = np.asarray(energy)
    absolute = 10 ** ((-70 + .691) / 10)
    gated = z[z >= absolute]
    if not len(gated):
        return -math.inf, peak
    relative = gated.mean() / 10
    gated = z[(z > absolute) & (z > relative)]
    return (-.691 + 10 * math.log10(float(gated.mean()))) if len(gated) else -math.inf, peak


def enhance_speech_audio(
    source: Path,
    destination: Path,
    *,
    target_lufs: float = -16.0,
    peak_limit_dbfs: float = -1.0,
) -> AudioEnhancement:
    """Create a BS.1770 loudness-normalized speech WAV for recognition.

    Integrated loudness is measured with pyloudnorm. The resulting gain is then
    applied in streaming chunks so long recordings do not require a second full
    audio copy in memory. A -1 dBFS peak ceiling prevents clipping, and the
    original media remains untouched.
    """
    source = validate_media_file(source)
    destination = destination.expanduser().resolve()
    if not -70.0 < target_lufs < 0.0:
        raise ValueError("Audio normalization settings are invalid")
    if not -20.0 <= peak_limit_dbfs <= 0.0:
        raise ValueError("Audio peak limit must be between -20 and 0 dBFS")

    try:
        import numpy as np
        import pyloudnorm as pyln
    except ImportError as exc:
        raise RuntimeError(
            "pyloudnorm, NumPy, and SciPy are required for LUFS normalization. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    decoded = destination.with_name(f"{destination.stem}-decoded.wav")
    try:
        _decode_pcm16(source, decoded)
        source_lufs, peak = _measure_loudness(decoded)

        if math.isfinite(source_lufs) and peak > 0:
            loudness_gain = 10.0 ** ((target_lufs - source_lufs) / 20.0)
            peak_ceiling = 10.0 ** (peak_limit_dbfs / 20.0)
            peak_gain = peak_ceiling / (peak / 32768.0)
            gain = min(loudness_gain, peak_gain)
            output_lufs = source_lufs + 20.0 * math.log10(gain)
        else:
            gain = 1.0
            output_lufs = source_lufs

        with wave.open(str(decoded), "rb") as input_wave, wave.open(
            str(destination), "wb"
        ) as output_wave:
            output_wave.setparams(input_wave.getparams())
            sample_rate = input_wave.getframerate()
            while chunk := input_wave.readframes(sample_rate * 4):
                samples = array("h")
                samples.frombytes(chunk)
                if sys.byteorder != "little":
                    samples.byteswap()
                for index, value in enumerate(samples):
                    samples[index] = max(-32768, min(32767, round(value * gain)))
                if sys.byteorder != "little":
                    samples.byteswap()
                output_wave.writeframesraw(samples.tobytes())
        return AudioEnhancement(destination, gain, source_lufs, output_lufs, peak)
    finally:
        decoded.unlink(missing_ok=True)


@contextmanager
def prepared_speech_audio(
    source: Path, *, target_lufs: float = -16.0
) -> Iterator[AudioEnhancement]:
    """Yield a temporary enhanced WAV and remove it after transcription."""
    source = validate_media_file(source)
    with tempfile.TemporaryDirectory(prefix="lass-stt-") as temporary_directory:
        destination = Path(temporary_directory) / "speech-normalized.wav"
        yield enhance_speech_audio(source, destination, target_lufs=target_lufs)
