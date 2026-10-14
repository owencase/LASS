import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from src.stt.audio_utils import enhance_speech_audio
from src.stt.transcriber import TranscriptionSegment, assess_transcription_quality, clean_transcription_segments


class SpeechQualityTests(unittest.TestCase):
    def test_quiet_pcm_audio_is_amplified_for_recognition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "quiet.wav"
            samples = array("h", [700, -700] * 8000)
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(samples.tobytes())

            enhancement = enhance_speech_audio(source, root / "enhanced.wav")

            self.assertGreater(enhancement.gain, 5)
            self.assertAlmostEqual(enhancement.output_lufs, -16.0, places=1)
            with wave.open(str(enhancement.path), "rb") as enhanced:
                output_samples = array("h")
                output_samples.frombytes(enhanced.readframes(enhanced.getnframes()))
            self.assertGreater(max(abs(value) for value in output_samples), 3500)
            import numpy as np
            import pyloudnorm as pyln

            normalized = np.asarray(output_samples, dtype=np.float32) / 32768.0
            measured_lufs = pyln.Meter(16000).integrated_loudness(normalized)
            self.assertAlmostEqual(measured_lufs, -16.0, places=1)


    def test_collapses_runaway_repetition_but_keeps_a_repeated_cue_twice(self) -> None:
        phrase = "전 페이지로 돌아오시기 바랍니다"
        segments = [
            TranscriptionSegment(0, 10, " ".join([phrase] * 5)),
            TranscriptionSegment(10, 12, phrase),
            TranscriptionSegment(12, 14, phrase),
            TranscriptionSegment(14, 16, phrase),
        ]

        cleaned = clean_transcription_segments(segments)

        self.assertEqual([segment.text for segment in cleaned], [phrase, phrase])


    def test_flags_repetitive_sparse_long_transcription(self) -> None:
        segments = [
            TranscriptionSegment(index * 120, (index + 1) * 120, "같은 안내 문장")
            for index in range(8)
        ]

        quality = assess_transcription_quality(segments, duration=1200)

        self.assertTrue(quality.suspicious)
        self.assertEqual(quality.repeated_segment_ratio, 1.0)
        self.assertGreaterEqual(len(quality.warnings), 2)
