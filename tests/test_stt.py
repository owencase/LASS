import tempfile
import time
import unittest
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.stt.audio_utils import AudioEnhancement, enhance_speech_audio
from src.stt.bigvox import BigvoxTranscriber
from src.stt.factory import create_transcriber
from src.stt.sensevoice import SenseVoiceTranscriber
from src.stt.transcriber import (
    TranscriptionSegment,
    WhisperTranscriber,
    assess_transcription_quality,
    clean_transcription_segments,
)


class WhisperTranscriberTests(unittest.TestCase):
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

    def test_uses_normalized_audio_and_fast_speech_vad_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "quiet.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * 160)
            model = Mock()
            model.transcribe.return_value = (
                iter([SimpleNamespace(start=0, end=1, text=" 빠른 발화 ")]),
                SimpleNamespace(language="ko", duration=1.0),
            )
            transcriber = WhisperTranscriber()
            transcriber._model = model
            prepared = AudioEnhancement(source, 4.0, -28.0, -16.0, 2000)

            with patch(
                "src.stt.transcriber.prepared_speech_audio"
            ) as prepare_audio:
                prepare_audio.return_value.__enter__.return_value = prepared
                result = transcriber.transcribe(source, language="ko")

            self.assertEqual(result.text, "빠른 발화")
            prepare_audio.assert_called_once_with(
                source.resolve(), target_lufs=-16.0
            )
            options = model.transcribe.call_args.kwargs
            self.assertEqual(options["beam_size"], 8)
            self.assertFalse(options["condition_on_previous_text"])
            self.assertEqual(options["temperature"], (0.0, 0.2, 0.4))
            self.assertEqual(options["no_repeat_ngram_size"], 3)
            self.assertEqual(options["vad_parameters"]["min_speech_duration_ms"], 100)
            self.assertEqual(options["vad_parameters"]["speech_pad_ms"], 400)

    def test_retries_without_vad_when_no_speech_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "quiet.wav"
            source.write_bytes(b"placeholder")
            model = Mock()
            model.transcribe.side_effect = [
                (iter(()), SimpleNamespace(language="ko", duration=1.0)),
                (
                    iter([SimpleNamespace(start=0, end=1, text="작은 목소리")]),
                    SimpleNamespace(language="ko", duration=1.0),
                ),
            ]
            transcriber = WhisperTranscriber()
            transcriber._model = model
            prepared = AudioEnhancement(source, 8.0, -34.0, -16.0, 1000)

            with patch(
                "src.stt.transcriber.prepared_speech_audio"
            ) as prepare_audio:
                prepare_audio.return_value.__enter__.return_value = prepared
                result = transcriber.transcribe(source)

            self.assertEqual(result.text, "작은 목소리")
            self.assertTrue(model.transcribe.call_args_list[0].kwargs["vad_filter"])
            self.assertFalse(model.transcribe.call_args_list[1].kwargs["vad_filter"])

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

    def test_long_recording_is_transcribed_in_context_reset_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "lecture.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * int(2.2 * 16000))
            model = Mock()
            model.transcribe.side_effect = [
                (
                    iter([SimpleNamespace(start=0.2, end=0.8, text="첫 구간")]),
                    SimpleNamespace(language="ko", duration=1.1),
                ),
                (
                    iter([SimpleNamespace(start=0.2, end=0.8, text="둘째 구간")]),
                    SimpleNamespace(language="ko", duration=1.2),
                ),
                (
                    iter([SimpleNamespace(start=0.05, end=0.15, text="마지막")]),
                    SimpleNamespace(language="ko", duration=0.3),
                ),
            ]
            transcriber = WhisperTranscriber()
            transcriber._model = model
            transcriber.LONG_AUDIO_SECONDS = 1
            transcriber.CHUNK_SECONDS = 1
            transcriber.CHUNK_OVERLAP_SECONDS = 0.1
            prepared = AudioEnhancement(source, 1.0, -16.0, -16.0, 1000)

            with patch(
                "src.stt.transcriber.prepared_speech_audio"
            ) as prepare_audio:
                prepare_audio.return_value.__enter__.return_value = prepared
                result = transcriber.transcribe(source, language="ko")

            self.assertEqual(model.transcribe.call_count, 3)
            self.assertEqual(
                [item.text for item in result.segments],
                ["첫 구간", "둘째 구간", "마지막"],
            )
            self.assertAlmostEqual(result.segments[1].start, 1.1)
            self.assertAlmostEqual(result.duration, 2.2)

    def test_long_recording_without_detected_speech_does_not_retry_as_one_block(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "silent-lecture.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * int(2.2 * 16000))
            model = Mock()
            model.transcribe.side_effect = [
                (iter(()), SimpleNamespace(language="ko", duration=1.1)),
                (iter(()), SimpleNamespace(language="ko", duration=1.2)),
                (iter(()), SimpleNamespace(language="ko", duration=0.3)),
            ]
            transcriber = WhisperTranscriber()
            transcriber._model = model
            transcriber.LONG_AUDIO_SECONDS = 1
            transcriber.CHUNK_SECONDS = 1
            transcriber.CHUNK_OVERLAP_SECONDS = 0.1
            prepared = AudioEnhancement(source, 1.0, -16.0, -16.0, 0)

            with patch(
                "src.stt.transcriber.prepared_speech_audio"
            ) as prepare_audio:
                prepare_audio.return_value.__enter__.return_value = prepared
                with self.assertRaisesRegex(RuntimeError, "긴 녹음에서 음성을"):
                    transcriber.transcribe(source, language="ko")

            self.assertEqual(model.transcribe.call_count, 3)


class SenseVoiceTranscriberTests(unittest.TestCase):
    def test_model_initialization_is_serialized_across_instances(self) -> None:
        counter_lock = Lock()
        active = 0
        maximum_active = 0

        def build_model(**_options):
            nonlocal active, maximum_active
            with counter_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.05)
            with counter_lock:
                active -= 1
            return Mock()

        transcribers = [SenseVoiceTranscriber(), SenseVoiceTranscriber()]
        with patch("funasr.AutoModel", side_effect=build_model):
            with ThreadPoolExecutor(max_workers=2) as executor:
                models = list(executor.map(lambda item: item._get_model(), transcribers))

        self.assertEqual(len(models), 2)
        self.assertEqual(maximum_active, 1)

    def test_factory_selects_sensevoice_without_removing_whisper(self) -> None:
        sensevoice = create_transcriber(
            backend="sensevoice",
            model_name="FunAudioLLM/SenseVoiceSmall",
            device="auto",
            compute_type="float32",
            target_lufs=-16.0,
        )
        whisper = create_transcriber(
            backend="faster-whisper",
            model_name="large-v3",
            device="auto",
            compute_type="int8",
            target_lufs=-16.0,
        )

        self.assertIsInstance(sensevoice, SenseVoiceTranscriber)
        self.assertIsInstance(whisper, WhisperTranscriber)

    def test_uses_funasr_vad_and_preserves_sentence_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "lecture.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * 32000)
            model = Mock()
            model.generate.return_value = [
                {
                    "text": "<|ko|><|NEUTRAL|><|Speech|>첫 문장 둘째 문장",
                    "sentence_info": [
                        {
                            "start": 120,
                            "end": 840,
                            "text": "<|ko|><|NEUTRAL|><|Speech|>첫 문장",
                        },
                        {
                            "start": 1000,
                            "end": 1880,
                            "text": "<|ko|><|NEUTRAL|><|Speech|>둘째 문장",
                        },
                    ],
                }
            ]
            transcriber = SenseVoiceTranscriber()
            transcriber._model = model
            prepared = AudioEnhancement(source, 1.0, -16.0, -16.0, 1000)

            with patch("src.stt.sensevoice.prepared_speech_audio") as prepare_audio:
                prepare_audio.return_value.__enter__.return_value = prepared
                result = transcriber.transcribe(source)

            self.assertEqual(result.text, "첫 문장 둘째 문장")
            self.assertEqual(result.language, "ko")
            self.assertAlmostEqual(result.duration, 2.0)
            self.assertAlmostEqual(result.segments[0].start, 0.12)
            self.assertAlmostEqual(result.segments[1].end, 1.88)
            options = model.generate.call_args.kwargs
            self.assertEqual(options["language"], "auto")
            self.assertTrue(options["merge_vad"])
            self.assertTrue(options["sentence_timestamp"])

    def test_empty_funasr_result_is_reported_as_no_speech(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "silent.wav"
            source.write_bytes(b"placeholder")
            model = Mock()
            model.generate.return_value = [{"text": "", "sentence_info": []}]
            transcriber = SenseVoiceTranscriber()
            transcriber._model = model
            prepared = AudioEnhancement(source, 1.0, -16.0, -16.0, 0)

            with patch("src.stt.sensevoice.prepared_speech_audio") as prepare_audio:
                prepare_audio.return_value.__enter__.return_value = prepared
                with self.assertRaisesRegex(
                    RuntimeError, "음성을 찾지 못했습니다"
                ):
                    transcriber.transcribe(source, language="ko")

    def test_retries_audible_input_without_vad_when_vad_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "speech.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x01\x00" * 32000)
            model = Mock()
            model.generate.return_value = [{"text": "", "sentence_info": []}]
            model.inference.return_value = [
                {"text": "<|ko|><|NEUTRAL|><|Speech|>다시 인식한 문장"}
            ]
            transcriber = SenseVoiceTranscriber()
            transcriber._model = model
            prepared = AudioEnhancement(source, 1.0, -30.0, -30.0, 1)

            with patch("src.stt.sensevoice.prepared_speech_audio") as prepare_audio:
                prepare_audio.return_value.__enter__.return_value = prepared
                result = transcriber.transcribe(source, language="ko")

            self.assertEqual(result.text, "다시 인식한 문장")
            model.inference.assert_called_once()


class BigvoxTranscriberTests(unittest.TestCase):
    def test_factory_selects_bigvox(self) -> None:
        transcriber = create_transcriber(
            backend="bigvox",
            model_name="bigdefence/Bigvox-Exaone4-Audio",
            device="auto",
            compute_type="float16",
            target_lufs=-16.0,
        )

        self.assertIsInstance(transcriber, BigvoxTranscriber)

    def test_transcribes_long_audio_in_thirty_second_model_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "lecture.wav"
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * int(2.2 * 16000))
            runtime = Mock()
            runtime.transcribe.side_effect = ["첫 구간", "둘째 구간", "마지막"]
            transcriber = BigvoxTranscriber()
            transcriber._runtime = runtime
            transcriber.CHUNK_SECONDS = 1
            prepared = AudioEnhancement(source, 1.0, -16.0, -16.0, 1000)

            with patch("src.stt.bigvox.prepared_speech_audio") as prepare_audio:
                prepare_audio.return_value.__enter__.return_value = prepared
                result = transcriber.transcribe(source, language="ko")

            self.assertEqual(result.text, "첫 구간 둘째 구간 마지막")
            self.assertEqual(result.language, "ko")
            self.assertAlmostEqual(result.duration, 2.2)
            self.assertEqual(runtime.transcribe.call_count, 3)
            self.assertEqual(
                [(segment.start, segment.end) for segment in result.segments],
                [(0.0, 1.0), (1.0, 2.0), (2.0, 2.2)],
            )

    def test_rejects_non_korean_language_hint(self) -> None:
        transcriber = BigvoxTranscriber()
        with self.assertRaisesRegex(ValueError, "한국어"):
            transcriber.transcribe(Path("unused.wav"), language="en")


if __name__ == "__main__":
    unittest.main()
