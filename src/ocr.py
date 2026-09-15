"""Bounded, page-at-a-time OCR with a native macOS fast path."""

from __future__ import annotations

import gc
import io
import sys
from typing import Any, BinaryIO


MAX_IMAGE_EDGE = 2400
MAX_IMAGE_PIXELS = 4_000_000


def _prepared_jpeg(source: bytes | BinaryIO) -> bytes:
    """Decode one image at a bounded size and return a compact RGB buffer."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("이미지 OCR에는 Pillow가 필요합니다.") from exc

    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(stream) as opened:
        opened.draft("RGB", (MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
        image = ImageOps.exif_transpose(opened)
        width, height = image.size
        if width <= 0 or height <= 0:
            return b""
        scale = min(
            1.0,
            MAX_IMAGE_EDGE / max(width, height),
            (MAX_IMAGE_PIXELS / (width * height)) ** 0.5,
        )
        if scale < 1.0:
            image.thumbnail(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        rgb = image.convert("RGB")
        output = io.BytesIO()
        rgb.save(output, format="JPEG", quality=90, optimize=False)
        return output.getvalue()


class OCRReader:
    """Recognize Korean and English while keeping only one image in memory."""

    def __init__(self) -> None:
        self._rapid: Any = None

    def read(self, source: bytes | BinaryIO) -> str:
        image = _prepared_jpeg(source)
        if not image:
            return ""
        try:
            if sys.platform == "darwin":
                return self._read_vision(image)
            return self._read_rapidocr(image)
        finally:
            del image
            gc.collect()

    @staticmethod
    def _read_vision(image: bytes) -> str:
        try:
            from Foundation import NSData
            import Vision
        except ImportError as exc:
            raise RuntimeError(
                "macOS 이미지 OCR에는 pyobjc-framework-Vision이 필요합니다."
            ) from exc

        data = NSData.dataWithBytes_length_(image, len(image))
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        if hasattr(request, "setAutomaticallyDetectsLanguage_"):
            request.setAutomaticallyDetectsLanguage_(True)
        try:
            request.setRecognitionLanguages_(["ko-KR", "en-US"])
        except Exception:
            pass
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
        succeeded, error = handler.performRequests_error_([request], None)
        if not succeeded:
            detail = f": {error}" if error is not None else ""
            raise RuntimeError(f"macOS Vision OCR을 실행할 수 없습니다{detail}")
        observations = list(request.results() or [])
        observations.sort(
            key=lambda item: (
                -float(item.boundingBox().origin.y),
                float(item.boundingBox().origin.x),
            )
        )
        lines = []
        for observation in observations:
            candidates = observation.topCandidates_(1)
            if candidates:
                text = str(candidates[0].string()).strip()
                if text:
                    lines.append(text)
        return "\n".join(lines)

    def _read_rapidocr(self, image: bytes) -> str:
        if self._rapid is None:
            try:
                from rapidocr import (
                    EngineType,
                    LangDet,
                    LangRec,
                    ModelType,
                    OCRVersion,
                    RapidOCR,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Windows/Linux 이미지 OCR에는 rapidocr와 onnxruntime이 필요합니다."
                ) from exc
            self._rapid = RapidOCR(
                params={
                    "Det.engine_type": EngineType.ONNXRUNTIME,
                    "Det.lang_type": LangDet.CH,
                    "Det.model_type": ModelType.MOBILE,
                    "Det.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.engine_type": EngineType.ONNXRUNTIME,
                    "Rec.lang_type": LangRec.KOREAN,
                    "Rec.model_type": ModelType.MOBILE,
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                }
            )
        result = self._rapid(image)
        return "\n".join(
            str(text).strip() for text in (result.txts or ()) if str(text).strip()
        )

    def close(self) -> None:
        self._rapid = None
        gc.collect()

    def __enter__(self) -> "OCRReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
