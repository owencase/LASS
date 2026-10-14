"""Bounded, page-at-a-time OCR."""

from __future__ import annotations

import gc
import io
import re
from statistics import median
from typing import Any, BinaryIO


MAX_IMAGE_EDGE = 2400
MAX_IMAGE_PIXELS = 4_000_000


def layout_text(result):
    """Recover line order, visible word gaps and code indentation from OCR boxes."""
    texts = list(result.txts or ())
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) != len(texts):
        return "\n".join(str(text) for text in texts if str(text).strip())
    words = getattr(result, "word_results", ()) or ()
    rows = []
    for index, (box, text) in enumerate(zip(boxes, texts)):
        x = min(float(p[0]) for p in box)
        y = min(float(p[1]) for p in box)
        height = max(float(p[1]) for p in box) - y
        width = max(float(p[0]) for p in box) - x
        unit = max(1, width / max(1, len(str(text))))
        if index < len(words) and isinstance(words[index], (tuple, list)):
            items = [item for item in words[index] if isinstance(item, (tuple, list)) and len(item) == 3 and item[2] is not None]
            if items and "".join(str(item[0]) for item in items).replace(" ", "") == str(text).replace(" ", ""):
                rebuilt, right = "", None
                for word, _, bounds in sorted(items, key=lambda item: min(p[0] for p in item[2])):
                    left = min(p[0] for p in bounds)
                    if right is not None and left - right > unit * .4:
                        rebuilt += " "
                    rebuilt += str(word)
                    right = max(p[0] for p in bounds)
                text = rebuilt
        rows.append((y, x, height, unit, str(text)))
    rows.sort()
    groups = []
    for row in rows:
        y, x, height, unit, text = row
        previous = groups[-1][0] if groups else None
        if previous and abs(y + height / 2 - previous[0] - previous[2] / 2) < max(height, previous[2]) * .6:
            groups[-1].append(row)
        else:
            groups.append([row])
    lines = []
    for group in groups:
        ordered = sorted(group, key=lambda row: row[1])
        first = ordered[0]
        lines.append((*first[:4], " ".join(row[4] for row in ordered)))
    baseline = min((line[1] for line in lines), default=0)
    unit = median([line[3] for line in lines]) if lines else 1
    output, in_code = [], False
    for _, x, _, _, text in lines:
        code = bool(re.search(r"\b(?:def|class|import|return|public|function|const|let)\b|[{};]|\b\w+\s*=|\w+\.\w+\(", text))
        if code != in_code:
            output.append("```" if code else "```")
            in_code = code
        indent = min(32, max(0, round((x - baseline) / max(1, unit)))) if code else 0
        output.append(" " * indent + text)
    if in_code:
        output.append("```")
    return "\n".join(output)


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
            return self._read_rapidocr(image)
        finally:
            del image
            gc.collect()

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
                    "Windows 이미지 OCR에는 rapidocr와 onnxruntime이 필요합니다."
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
        result = self._rapid(image, return_word_box=True, return_single_char_box=True)
        return layout_text(result)

    def close(self) -> None:
        self._rapid = None
        gc.collect()

    def __enter__(self) -> "OCRReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
