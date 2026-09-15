"""Memory-bounded PDF and PowerPoint text extraction with OCR fallback."""

from __future__ import annotations

import gc
import io
import logging
import math
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

from src.ocr import MAX_IMAGE_PIXELS, OCRReader


DOCUMENT_EXTENSIONS = {".pdf", ".ppt", ".pptx"}
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class ExtractedPDF:
    text: str
    page_count: int
    pages_with_text: int
    pages_ocr: int = 0


ExtractedDocument = ExtractedPDF
PDFSource = Union[Path, bytes, bytearray, memoryview]

logging.getLogger("pypdf").setLevel(logging.ERROR)


def extract_pdf_page(source: PDFSource, page_number: int) -> tuple[bytes, int]:
    """Return a standalone one-page PDF and the source document page count."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        raise RuntimeError("PDF 미리보기에는 pypdf가 필요합니다.") from exc

    stream: object
    should_close = False
    if isinstance(source, Path):
        stream = source.expanduser().resolve().open("rb")
        should_close = True
    else:
        stream = io.BytesIO(bytes(source))
    try:
        reader = PdfReader(stream, strict=False)
        if reader.is_encrypted:
            raise ValueError("암호로 잠긴 PDF는 먼저 잠금을 해제해 주세요.")
        page_count = len(reader.pages)
        if page_count <= 0:
            raise ValueError("PDF에 표시할 페이지가 없습니다.")
        if not 1 <= page_number <= page_count:
            raise ValueError(f"PDF 페이지는 1에서 {page_count} 사이여야 합니다.")
        writer = PdfWriter()
        writer.add_page(reader.pages[page_number - 1])
        output = io.BytesIO()
        writer.write(output)
        return output.getvalue(), page_count
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"PDF 페이지를 준비할 수 없습니다: {exc}") from exc
    finally:
        if should_close:
            stream.close()


def _visible_characters(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _merge_text(native: str, recognized: str) -> str:
    native = native.strip()
    recognized = recognized.strip()
    if not native:
        return recognized
    if not recognized:
        return native
    existing = {re.sub(r"\s+", "", line).casefold() for line in native.splitlines()}
    extra = [
        line
        for line in recognized.splitlines()
        if re.sub(r"\s+", "", line).casefold() not in existing
    ]
    return native + ("\n" + "\n".join(extra) if extra else "")


def _ocr_pdf_page(page: object, reader: OCRReader) -> str:
    import pymupdf

    rect = page.rect
    area = max(1.0, float(rect.width * rect.height))
    scale = min(2.0, math.sqrt(MAX_IMAGE_PIXELS / area))
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        colorspace=pymupdf.csRGB,
        alpha=False,
    )
    try:
        encoded = pixmap.tobytes("png")
    finally:
        del pixmap
    try:
        return reader.read(encoded)
    finally:
        del encoded
        gc.collect()


def extract_pdf_text(
    path: Path,
    *,
    progress: Optional[ProgressCallback] = None,
) -> ExtractedPDF:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PDF file not found: {source}")
    if source.suffix.lower() != ".pdf":
        raise ValueError("선택한 문서는 PDF가 아닙니다.")
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PDF 이미지 OCR에는 PyMuPDF가 필요합니다.") from exc

    sections: List[str] = []
    pages_ocr = 0
    reader: Optional[OCRReader] = None
    page_count = 0
    try:
        document = pymupdf.open(source)
        try:
            if document.needs_pass and not document.authenticate(""):
                raise ValueError("암호로 잠긴 PDF는 먼저 잠금을 해제해 주세요.")
            page_count = document.page_count
            for index in range(page_count):
                page = document.load_page(index)
                native = page.get_text("text", sort=True).strip()
                has_images = bool(page.get_images(full=True))
                needs_ocr = _visible_characters(native) < 40 or (
                    has_images and _visible_characters(native) < 160
                )
                if needs_ocr:
                    if progress:
                        progress(
                            f"{index + 1}/{page_count}페이지 이미지 글자를 인식하고 있습니다."
                        )
                    if reader is None:
                        reader = OCRReader()
                    recognized = _ocr_pdf_page(page, reader)
                    native = _merge_text(native, recognized)
                    if recognized.strip():
                        pages_ocr += 1
                if native.strip():
                    sections.append(f"## {index + 1}페이지\n\n{native.strip()}")
                del page
        finally:
            document.close()
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"PDF 내용을 읽을 수 없습니다: {exc}") from exc
    finally:
        if reader is not None:
            reader.close()
        gc.collect()
    if not sections:
        raise ValueError("PDF의 텍스트와 이미지에서 읽을 수 있는 글자를 찾지 못했습니다.")
    return ExtractedPDF("\n\n".join(sections), page_count, len(sections), pages_ocr)


def _slide_number(name: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", name)
    return int(match.group(1)) if match else 0


def _slide_image_paths(archive: zipfile.ZipFile, slide_name: str) -> list[str]:
    slide_path = Path(slide_name)
    rel_name = str(
        slide_path.parent / "_rels" / f"{slide_path.name}.rels"
    ).replace(os.sep, "/")
    try:
        root = ET.fromstring(archive.read(rel_name))
    except KeyError:
        return []
    names = set(archive.namelist())
    paths = []
    for relation in root:
        if not str(relation.attrib.get("Type", "")).endswith("/image"):
            continue
        if relation.attrib.get("TargetMode") == "External":
            continue
        target = relation.attrib.get("Target", "")
        resolved = posixpath.normpath(
            posixpath.join(posixpath.dirname(slide_name), target)
        )
        if resolved in names and resolved not in paths:
            paths.append(resolved)
    return paths


def extract_pptx_text(
    path: Path,
    *,
    progress: Optional[ProgressCallback] = None,
) -> ExtractedDocument:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PowerPoint file not found: {source}")
    if source.suffix.lower() != ".pptx":
        raise ValueError("선택한 문서는 PPTX가 아닙니다.")

    sections: list[str] = []
    pages_ocr = 0
    reader: Optional[OCRReader] = None
    slides: list[str] = []
    try:
        with zipfile.ZipFile(source) as archive:
            slides = sorted(
                (
                    name
                    for name in archive.namelist()
                    if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
                ),
                key=_slide_number,
            )
            for index, slide_name in enumerate(slides, start=1):
                root = ET.fromstring(archive.read(slide_name))
                native_lines = [
                    (node.text or "").strip()
                    for node in root.iter()
                    if node.tag.endswith("}t") and (node.text or "").strip()
                ]
                text = "\n".join(native_lines)
                recognized_parts = []
                image_paths = _slide_image_paths(archive, slide_name)
                for image_index, image_path in enumerate(image_paths, start=1):
                    if progress:
                        progress(
                            f"{index}/{len(slides)}슬라이드 이미지 "
                            f"{image_index}/{len(image_paths)}의 글자를 인식하고 있습니다."
                        )
                    if reader is None:
                        reader = OCRReader()
                    with archive.open(image_path) as image_stream:
                        recognized = reader.read(image_stream)
                    if recognized.strip():
                        recognized_parts.append(recognized)
                recognized_text = "\n".join(recognized_parts)
                text = _merge_text(text, recognized_text)
                if recognized_text.strip():
                    pages_ocr += 1
                if text.strip():
                    sections.append(f"## {index}슬라이드\n\n{text.strip()}")
                del root, native_lines, recognized_parts
                gc.collect()
    except (ValueError, zipfile.BadZipFile):
        raise ValueError("PPTX 파일이 손상되었거나 올바른 PowerPoint 형식이 아닙니다.")
    except Exception as exc:
        raise RuntimeError(f"PPTX 내용을 읽을 수 없습니다: {exc}") from exc
    finally:
        if reader is not None:
            reader.close()
        gc.collect()
    if not sections:
        raise ValueError("PPTX의 텍스트와 이미지에서 읽을 수 있는 글자를 찾지 못했습니다.")
    return ExtractedDocument("\n\n".join(sections), len(slides), len(sections), pages_ocr)


def _soffice_path() -> Optional[str]:
    found = shutil.which("soffice") or shutil.which("libreoffice")
    if found:
        return found
    candidates = [
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        r"C:\Program Files\LibreOffice\program\soffice.exe",
    ]
    return next((candidate for candidate in candidates if Path(candidate).is_file()), None)


def _convert_ppt_to_pptx(path: Path, destination: Path) -> Path:
    executable = _soffice_path()
    if not executable:
        raise RuntimeError(
            "구형 PPT 파일을 읽으려면 LibreOffice가 필요합니다. "
            "PPTX로 저장한 뒤 다시 추가해 주세요."
        )
    command = [
        executable,
        "--headless",
        f"-env:UserInstallation={destination.as_uri()}/profile",
        "--convert-to",
        "pptx",
        "--outdir",
        str(destination),
        str(path),
    ]
    options = {
        "capture_output": True,
        "text": True,
        "timeout": 300,
        "check": False,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    completed = subprocess.run(command, **options)
    converted = destination / f"{path.stem}.pptx"
    if completed.returncode != 0 or not converted.is_file():
        detail = (completed.stderr or completed.stdout or "변환 결과 없음").strip()
        raise RuntimeError(f"PPT를 PPTX로 변환하지 못했습니다: {detail}")
    return converted


def extract_document_text(
    path: Path,
    *,
    progress: Optional[ProgressCallback] = None,
) -> ExtractedDocument:
    source = path.expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(source, progress=progress)
    if suffix == ".pptx":
        return extract_pptx_text(source, progress=progress)
    if suffix == ".ppt":
        with tempfile.TemporaryDirectory(prefix="lass-ppt-") as directory:
            converted = _convert_ppt_to_pptx(source, Path(directory))
            return extract_pptx_text(converted, progress=progress)
    raise ValueError("PDF, PPT 또는 PPTX 문서를 선택해 주세요.")
