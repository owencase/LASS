"""Disk-cached, bounded visible-page previews for PDF and PowerPoint."""

import base64
import hashlib
import io
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path
from threading import RLock
from subprocess import TimeoutExpired

from src.pdf_utils import convert_presentation, _presentation_slides, _slide_image_paths

_lock = RLock()
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}


def _color(node, default):
    if node is None:
        return default
    rgb = node.find(".//a:srgbClr", NS)
    value = rgb.get("val", "") if rgb is not None else ""
    return "#" + value if re.fullmatch(r"[a-fA-F0-9]{6}", value) else default


def _pptx_page(source, output, page_number):
    """Basic content preview when neither Office renderer is installed."""
    from PIL import Image, UnidentifiedImageError

    with zipfile.ZipFile(source) as archive:
        slides = _presentation_slides(archive)
        # Presentation order can differ from slide file numbering.
        try:
            presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
            size = presentation.find("p:sldSz", NS)
            width, height = int(size.get("cx")), int(size.get("cy"))
        except (KeyError, AttributeError, TypeError):
            width, height = 12192000, 6858000
        if not 1 <= page_number <= len(slides):
            raise ValueError(f"문서 페이지는 1에서 {len(slides)} 사이여야 합니다.")
        root = ET.fromstring(archive.read(slides[page_number - 1]))
        factor = 960 / max(1, width)
        canvas_height = height * factor
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{canvas_height}" viewBox="0 0 960 {canvas_height}">',
                 f'<rect width="100%" height="100%" fill="{_color(root.find("p:cSld/p:bg", NS), "#ffffff")}"/>']
        tree = root.find("p:cSld/p:spTree", NS)
        if tree is None:
            raise ValueError("슬라이드 내용을 찾을 수 없습니다.")
        y_fallback = 30
        image_paths = _slide_image_paths(archive, slides[page_number - 1])
        for shape in tree:
            xfrm = shape.find(".//a:xfrm", NS)
            off = xfrm.find("a:off", NS) if xfrm is not None else None
            ext = xfrm.find("a:ext", NS) if xfrm is not None else None
            x = int(off.get("x", 0)) * factor if off is not None else 30
            y = int(off.get("y", 0)) * factor if off is not None else y_fallback
            w = int(ext.get("cx", 0)) * factor if ext is not None else 900
            h = int(ext.get("cy", 0)) * factor if ext is not None else 120
            if shape.tag.endswith("}pic"):
                blip = shape.find(".//a:blip", NS)
                rel_id = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed") if blip is not None else None
                rels_name = str(Path(slides[page_number - 1]).parent / "_rels" / (Path(slides[page_number - 1]).name + ".rels")).replace("\\", "/")
                rels = ET.fromstring(archive.read(rels_name))
                target = next((r.get("Target") for r in rels if r.get("Id") == rel_id), "")
                image_path = posixpath.normpath(target.lstrip("/") if target.startswith("/") else posixpath.join(posixpath.dirname(slides[page_number - 1]), target))
                if image_path in image_paths and archive.getinfo(image_path).file_size <= 32 * 1024 * 1024:
                    try:
                        with archive.open(image_path) as handle, Image.open(handle) as image:
                            if image.width * image.height > 16_000_000:
                                continue
                            image.thumbnail((1600, 1600))
                            data = io.BytesIO()
                            image.convert("RGB").save(data, "JPEG", quality=82)
                    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                        continue
                    encoded = base64.b64encode(data.getvalue()).decode("ascii")
                    parts.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/jpeg;base64,{encoded}"/>')
                continue
            body = shape.find("p:txBody", NS)
            if body is None:
                continue
            fill = _color(shape.find("p:spPr/a:solidFill", NS), "none")
            parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"/>')
            paragraphs = []
            for paragraph in body.findall("a:p", NS):
                runs = []
                for run in paragraph.findall("a:r", NS):
                    props = run.find("a:rPr", NS)
                    size = int(props.get("sz", "1800")) / 100 * 12700 * factor if props is not None else 18
                    color = _color(props, "#172033")
                    bold = "bold" if props is not None and props.get("b") == "1" else "normal"
                    runs.append(f'<span style="font-size:{size}px;color:{color};font-weight:{bold}">{escape(run.findtext("a:t", "", NS))}</span>')
                paragraphs.append('<p style="margin:0;line-height:1.15">' + "".join(runs) + '</p>')
            parts.append(f'<foreignObject x="{x}" y="{y}" width="{w}" height="{h}"><div xmlns="http://www.w3.org/1999/xhtml" style="font-family:Segoe UI,Malgun Gothic,sans-serif;padding:3px 7px;overflow:hidden;overflow-wrap:anywhere">' + "".join(paragraphs) + '</div></foreignObject>')
            y_fallback += h
        parts.append('</svg>')
        output.write_text("".join(parts), encoding="utf-8")
        return len(slides)


def preview_page(source: Path, cache_dir: Path, page_number: int) -> tuple[Path, int, bool]:
    """Return a cached image path, page count, and whether a basic preview is used."""
    import pymupdf

    source = source.resolve()
    stat = source.stat()
    digest = hashlib.sha256(f"preview-v2:{source}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()
    folder = cache_dir / digest
    with _lock:
        folder.mkdir(parents=True, exist_ok=True)
        pdf = source
        basic = False
        if source.suffix.lower() != ".pdf":
            pdf = folder / f"{source.stem}.pdf"
            failed_marker = folder / "basic-preview"
            if not pdf.is_file() and not failed_marker.exists():
                try:
                    convert_presentation(source, folder)
                except (RuntimeError, OSError, TimeoutExpired):
                    if source.suffix.lower() != ".pptx":
                        raise
                    failed_marker.touch()
            basic = not pdf.is_file()
        output = folder / f"page-{page_number}{'.svg' if basic else '.jpg'}"
        count_file = folder / "page-count.txt"
        if output.is_file() and count_file.is_file():
            return output, int(count_file.read_text()), basic
        if basic:
            try:
                count = _pptx_page(source, output, page_number)
            except (zipfile.BadZipFile, ET.ParseError, KeyError, TypeError) as exc:
                raise ValueError("PowerPoint 파일이 손상되었거나 필수 슬라이드 정보를 읽을 수 없습니다.") from exc
        else:
            with pymupdf.open(pdf) as document:
                if document.needs_pass and not document.authenticate(""):
                    raise ValueError("암호로 잠긴 문서는 먼저 잠금을 해제해 주세요.")
                count = document.page_count
                if not 1 <= page_number <= count:
                    raise ValueError(f"문서 페이지는 1에서 {count} 사이여야 합니다.")
                page = document.load_page(page_number - 1)
                # Bound raster cost even for unusually large page dimensions.
                scale = min(2, 1600 / max(page.rect.width, page.rect.height, 1))
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False)
                pixmap.save(output, jpg_quality=85)
        count_file.write_text(str(count))
        return output, count, basic
