"""Migrate the old folder layout into the two-database layout."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .lecture_repository import LectureRepository, LectureSegment


MIGRATION_KEY = "legacy_markdown_import_v1"
SEGMENT_PATTERN = re.compile(
    r"^-\s+`(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})`\s*(?P<text>.*)$"
)
UPLOAD_PREFIX_PATTERN = re.compile(r"^[0-9a-fA-F]{8}_")


@dataclass(frozen=True)
class LegacyReport:
    title: str
    prompt_name: str
    language: Optional[str]
    summary: str
    transcript: str
    segments: List[LectureSegment]


@dataclass(frozen=True)
class MigrationResult:
    imported_reports: int = 0
    embedded_media: int = 0
    removed_directories: int = 0
    warnings: tuple[str, ...] = ()


def _section(lines: List[str], heading: str, next_heading: Optional[str]) -> str:
    try:
        start = lines.index(heading) + 1
    except ValueError:
        return ""
    if next_heading is None:
        end = len(lines)
    else:
        try:
            end = lines.index(next_heading, start)
        except ValueError:
            end = len(lines)
    return "\n".join(lines[start:end]).strip()


def parse_legacy_report(path: Path) -> LegacyReport:
    lines = path.read_text(encoding="utf-8").splitlines()
    heading = next((line[2:].strip() for line in lines if line.startswith("# ")), path.stem)
    title = heading[:-3].strip() if heading.endswith(" 요약") else heading
    title = UPLOAD_PREFIX_PATTERN.sub("", title)
    prompt_name = ""
    language: Optional[str] = None
    for line in lines:
        if line.startswith("- 요약 형식:"):
            prompt_name = line.split(":", 1)[1].strip()
        elif line.startswith("- 감지 언어:"):
            value = line.split(":", 1)[1].strip()
            language = None if value in {"", "unknown"} else value

    summary = _section(lines, "## 요약", "## 타임스탬프 스크립트")
    segments: List[LectureSegment] = []
    for line in lines:
        match = SEGMENT_PATTERN.match(line)
        if not match:
            continue
        start = (
            int(match.group("hours")) * 3600
            + int(match.group("minutes")) * 60
            + int(match.group("seconds"))
        )
        segments.append(
            LectureSegment(start=float(start), end=float(start), text=match.group("text"))
        )
    transcript = " ".join(segment.text for segment in segments)
    return LegacyReport(
        title=title,
        prompt_name=prompt_name,
        language=language,
        summary=summary,
        transcript=transcript,
        segments=segments,
    )


def import_legacy_reports(
    repository: LectureRepository, input_directory: Path, output_directory: Path
) -> int:
    """Import old Markdown reports and their media BLOBs exactly once."""
    if repository.get_metadata(MIGRATION_KEY) == "complete":
        return 0

    input_files = (
        {path.stem: path for path in input_directory.iterdir() if path.is_file()}
        if input_directory.is_dir()
        else {}
    )
    imported = 0
    if output_directory.is_dir():
        for report_path in sorted(output_directory.glob("*_summary.md")):
            source_stem = report_path.stem[: -len("_summary")]
            source_path = input_files.get(source_stem)
            if source_path is None:
                continue
            report = parse_legacy_report(report_path)
            media_data = source_path.read_bytes()
            record = repository.create(
                title=report.title,
                source_name=UPLOAD_PREFIX_PATTERN.sub("", source_path.name),
                source_path=source_path,
                media_data=media_data,
                media_type=source_path.suffix.lower(),
            )
            duration = report.segments[-1].start if report.segments else None
            repository.complete(
                record.id,
                prompt_name=report.prompt_name,
                language=report.language,
                duration=duration,
                transcript=report.transcript,
                segments=report.segments,
                summary=report.summary,
            )
            imported += 1

    repository.set_metadata(MIGRATION_KEY, "complete")
    return imported


def migrate_legacy_storage(
    repository: LectureRepository, data_directory: Path
) -> MigrationResult:
    """Embed legacy media while keeping PDFs on disk for Range streaming."""
    data_directory = data_directory.expanduser().resolve()
    input_directory = data_directory / "inputs"
    output_directory = data_directory / "outputs"
    vector_directory = data_directory / "vector_db"
    imported = import_legacy_reports(repository, input_directory, output_directory)
    embedded = 0
    warnings: List[str] = []
    referenced_paths: set[Path] = set()

    for lecture in repository.list():
        if not lecture.source_path:
            continue
        source = Path(lecture.source_path).expanduser().resolve()
        referenced_paths.add(source)
        if lecture.media_type.lower() == ".pdf":
            continue
        if lecture.has_media:
            repository.clear_source_path(lecture.id)
            continue
        if not source.is_file():
            warnings.append(f"원본을 찾지 못해 폴더를 보존했습니다: {lecture.source_name}")
            continue
        repository.set_media(lecture.id, source.read_bytes(), clear_source_path=True)
        embedded += 1

    input_files = {
        path.resolve()
        for path in input_directory.iterdir()
        if path.is_file() and path.name != ".gitkeep"
    } if input_directory.is_dir() else set()
    unknown_inputs = input_files - referenced_paths
    if unknown_inputs:
        warnings.append("DB에 연결되지 않은 입력 파일이 있어 inputs 폴더를 보존했습니다.")

    input_stems = {path.stem for path in input_files}
    expected_reports = (
        {
            path
            for path in output_directory.glob("*_summary.md")
            if path.stem[: -len("_summary")] in input_stems
        }
        if output_directory.is_dir()
        else set()
    )
    unknown_outputs = {
        path for path in output_directory.iterdir()
        if path.is_file() and path.name != ".gitkeep" and path not in expected_reports
    } if output_directory.is_dir() else set()
    if unknown_outputs:
        warnings.append("알 수 없는 결과 파일이 있어 outputs 폴더를 보존했습니다.")

    removed = 0
    if input_directory.is_dir() and not unknown_inputs and not warnings:
        shutil.rmtree(input_directory)
        removed += 1
    if output_directory.is_dir() and not unknown_outputs and not warnings:
        shutil.rmtree(output_directory)
        removed += 1
    if vector_directory.is_dir():
        shutil.rmtree(vector_directory)
        removed += 1

    for auxiliary_name in (
        "lass.db-shm",
        "lass.db-wal",
        "vectors.db-shm",
        "vectors.db-wal",
    ):
        auxiliary_path = data_directory / auxiliary_name
        if auxiliary_path.is_file():
            auxiliary_path.unlink()

    return MigrationResult(
        imported_reports=imported,
        embedded_media=embedded,
        removed_directories=removed,
        warnings=tuple(warnings),
    )
