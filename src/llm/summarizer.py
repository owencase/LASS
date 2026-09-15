"""Prompt-aware summary generation."""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable, Iterator, TextIO

from .model_loader import LocalEXAONEClient
from .prompt_manager import PromptTemplate


class Summarizer:
    def __init__(self, client: LocalEXAONEClient) -> None:
        self.client = client

    def summarize(self, context: str, template: PromptTemplate) -> str:
        if not context.strip():
            raise ValueError("Summary context must not be empty")
        return self.client.chat(template.system_prompt, template.render(context))

    def summarize_pages(
        self,
        pages: Iterable[str],
        template: PromptTemplate,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> str:
        """Summarize one context page at a time, then synthesize disk-spooled notes.

        This mirrors a machine-learning dataloader: only one source window and its
        inference tensors are active at once. Intermediate notes are written to a
        temporary file instead of accumulating as a Python list.
        """
        condensed_context = self.condense_pages(
            pages,
            objective=template.description,
            progress=progress,
        )
        return self.summarize(condensed_context, template)

    def condense_pages(
        self,
        pages: Iterable[str],
        *,
        objective: str,
        progress: Callable[[str], None] | None = None,
    ) -> str:
        """Read pages serially and hierarchically reduce disk-spooled evidence."""
        cleaned = (page.strip() for page in pages if page and page.strip())
        try:
            first = next(cleaned)
        except StopIteration as exc:
            raise ValueError("Summary context must not be empty") from exc
        try:
            second = next(cleaned)
        except StopIteration:
            return first

        def source_pages() -> Iterator[str]:
            yield first
            yield second
            yield from cleaned

        with TemporaryDirectory(prefix="lass-summary-") as temporary:
            current_path = Path(temporary) / "notes-0.jsonl"
            page_count = 0
            with current_path.open("w", encoding="utf-8") as notes_file:
                for page_count, page in enumerate(source_pages(), start=1):
                    if progress is not None:
                        progress(f"자료 조각 {page_count}번만 메모리에 불러와 읽고 있습니다.")
                    note = self.client.chat(
                        (
                            "당신은 긴 자료를 메모리 절약 방식으로 읽는 분석가입니다. "
                            "현재 조각에 실제로 있는 핵심 사실, 결론, 용어와 수치만 간결한 "
                            "한국어 근거 메모로 남기세요. 다른 조각의 내용을 추측하지 마세요."
                        ),
                        f"최종 작업 목적: {objective}\n\n[현재 자료 조각]\n{page}",
                        max_new_tokens=min(128, self.client.max_new_tokens),
                    )
                    self._write_note(notes_file, note)

            pass_number = 0
            note_count = page_count
            while note_count > 4:
                pass_number += 1
                next_path = Path(temporary) / f"notes-{pass_number}.jsonl"
                reduced_count = 0
                notes = self._read_notes(current_path)
                with next_path.open("w", encoding="utf-8") as output:
                    while group := list(islice(notes, 4)):
                        reduced_count += 1
                        if progress is not None:
                            progress(
                                f"디스크 메모를 4개씩 압축하고 있습니다 · {pass_number}단계 {reduced_count}묶음"
                            )
                        compact = self.client.chat(
                            (
                                "당신은 근거 메모를 손실 적게 병합하는 편집자입니다. "
                                "중복만 제거하고 사실, 수치, 결론, 서로 다른 관점은 보존하세요."
                            ),
                            f"최종 작업 목적: {objective}\n\n" + "\n\n".join(group),
                            max_new_tokens=min(160, self.client.max_new_tokens),
                        )
                        self._write_note(output, compact)
                current_path = next_path
                note_count = reduced_count

            if progress is not None:
                progress(f"압축된 핵심 메모 {note_count}개로 최종 요약을 작성합니다.")
            # This final join is deliberately bounded: the reduction loop leaves
            # at most four short notes, regardless of the original document size.
            return "\n\n".join(self._read_notes(current_path))

    @staticmethod
    def _write_note(file: TextIO, note: str) -> None:
        file.write(json.dumps(note.strip(), ensure_ascii=False) + "\n")

    @staticmethod
    def _read_notes(path: Path) -> Iterator[str]:
        with path.open("r", encoding="utf-8") as notes_file:
            for line in notes_file:
                if line.strip():
                    note = json.loads(line)
                    if isinstance(note, str) and note.strip():
                        yield note.strip()
