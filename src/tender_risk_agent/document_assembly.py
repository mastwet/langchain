from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class AssemblyRecord:
    source: Path
    original: str
    chunks: list[str]
    merged: str
    output_dir: Path


class DocumentAssembler:
    def __init__(
        self,
        output_dir: Path,
        *,
        base_dir: Path | None = None,
        chunk_separator: str = "\n\n---\n\n",
    ) -> None:
        self.output_dir = output_dir
        self.base_dir = base_dir
        self.chunk_separator = chunk_separator

    def merge_chunks(self, chunks: Iterable[str]) -> str:
        return self.chunk_separator.join(chunks)

    def record(self, source: Path, original: str, chunks: list[str]) -> AssemblyRecord:
        merged = self.merge_chunks(chunks)
        return AssemblyRecord(
            source=source,
            original=original,
            chunks=chunks,
            merged=merged,
            output_dir=self.output_dir,
        )

    def record_and_save(self, source: Path, original: str, chunks: list[str]) -> AssemblyRecord:
        record = self.record(source, original, chunks)
        self._write_record(record)
        return record

    def _write_record(self, record: AssemblyRecord) -> None:
        relative = (
            record.source.relative_to(self.base_dir)
            if self.base_dir
            else Path(record.source.name)
        )
        target_dir = self.output_dir / relative.parent / record.source.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        (target_dir / "original.md").write_text(record.original, encoding="utf-8")
        (target_dir / "chunks.json").write_text(
            json.dumps(record.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (target_dir / "merged.md").write_text(record.merged, encoding="utf-8")
