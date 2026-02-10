from __future__ import annotations

import re
from dataclasses import dataclass


HEADING_SPLIT = re.compile(r"(?=^#{1,4}\s)", re.MULTILINE)


@dataclass(frozen=True)
class TextChunker:
    chunk_size: int = 1000
    chunk_overlap: int = 200

    def split(self, content: str) -> list[str]:
        sections = [s.strip() for s in HEADING_SPLIT.split(content) if s.strip()]
        chunks: list[str] = []
        for sec in sections:
            if len(sec) <= self.chunk_size:
                chunks.append(sec)
                continue
            start = 0
            while start < len(sec):
                end = min(start + self.chunk_size, len(sec))
                chunks.append(sec[start:end])
                if end == len(sec):
                    break
                start = max(0, end - self.chunk_overlap)
        return chunks or [content]
