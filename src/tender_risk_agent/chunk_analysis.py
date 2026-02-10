from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re

from .image_recognition import IMAGE_PATTERN, ImageRecognizer
from .models import DocumentChunk


def load_chunk_prompt(prompt_path: Path | None = None) -> str:
    if prompt_path is None:
        prompt_path = Path(__file__).resolve().parent / "prompts" / "chunk_analysis_prompt.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


@dataclass(slots=True)
class ChunkAnalysisResult:
    raw: str
    image_notes: list[str]


class ChunkAnalyzer:
    def __init__(
        self,
        llm: Any,
        prompt_text: str | None = None,
        image_recognizer: ImageRecognizer | None = None,
    ) -> None:
        self.llm = llm
        self.prompt_text = prompt_text or load_chunk_prompt()
        self.image_recognizer = image_recognizer

    def analyze(self, chunk: DocumentChunk) -> ChunkAnalysisResult:
        image_notes: list[str] = []
        content = chunk.content
        if chunk.has_image and self.image_recognizer:
            results = self.image_recognizer.analyze_markdown(chunk.content)
            image_notes.extend([result.note for result in results])
            content = self._inject_image_notes(chunk.content, image_notes)

        prompt = self.prompt_text.format(
            filename=chunk.source_file,
            chunk_type=chunk.chunk_type.value,
            content=content,
        )
        raw = str(self.llm.invoke(prompt).content)
        return ChunkAnalysisResult(raw=raw, image_notes=image_notes)

    @staticmethod
    def _inject_image_notes(content: str, notes: list[str]) -> str:
        if not notes:
            return content
        note_iter = iter(notes)

        def _repl(match: re.Match) -> str:
            try:
                note = next(note_iter)
            except StopIteration:
                return match.group(0)
            payload = json.dumps({"image_analysis": note}, ensure_ascii=False)
            return f"{match.group(0)}\n\n<!-- {payload} -->"

        return IMAGE_PATTERN.sub(_repl, content)
