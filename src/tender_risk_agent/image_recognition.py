from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def load_image_prompt(prompt_path: Path | None = None) -> str:
    if prompt_path is None:
        prompt_path = Path(__file__).resolve().parent / "prompts" / "image_analysis_prompt.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


@dataclass(slots=True)
class ImageRecognitionResult:
    image_path: str
    note: str


class ImageRecognizer:
    def __init__(
        self,
        llm: Any | None = None,
        prompt_text: str | None = None,
        model: str = "gemini-2.0-flash-exp",
    ) -> None:
        self.llm = llm or ChatGoogleGenerativeAI(model=model)
        self.prompt_text = prompt_text or load_image_prompt()

    @staticmethod
    def extract_image_refs(content: str) -> list[str]:
        return IMAGE_PATTERN.findall(content)

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type:
            mime_type = "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def analyze_images(self, image_refs: Iterable[str]) -> list[ImageRecognitionResult]:
        results: list[ImageRecognitionResult] = []
        for image_path in image_refs:
            path = Path(image_path)
            data_url = self._image_to_data_url(path)
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": self.prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            )
            response = self.llm.invoke([msg])
            note = str(response.content)
            results.append(ImageRecognitionResult(image_path=image_path, note=note))
        return results

    def analyze_markdown(self, content: str) -> list[ImageRecognitionResult]:
        return self.analyze_images(self.extract_image_refs(content))
