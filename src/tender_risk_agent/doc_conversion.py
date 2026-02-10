from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Iterable, Sequence


SUPPORTED_EXTENSIONS = {".doc", ".docx", ".rtf", ".odt"}


@dataclass(frozen=True)
class ConversionResult:
    source: Path
    output: Path


def find_doc_files(input_dir: Path, recursive: bool = True) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    pattern = "**/*" if recursive else "*"
    return [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _ensure_pandoc(pandoc_path: str) -> str:
    resolved = shutil.which(pandoc_path) if pandoc_path else None
    if not resolved:
        raise FileNotFoundError(
            "pandoc not found on PATH. Please install pandoc and try again."
        )
    return resolved


def convert_doc_to_markdown(
    source: Path,
    output: Path,
    *,
    pandoc_path: str = "pandoc",
    extra_args: Sequence[str] | None = None,
) -> ConversionResult:
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported source extension: {source.suffix}")

    output.parent.mkdir(parents=True, exist_ok=True)
    pandoc_exe = _ensure_pandoc(pandoc_path)

    args = [
        pandoc_exe,
        str(source),
        "--to",
        "gfm",
        "--output",
        str(output),
    ]
    if extra_args:
        args.extend(extra_args)

    subprocess.run(args, check=True)
    return ConversionResult(source=source, output=output)


def convert_docs_in_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    recursive: bool = True,
    keep_structure: bool = True,
    pandoc_path: str = "pandoc",
    extra_args: Sequence[str] | None = None,
) -> list[ConversionResult]:
    sources = find_doc_files(input_dir, recursive=recursive)
    results: list[ConversionResult] = []

    for source in sources:
        relative = source.relative_to(input_dir) if keep_structure else Path(source.name)
        output = output_dir / relative.with_suffix(".md")
        result = convert_doc_to_markdown(
            source,
            output,
            pandoc_path=pandoc_path,
            extra_args=extra_args,
        )
        results.append(result)

    return results
