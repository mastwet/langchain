from __future__ import annotations

import argparse
from pathlib import Path

from langchain_openai import ChatOpenAI

from .doc_conversion import convert_docs_in_dir
from .pipeline import TenderRiskPipeline
from .reporting import build_context, build_final_report, render_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tender document collusion risk recognition system"
    )
    parser.add_argument(
        "--convert-docs-from",
        default=None,
        help="Directory with .doc/.docx/.rtf/.odt to convert before processing",
    )
    parser.add_argument(
        "--convert-docs-to",
        default=None,
        help="Output directory for converted markdown (defaults to --docs-dir)",
    )
    parser.add_argument("--docs-dir", default="docs", help="Markdown docs directory")
    parser.add_argument("--project-name", default="Unnamed project", help="Project name")
    parser.add_argument("--output-dir", default="outputs", help="Report output directory")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.convert_docs_from:
        output_dir = Path(args.convert_docs_to or args.docs_dir)
        convert_docs_in_dir(Path(args.convert_docs_from), output_dir)

    llm = ChatOpenAI(model=args.model, temperature=0)
    pipeline = TenderRiskPipeline(llm=llm, vision_tool=None)

    profiles, cross_findings = pipeline.build_profiles(args.docs_dir)
    context = build_context(project_name=args.project_name, supplier_count=len(profiles))
    final_report = build_final_report(context, profiles, cross_findings)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_text = render_markdown(final_report)
    json_text = pipeline.profiles_to_json(profiles, cross_findings)

    (output_dir / "risk_report.md").write_text(markdown_text, encoding="utf-8")
    (output_dir / "risk_report.json").write_text(json_text, encoding="utf-8")

    print(f"Done. Reports saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
