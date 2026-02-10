import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tender_risk_agent.doc_conversion import (
    SUPPORTED_EXTENSIONS,
    convert_doc_to_markdown,
    convert_docs_in_dir,
    find_doc_files,
)


class DocConversionTests(unittest.TestCase):
    def test_find_doc_files_filters_extensions(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.docx").write_text("x", encoding="utf-8")
            (root / "b.txt").write_text("x", encoding="utf-8")
            (root / "c.rtf").write_text("x", encoding="utf-8")
            found = find_doc_files(root)
            suffixes = {p.suffix.lower() for p in found}
            self.assertTrue(suffixes.issubset(SUPPORTED_EXTENSIONS))
            self.assertIn(".docx", suffixes)
            self.assertIn(".rtf", suffixes)

    def test_convert_doc_to_markdown_invokes_pandoc(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "input.docx"
            source.write_text("x", encoding="utf-8")
            output = root / "out" / "input.md"

            with patch("tender_risk_agent.doc_conversion.shutil.which", return_value="pandoc"), patch(
                "tender_risk_agent.doc_conversion.subprocess.run"
            ) as runner:
                result = convert_doc_to_markdown(source, output)
                runner.assert_called_once()
                self.assertEqual(result.source, source)
                self.assertEqual(result.output, output)

    def test_convert_docs_in_dir_keeps_structure(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "supplier" / "docs"
            nested.mkdir(parents=True)
            (nested / "a.doc").write_text("x", encoding="utf-8")
            output = root / "out"

            with patch("tender_risk_agent.doc_conversion.shutil.which", return_value="pandoc"), patch(
                "tender_risk_agent.doc_conversion.subprocess.run"
            ):
                results = convert_docs_in_dir(root, output, keep_structure=True)

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].output.as_posix().endswith("supplier/docs/a.md"))


if __name__ == "__main__":
    unittest.main()
