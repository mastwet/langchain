import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tender_risk_agent.document_assembly import DocumentAssembler


class DocumentAssemblyTests(unittest.TestCase):
    def test_record_and_save_writes_outputs(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "docs"
            base.mkdir()
            source = base / "supplier" / "doc.md"
            source.parent.mkdir(parents=True)
            source.write_text("original", encoding="utf-8")

            output_dir = root / "assembled"
            assembler = DocumentAssembler(output_dir, base_dir=base)

            chunks = ["a", "b"]
            record = assembler.record_and_save(source, "original", chunks)

            target_dir = output_dir / "supplier" / "doc"
            self.assertTrue((target_dir / "original.md").exists())
            self.assertTrue((target_dir / "chunks.json").exists())
            self.assertTrue((target_dir / "merged.md").exists())
            self.assertEqual((target_dir / "original.md").read_text(encoding="utf-8"), "original")
            self.assertEqual(json.loads((target_dir / "chunks.json").read_text(encoding="utf-8")), chunks)
            self.assertEqual((target_dir / "merged.md").read_text(encoding="utf-8"), "a\n\n---\n\nb")
            self.assertEqual(record.source, source)


if __name__ == "__main__":
    unittest.main()
