import unittest

from tender_risk_agent.text_chunking import TextChunker


class TextChunkingTests(unittest.TestCase):
    def test_split_preserves_small_sections(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        content = "# Title\nshort paragraph"
        chunks = chunker.split(content)
        self.assertEqual(chunks, [content])

    def test_split_large_section_with_overlap(self):
        chunker = TextChunker(chunk_size=10, chunk_overlap=3)
        content = "# T\n" + ("A" * 25)
        chunks = chunker.split(content)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0].endswith("A" * 10))
        self.assertTrue(chunks[1].startswith("A" * 3))

    def test_split_without_headings(self):
        chunker = TextChunker(chunk_size=8, chunk_overlap=2)
        content = "B" * 20
        chunks = chunker.split(content)
        self.assertGreater(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
