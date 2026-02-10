import unittest

from tender_risk_agent.chunk_analysis import ChunkAnalyzer, ChunkAnalysisResult
from tender_risk_agent.models import ChunkType, DocumentChunk


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return _Resp("analysis result")


class _FakeImageRecognizer:
    def __init__(self):
        self.calls = []

    def analyze_markdown(self, content):
        self.calls.append(content)
        return [type("R", (), {"note": "image note"})()]


class ChunkAnalysisTests(unittest.TestCase):
    def test_analyze_includes_llm_output(self):
        llm = _FakeLLM()
        analyzer = ChunkAnalyzer(llm, prompt_text="file={filename} type={chunk_type} {content}")
        chunk = DocumentChunk(
            supplier="A",
            source_file="a.md",
            title="t",
            content="hello",
            chunk_type=ChunkType.OTHER,
            has_image=False,
        )
        result = analyzer.analyze(chunk)
        self.assertIsInstance(result, ChunkAnalysisResult)
        self.assertEqual(result.raw, "analysis result")
        self.assertEqual(result.image_notes, [])
        self.assertTrue(llm.calls)

    def test_analyze_uses_image_recognizer(self):
        llm = _FakeLLM()
        images = _FakeImageRecognizer()
        analyzer = ChunkAnalyzer(llm, prompt_text="{content}", image_recognizer=images)
        chunk = DocumentChunk(
            supplier="A",
            source_file="a.md",
            title="t",
            content="![x](img.png)",
            chunk_type=ChunkType.IMAGE_OR_TABLE,
            has_image=True,
        )
        result = analyzer.analyze(chunk)
        self.assertEqual(result.image_notes, ["image note"])
        self.assertEqual(len(images.calls), 1)
        self.assertIn('<!-- {"image_analysis": "image note"} -->', llm.calls[0])


if __name__ == "__main__":
    unittest.main()
