import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tender_risk_agent.image_recognition import ImageRecognizer, load_image_prompt


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return type("Resp", (), {"content": "mock note"})()


class ImageRecognitionTests(unittest.TestCase):
    def test_load_image_prompt_has_content(self):
        prompt = load_image_prompt()
        self.assertTrue(prompt)

    def test_analyze_markdown_extracts_images(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            img_a = root / "a.png"
            img_b = root / "b.jpg"
            img_a.write_bytes(b"fake")
            img_b.write_bytes(b"fake")

            llm = _FakeLLM()
            recognizer = ImageRecognizer(llm, prompt_text="analyze this")
            content = f"Text ![]({img_a}) middle ![x]({img_b}) end"
            results = recognizer.analyze_markdown(content)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].image_path, str(img_a))
            self.assertEqual(results[1].image_path, str(img_b))
            self.assertEqual(len(llm.calls), 2)


if __name__ == "__main__":
    unittest.main()
