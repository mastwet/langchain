import unittest

from tender_risk_agent.models import ChunkType, EvidenceItem, SupplierRiskProfile
from tender_risk_agent.pipeline import TenderRiskPipeline


class _Resp:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def invoke(self, _msg):
        return _Resp("风险评分 6分；疑点：模板高度一致")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = TenderRiskPipeline(llm=FakeLLM())

    def test_classify_chunk(self):
        self.assertEqual(self.pipeline._classify_chunk("技术参数与实施方案"), ChunkType.TECHNICAL_SOLUTION)
        self.assertEqual(self.pipeline._classify_chunk("无重大违法记录声明"), ChunkType.LEGAL_DECLARATION)

    def test_conflict_resolution(self):
        items = [
            EvidenceItem(supplier="A", expert_role="x", chunk_title="c", risk_score=1),
            EvidenceItem(supplier="A", expert_role="y", chunk_title="c", risk_score=8),
        ]
        self.pipeline._resolve_conflict(items)
        self.assertTrue(any("二次研判" in s for s in items[0].suggestions))

    def test_apply_cross_findings(self):
        profiles = [SupplierRiskProfile("A"), SupplierRiskProfile("B")]
        findings = self.pipeline._relation_graph_findings(
            [
                type("Chunk", (), {"supplier": "A", "content": "电话 13800138000"})(),
                type("Chunk", (), {"supplier": "B", "content": "联系人 13800138000"})(),
            ]
        )
        self.assertTrue(findings)
        self.pipeline._apply_cross_findings_to_profiles(profiles, findings)
        self.assertGreater(len(profiles[0].evidence), 0)


if __name__ == "__main__":
    unittest.main()
