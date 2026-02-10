from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .chunk_analysis import ChunkAnalyzer
from .document_assembly import DocumentAssembler
from .image_recognition import IMAGE_PATTERN, ImageRecognizer
from .models import ChunkType, CrossSupplierFinding, DocumentChunk, EvidenceItem, SupplierRiskProfile
from .prompts import AGENT_PROMPT, EXPERTS
from .text_chunking import TextChunker

PHONE_PATTERN = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ACCOUNT_PATTERN = re.compile(r"\b\d{12,19}\b")

MANDATORY_DECLARATIONS = ["中小企业声明函", "无重大违法记录声明", "授权委托书", "无围标串标声明"]


class TenderRiskPipeline:
    """采购文件围串标识别主流程。"""

    def __init__(
        self,
        llm: Any,
        vision_tool: Any | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        image_recognizer: ImageRecognizer | None = None,
        chunk_analyzer: ChunkAnalyzer | None = None,
        document_assembler: DocumentAssembler | None = None,
        supervisor_graph: Any | None = None,
    ) -> None:
        self.llm = llm
        self.vision_tool = vision_tool
        if image_recognizer:
            self.image_recognizer = image_recognizer
        elif vision_tool:
            self.image_recognizer = ImageRecognizer(vision_tool)
        else:
            self.image_recognizer = None
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_analyzer = chunk_analyzer or ChunkAnalyzer(
            llm=self.llm, image_recognizer=self.image_recognizer
        )
        self.document_assembler = document_assembler
        self.supervisor_graph = supervisor_graph

    def load_markdown_docs(self, docs_dir: str | Path) -> dict[str, list[Path]]:
        docs_path = Path(docs_dir)
        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in docs_path.glob("**/*.md"):
            if "询价文件" in str(path):
                continue
            grouped[path.parent.name].append(path)
        return grouped

    def split_and_classify(self, supplier: str, file_path: Path) -> list[DocumentChunk]:
        content = file_path.read_text(encoding="utf-8")
        chunks = self.chunker.split(content)
        if self.document_assembler:
            self.document_assembler.record_and_save(file_path, content, chunks)
        return [
            DocumentChunk(
                supplier=supplier,
                source_file=str(file_path),
                title=f"{file_path.stem}-chunk-{idx}",
                content=chunk,
                chunk_type=self._classify_chunk(chunk),
                has_image=bool(IMAGE_PATTERN.search(chunk)),
            )
            for idx, chunk in enumerate(chunks, start=1)
        ]

    def analyze_chunk(self, chunk: DocumentChunk) -> dict[str, Any]:
        result = self.chunk_analyzer.analyze(chunk)
        return {"raw": result.raw, "image_notes": result.image_notes}

    def run_supervisor_team(self, supplier: str, evidence_text: str) -> str | None:
        if not self.supervisor_graph:
            return None
        state = {
            "messages": [f"Supplier: {supplier}\nEvidence:\n{evidence_text}"],
            "next": "",
        }
        result = self.supervisor_graph.invoke(state)
        messages = result.get("messages", [])
        if not messages:
            return None
        last = messages[-1]
        if hasattr(last, "content"):
            return str(last.content)
        return str(last)


    def run_expert_panel(self, supplier: str, evidence_text: str, source_refs: list[str]) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for expert in EXPERTS:
            prompt = AGENT_PROMPT.format(
                company=supplier,
                expert_role=expert["name"],
                skills="、".join(expert["skills"]),
                focus=expert["focus"],
                evidence=evidence_text,
            )
            response = str(self.llm.invoke(prompt).content)
            score = self._extract_score(response)
            items.append(
                EvidenceItem(
                    supplier=supplier,
                    expert_role=expert["name"],
                    chunk_title="联合评审",
                    risk_score=score,
                    suspicious_points=[response],
                    source_refs=source_refs,
                    suggestions=["建议人工复核该专家意见"],
                )
            )

        self._resolve_conflict(items)
        return items

    def build_profiles(self, docs_dir: str | Path) -> tuple[list[SupplierRiskProfile], list[CrossSupplierFinding]]:
        grouped = self.load_markdown_docs(docs_dir)
        profiles: list[SupplierRiskProfile] = []
        all_chunks: list[DocumentChunk] = []

        for supplier, files in grouped.items():
            profile = SupplierRiskProfile(supplier=supplier)
            evidence_pool: list[str] = []
            source_refs: list[str] = []

            for file in files:
                chunks = self.split_and_classify(supplier, file)
                all_chunks.extend(chunks)
                source_refs.append(str(file))
                for chunk in chunks:
                    analysis = self.analyze_chunk(chunk)
                    evidence_pool.append(analysis["raw"])

            supervisor_summary = self.run_supervisor_team(supplier, "\n".join(evidence_pool))
            if supervisor_summary:
                evidence_pool.append(supervisor_summary)
            profile.evidence.extend(self.run_expert_panel(supplier, "\n".join(evidence_pool), source_refs))
            self._append_mandatory_declaration_checks(profile, all_text="\n".join(evidence_pool), source_refs=source_refs)
            profiles.append(profile)

        cross_findings = self._cross_supplier_findings(all_chunks)
        self._apply_cross_findings_to_profiles(profiles, cross_findings)
        return profiles, cross_findings

    def _cross_supplier_findings(self, chunks: list[DocumentChunk]) -> list[CrossSupplierFinding]:
        findings: list[CrossSupplierFinding] = []
        findings.extend(self._text_similarity_findings(chunks))
        findings.extend(self._relation_graph_findings(chunks))
        return findings

    def _text_similarity_findings(self, chunks: list[DocumentChunk]) -> list[CrossSupplierFinding]:
        scoped = [c for c in chunks if c.chunk_type in {ChunkType.TECHNICAL_SOLUTION, ChunkType.LEGAL_DECLARATION}]
        results: list[CrossSupplierFinding] = []

        for i in range(len(scoped)):
            for j in range(i + 1, len(scoped)):
                a, b = scoped[i], scoped[j]
                if a.supplier == b.supplier:
                    continue
                sim = SequenceMatcher(None, a.content, b.content).ratio()
                if sim >= 0.8:
                    results.append(
                        CrossSupplierFinding(
                            finding_type="技术/声明文本高相似",
                            suppliers=sorted({a.supplier, b.supplier}),
                            detail=f"{a.title} 与 {b.title} 相似度 {sim:.2f}",
                            score_impact=1.5,
                        )
                    )
        return results

    def _relation_graph_findings(self, chunks: list[DocumentChunk]) -> list[CrossSupplierFinding]:
        supplier_tokens: dict[str, set[str]] = defaultdict(set)

        for chunk in chunks:
            text = chunk.content
            tokens = set(PHONE_PATTERN.findall(text)) | set(EMAIL_PATTERN.findall(text)) | set(ACCOUNT_PATTERN.findall(text))
            for token in tokens:
                supplier_tokens[chunk.supplier].add(token)

        results: list[CrossSupplierFinding] = []
        suppliers = list(supplier_tokens.keys())
        for i in range(len(suppliers)):
            for j in range(i + 1, len(suppliers)):
                s1, s2 = suppliers[i], suppliers[j]
                shared = supplier_tokens[s1] & supplier_tokens[s2]
                if shared:
                    results.append(
                        CrossSupplierFinding(
                            finding_type="联系方式/账户交叉",
                            suppliers=[s1, s2],
                            detail=f"共享标识: {', '.join(sorted(shared)[:5])}",
                            score_impact=2.0,
                        )
                    )
        return results

    @staticmethod
    def _append_mandatory_declaration_checks(profile: SupplierRiskProfile, all_text: str, source_refs: list[str]) -> None:
        for item in MANDATORY_DECLARATIONS:
            if item not in all_text:
                profile.evidence.append(
                    EvidenceItem(
                        supplier=profile.supplier,
                        expert_role="法律合规专家",
                        chunk_title="必备文件检查",
                        risk_score=7.5,
                        suspicious_points=[f"缺少必备文件：{item}"],
                        legal_risks=["可能违反采购文件实质性响应要求"],
                        source_refs=source_refs,
                        suggestions=["要求供应商澄清并补充原件核验"],
                    )
                )

    @staticmethod
    def _apply_cross_findings_to_profiles(profiles: list[SupplierRiskProfile], findings: list[CrossSupplierFinding]) -> None:
        profile_map = {p.supplier: p for p in profiles}
        for finding in findings:
            for supplier in finding.suppliers:
                profile = profile_map.get(supplier)
                if not profile:
                    continue
                profile.evidence.append(
                    EvidenceItem(
                        supplier=supplier,
                        expert_role="行为模式分析专家",
                        chunk_title="跨供应商关联发现",
                        risk_score=min(10.0, 6.0 + finding.score_impact),
                        suspicious_points=[f"{finding.finding_type}: {finding.detail}"],
                        suggestions=["建议核验历史投标记录、IP 与工商关联关系"],
                    )
                )

    @staticmethod
    def _classify_chunk(text: str) -> ChunkType:
        rules = {
            ChunkType.PROCUREMENT_REQUIREMENT: ["技术规格", "服务要求", "交付时间", "采购需求"],
            ChunkType.COMMERCIAL_TERMS: ["报价", "付款", "履约保证金", "商务", "工程量清单"],
            ChunkType.QUALIFICATION: ["营业执照", "业绩", "证书", "资质"],
            ChunkType.TECHNICAL_SOLUTION: ["实施方案", "设备清单", "技术参数"],
            ChunkType.LEGAL_DECLARATION: ["承诺函", "授权书", "无围标串标声明", "中小企业声明函", "无重大违法记录声明"],
        }
        if IMAGE_PATTERN.search(text) or "|" in text:
            return ChunkType.IMAGE_OR_TABLE
        for chunk_type, keywords in rules.items():
            if any(word in text for word in keywords):
                return chunk_type
        return ChunkType.OTHER

    @staticmethod
    def _extract_score(text: str) -> float:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*分", text)
        if match:
            return min(10.0, max(0.0, float(match.group(1))))
        number = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\b", text)
        if number:
            return min(10.0, max(0.0, float(number.group(1))))
        return 5.0

    @staticmethod
    def _resolve_conflict(items: list[EvidenceItem]) -> None:
        if not items:
            return
        scores = [i.risk_score for i in items]
        if max(scores) - min(scores) >= 4:
            for item in items:
                item.suggestions.append("专家意见分歧较大，ChiefAuditorAgent应发起二次研判。")

    @staticmethod
    def profiles_to_json(profiles: list[SupplierRiskProfile], cross_findings: list[CrossSupplierFinding]) -> str:
        return json.dumps(
            {
                "suppliers": [
                    {
                        "supplier": profile.supplier,
                        "avg_score": profile.avg_score,
                        "risk_level": profile.risk_level.value,
                        "evidence": [
                            {
                                "expert_role": ev.expert_role,
                                "chunk_title": ev.chunk_title,
                                "risk_score": ev.risk_score,
                                "suspicious_points": ev.suspicious_points,
                                "legal_risks": ev.legal_risks,
                                "source_refs": ev.source_refs,
                                "suggestions": ev.suggestions,
                            }
                            for ev in profile.evidence
                        ],
                    }
                    for profile in profiles
                ],
                "cross_supplier_findings": [
                    {
                        "finding_type": f.finding_type,
                        "suppliers": f.suppliers,
                        "detail": f.detail,
                        "score_impact": f.score_impact,
                    }
                    for f in cross_findings
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
