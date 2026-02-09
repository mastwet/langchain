from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ChunkType(str, Enum):
    PROCUREMENT_REQUIREMENT = "采购需求"
    COMMERCIAL_TERMS = "商务条款"
    QUALIFICATION = "资质证明"
    TECHNICAL_SOLUTION = "技术方案"
    LEGAL_DECLARATION = "法律声明"
    IMAGE_OR_TABLE = "图片/表格"
    OTHER = "其他"


class RiskLevel(str, Enum):
    HIGH = "高风险"
    MEDIUM = "中风险"
    LOW = "低风险"


@dataclass(slots=True)
class DocumentChunk:
    supplier: str
    source_file: str
    title: str
    content: str
    chunk_type: ChunkType
    has_image: bool = False


@dataclass(slots=True)
class EvidenceItem:
    supplier: str
    expert_role: str
    chunk_title: str
    risk_score: float
    suspicious_points: list[str] = field(default_factory=list)
    key_data: dict[str, Any] = field(default_factory=dict)
    legal_risks: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CrossSupplierFinding:
    finding_type: str
    suppliers: list[str]
    detail: str
    score_impact: float


@dataclass(slots=True)
class SupplierRiskProfile:
    supplier: str
    evidence: list[EvidenceItem] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        if not self.evidence:
            return 0.0
        return round(sum(item.risk_score for item in self.evidence) / len(self.evidence), 2)

    @property
    def risk_level(self) -> RiskLevel:
        score = self.avg_score
        if score >= 7.0:
            return RiskLevel.HIGH
        if score >= 4.0:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


@dataclass(slots=True)
class AnalysisContext:
    project_name: str
    project_code: str
    buyer: str
    analyzed_at: datetime
    supplier_count: int


@dataclass(slots=True)
class FinalReport:
    context: AnalysisContext
    high_risk_suppliers: list[str]
    supplier_profiles: list[SupplierRiskProfile]
    legal_basis: list[str]
    financial_evidence: list[str]
    technical_evidence: list[str]
    behavioral_evidence: list[str]
    legal_evidence: list[str]
    suggestions: list[str]
    cross_supplier_findings: list[CrossSupplierFinding] = field(default_factory=list)
