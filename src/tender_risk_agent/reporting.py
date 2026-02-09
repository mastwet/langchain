from __future__ import annotations

from datetime import datetime

from .models import AnalysisContext, CrossSupplierFinding, FinalReport, SupplierRiskProfile


def build_final_report(
    context: AnalysisContext,
    profiles: list[SupplierRiskProfile],
    cross_findings: list[CrossSupplierFinding],
) -> FinalReport:
    high_risk = [p.supplier for p in profiles if p.risk_level.value == "高风险"]

    def gather_by_role(role_kw: str) -> list[str]:
        rows: list[str] = []
        for profile in profiles:
            for ev in profile.evidence:
                if role_kw in ev.expert_role:
                    rows.extend(ev.suspicious_points)
        return rows[:10]

    return FinalReport(
        context=context,
        high_risk_suppliers=high_risk,
        supplier_profiles=profiles,
        legal_basis=[
            "《招标投标法实施条例》第三十九条、第四十条",
            "《政府采购法》第七十七条",
            "《政府采购货物和服务招标投标管理办法》第三十七条",
        ],
        financial_evidence=gather_by_role("财务"),
        technical_evidence=gather_by_role("技术"),
        behavioral_evidence=gather_by_role("行为"),
        legal_evidence=gather_by_role("法律"),
        suggestions=[
            "对高风险供应商发起澄清程序并核验原始材料",
            "必要时移交监管与纪检部门进行进一步调查",
        ],
        cross_supplier_findings=cross_findings,
    )


def render_markdown(report: FinalReport) -> str:
    ctx = report.context
    lines = [
        "## 围串标风险分析报告（自动生成）",
        "",
        "### 一、基本情况",
        f"- 项目编号：{ctx.project_code}",
        f"- 采购人：{ctx.buyer}",
        f"- 分析时间：{ctx.analyzed_at.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "### 二、高风险迹象汇总",
        "| 供应商 | 风险类型 | 具体表现 | 关联供应商 |",
        "|--------|----------|----------|------------|",
    ]

    for profile in report.supplier_profiles:
        detail = "；".join([p for ev in profile.evidence for p in ev.suspicious_points][:2]) or "无"
        related_set: set[str] = set()
        for finding in report.cross_supplier_findings:
            if profile.supplier in finding.suppliers:
                related_set.update([s for s in finding.suppliers if s != profile.supplier])
        related = "、".join(sorted(related_set)) or "无"
        lines.append(f"| {profile.supplier} | {profile.risk_level.value} | {detail} | {related} |")

    lines.extend(["", "### 三、跨供应商关联发现"])
    if report.cross_supplier_findings:
        for f in report.cross_supplier_findings:
            lines.append(f"- [{f.finding_type}] {', '.join(f.suppliers)}：{f.detail}")
    else:
        lines.append("- 未发现明显跨供应商关联。")

    lines.extend(
        [
            "",
            "### 四、法律依据",
            *[f"- {law}" for law in report.legal_basis],
            "",
            "### 五、处理建议",
            *[f"- {s}" for s in report.suggestions],
        ]
    )
    return "\n".join(lines)


def build_context(project_name: str, supplier_count: int) -> AnalysisContext:
    return AnalysisContext(
        project_name=project_name,
        project_code="ZB2024-XXX",
        buyer="XX单位",
        analyzed_at=datetime.now(),
        supplier_count=supplier_count,
    )
