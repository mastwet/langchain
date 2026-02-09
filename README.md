# 基于 LangChain 的采购文件围串标行为智能识别系统（设计稿）

本仓库提供一个可落地的 **AI Agent 智能体系统设计与代码骨架**，用于对采购投标文件进行围标/串标风险识别。

## 目标
- 输入：`docs/` 目录下各供应商 Markdown 化投标文件（含 `.img/`）与询价/招标文件。
- 输出：结构化风险报告（Markdown + JSON）。
- 核心：分块解析、多模态识别、单文件证据提取、多专家 Agent 协同、总审计官汇总与冲突仲裁。

## 代码结构
- `src/tender_risk_agent/models.py`：领域数据模型。
- `src/tender_risk_agent/pipeline.py`：LangChain 主流程编排（含跨供应商相似度与关联关系分析）。
- `src/tender_risk_agent/prompts.py`：审计提示词模板。
- `src/tender_risk_agent/reporting.py`：Markdown/JSON 报告生成。
- `src/tender_risk_agent/main.py`：CLI 入口。
- `tests/test_pipeline.py`：分块分类、冲突仲裁、跨供应商规则检测测试。

## 关键特性映射
1. **标题结构智能分块**：按 `#` / `##` / `###` / `####` 递归切分并保留层级。
2. **分块分类**：采购需求、商务条款、资质证明、技术方案、法律声明、图片/表格。
3. **多模态图片解析**：检测 `![](...)` 后触发视觉工具。
4. **单文件证据提取**：提取关键数据、异常特征、法律风险点、关联线索。
5. **多专家协同**：财务、法律、技术、行为四专家 + `ChiefAuditorAgent`。
6. **争议解决机制**：专家评分分歧过大时进行二次研判。
7. **国内招投标专项规则**：
   - 保证金同账户/联系方式交叉检查；
   - 法定必备声明函缺失检查；
   - 跨供应商技术方案/声明文本相似度（Simhash）检测。

## 运行示例
```bash
PYTHONPATH=src python -m tender_risk_agent.main --docs-dir ./docs --project-name "XX采购项目"
```

## 测试
```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

> 说明：本仓库聚焦系统设计与工程化骨架，具体 LLM 与多模态模型可按部署环境替换。
