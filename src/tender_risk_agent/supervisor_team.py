from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


class TeamState(TypedDict):
    messages: Annotated[list, "Chat history"]
    next: str


class Router(BaseModel):
    next_step: str


def load_supervisor_prompt(prompt_path: Path | None = None) -> str:
    if prompt_path is None:
        prompt_path = Path(__file__).resolve().parent / "prompts" / "supervisor_prompt.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


def load_sub_supervisor_prompt(prompt_path: Path | None = None) -> str:
    if prompt_path is None:
        prompt_path = Path(__file__).resolve().parent / "prompts" / "sub_supervisor_prompt.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


def load_expert_prompt(filename: str) -> str:
    path = Path(__file__).resolve().parent / "prompts" / filename
    return path.read_text(encoding="utf-8").strip()


@dataclass(slots=True)
class ExpertConfig:
    name: str
    system_prompt: str
    tools: list


def build_supervisor_node(
    llm, options: list[str], prompt_text: str | None = None
) -> Callable[[TeamState], dict[str, str]]:
    prompt_text = prompt_text or load_supervisor_prompt()

    def _node(state: TeamState) -> dict[str, str]:
        prompt = prompt_text.format(options=", ".join(options))
        response = llm.with_structured_output(Router).invoke(state["messages"] + [prompt])
        return {"next": response.next_step}

    return _node


def build_default_experts() -> list[ExpertConfig]:
    return [
        ExpertConfig(
            name="LeadReviewer",
            system_prompt=load_expert_prompt("lead_reviewer_prompt.txt"),
            tools=[],
        ),
        ExpertConfig(
            name="FinancialAnalyst",
            system_prompt=load_expert_prompt("financial_analyst_prompt.txt"),
            tools=[],
        ),
        ExpertConfig(
            name="DocumentAuditor",
            system_prompt=load_expert_prompt("document_auditor_prompt.txt"),
            tools=[],
        ),
    ]


def build_supervisor_graph(
    llm,
    experts: list[ExpertConfig],
    *,
    supervisor_prompt: str | None = None,
) -> StateGraph:
    options = [expert.name for expert in experts] + ["FINISH"]
    supervisor_node = build_supervisor_node(llm, options, supervisor_prompt)

    builder = StateGraph(TeamState)
    for expert in experts:
        agent = create_react_agent(llm, tools=expert.tools, state_modifier=expert.system_prompt)

        def _run(state: TeamState, _agent=agent) -> dict[str, list]:
            return {"messages": [_agent.invoke(state)["messages"][-1]]}

        builder.add_node(expert.name, _run)

    builder.add_node("supervisor", supervisor_node)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {expert.name: expert.name for expert in experts} | {"FINISH": END},
    )
    for expert in experts:
        builder.add_edge(expert.name, "supervisor")

    return builder


def build_admin_team_graph(
    llm,
    experts: list[ExpertConfig],
    *,
    supervisor_prompt: str | None = None,
    sub_supervisor_prompt: str | None = None,
) -> StateGraph:
    supervisor_node = build_supervisor_node(
        llm, ["SubSupervisor", "FINISH"], supervisor_prompt
    )
    sub_supervisor_node = build_supervisor_node(
        llm, [expert.name for expert in experts] + ["RETURN"], sub_supervisor_prompt
    )

    builder = StateGraph(TeamState)
    for expert in experts:
        agent = create_react_agent(llm, tools=expert.tools, state_modifier=expert.system_prompt)

        def _run(state: TeamState, _agent=agent) -> dict[str, list]:
            return {"messages": [_agent.invoke(state)["messages"][-1]]}

        builder.add_node(expert.name, _run)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("sub_supervisor", sub_supervisor_node)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"SubSupervisor": "sub_supervisor", "FINISH": END},
    )
    builder.add_conditional_edges(
        "sub_supervisor",
        lambda x: x["next"],
        {expert.name: expert.name for expert in experts} | {"RETURN": "supervisor"},
    )
    for expert in experts:
        builder.add_edge(expert.name, "sub_supervisor")

    return builder


def compile_supervisor_graph(
    llm,
    experts: list[ExpertConfig],
    *,
    supervisor_prompt: str | None = None,
):
    return build_supervisor_graph(
        llm, experts, supervisor_prompt=supervisor_prompt
    ).compile()


def compile_admin_team_graph(
    llm,
    experts: list[ExpertConfig],
    *,
    supervisor_prompt: str | None = None,
    sub_supervisor_prompt: str | None = None,
):
    return build_admin_team_graph(
        llm,
        experts,
        supervisor_prompt=supervisor_prompt,
        sub_supervisor_prompt=sub_supervisor_prompt,
    ).compile()
