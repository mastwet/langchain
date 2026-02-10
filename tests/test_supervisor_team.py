import unittest
from unittest.mock import patch

from tender_risk_agent.supervisor_team import (
    ExpertConfig,
    build_admin_team_graph,
    build_default_experts,
    build_supervisor_graph,
    build_supervisor_node,
)


class _FakeLLM:
    def __init__(self, choice: str):
        self.choice = choice
        self.calls = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return type("Resp", (), {"next_step": self.choice})()


class SupervisorTeamTests(unittest.TestCase):
    def test_supervisor_node_returns_next(self):
        llm = _FakeLLM("ExpertA")
        node = build_supervisor_node(llm, ["ExpertA", "FINISH"], prompt_text="Pick: {options}")
        result = node({"messages": ["hi"], "next": ""})
        self.assertEqual(result["next"], "ExpertA")
        self.assertTrue(llm.calls)

    @patch("tender_risk_agent.supervisor_team.create_react_agent")
    def test_build_supervisor_graph_has_expert_nodes(self, create_agent):
        create_agent.return_value = type("Agent", (), {"invoke": lambda self, s: {"messages": ["ok"]}})()
        llm = _FakeLLM("FINISH")
        experts = [ExpertConfig(name="ExpertA", system_prompt="x", tools=[])]
        graph = build_supervisor_graph(llm, experts)
        compiled = graph.compile()
        self.assertTrue(compiled)

    @patch("tender_risk_agent.supervisor_team.create_react_agent")
    def test_build_admin_team_graph_compiles(self, create_agent):
        create_agent.return_value = type("Agent", (), {"invoke": lambda self, s: {"messages": ["ok"]}})()
        llm = _FakeLLM("RETURN")
        experts = [ExpertConfig(name="ExpertA", system_prompt="x", tools=[])]
        graph = build_admin_team_graph(
            llm,
            experts,
            supervisor_prompt="Pick: {options}",
            sub_supervisor_prompt="Pick: {options}",
        )
        compiled = graph.compile()
        self.assertTrue(compiled)

    def test_build_default_experts(self):
        experts = build_default_experts()
        self.assertEqual(len(experts), 3)


if __name__ == "__main__":
    unittest.main()
