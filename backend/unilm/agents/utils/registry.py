from typing import overload, Literal

from unilm.agents.base import BaseModel
from unilm.agents.coding_agent import CodingAgent
from unilm.agents.planner import PlannerAgent
from unilm.agents.summerizer import SummerizerAgent
from unilm.agents.theoretician import TheoreticianAgent
from unilm.agents.reviewer import ReviewerAgent


class AgentRegistry:
    def __init__(self):
        self.agents = {
            "coder": CodingAgent(),
            "planner": PlannerAgent(),
            "summerizer": SummerizerAgent(),
            "theoretician": TheoreticianAgent(),
            "reviewer": ReviewerAgent(),
        }

    @overload
    def get_agent(self, name: Literal["planner"]) -> PlannerAgent: ...

    @overload
    def get_agent(self, name: str) -> BaseModel: ...

    def get_agent(self, name: str) -> BaseModel | PlannerAgent | CodingAgent:
        if name not in self.agents:
            raise ValueError(f"Agent {name} not exist!")
        return self.agents.get(name, self.agents.get("coder"))
