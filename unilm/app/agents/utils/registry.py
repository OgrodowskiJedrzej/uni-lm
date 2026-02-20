from typing import overload, Literal

from agents.base import BaseModel
from agents.coding_agent import CodingAgent
from agents.planner import PlannerAgent
from agents.summerizer import SummerizerAgent
from agents.theoretician import TheoreticianAgent
from agents.reviewer import ReviewerAgent

class AgentRegistry:
    def __init__(self):
        self.agents = {
            "coder": CodingAgent(),
            "planner": PlannerAgent(),
            "summerizer": SummerizerAgent(),
            "theoretician": TheoreticianAgent(),
            "reviewer": ReviewerAgent()
        }
    
    @overload
    def get_agent(self, name: Literal["planner"]) -> PlannerAgent: ...
    
    @overload
    def get_agent(self, name: str) -> BaseModel: ...
    
    def get_agent(self, name: str) -> BaseModel | PlannerAgent | CodingAgent:
        if name not in self.agents:
            raise ValueError(f"Agent {name} not exist!")
        return self.agents.get(name, self.agents.get("coder"))