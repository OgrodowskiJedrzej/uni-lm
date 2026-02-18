from typing import overload, Literal
from .base import BaseModel
from .coding_agent import CodingAgent
from .planner import PlannerAgent
from .summerizer import SummerizerAgent
from .theoretician import TheoreticianAgent
from .reviewer import ReviewerAgent

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