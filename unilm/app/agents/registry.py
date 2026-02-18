from typing import overload, Literal
from .base import BaseModel
from .coding_agent import CodingAgent
from .planner import PlannerAgent

class AgentRegistry:
    def __init__(self):
        self.agents = {
            "coder": CodingAgent(),
            "planner": PlannerAgent()
        }
    
    @overload
    def get_agent(self, name: Literal["planner"]) -> PlannerAgent: ...
    
    @overload
    def get_agent(self, name: Literal["coder"]) -> CodingAgent: ...
    
    @overload
    def get_agent(self, name: str) -> BaseModel: ...
    
    def get_agent(self, name: str) -> BaseModel | PlannerAgent | CodingAgent:
        return self.agents.get(name, self.agents.get("coder"))