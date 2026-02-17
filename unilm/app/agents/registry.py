from .base import BaseModel
from .coding_agent import CodingAgent
from .planner import PlannerAgent

class AgentRegistry:
    def __init__(self):
        self.agents = {
            "coder": CodingAgent(),
            "planner": PlannerAgent()
        }
    
    def get_agent(self, name: str) -> BaseModel:
        return self.agents.get(name, self.agents("coder"))