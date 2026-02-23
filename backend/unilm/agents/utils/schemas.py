from pydantic import BaseModel, Field
from typing import Literal


class Task(BaseModel):
    agent: Literal["coder", "theoretician"]
    description: str = Field(description="Instructions for agent what to do.")


class Plan(BaseModel):
    tasks: list[Task] = Field(description="List of steps to solve problem.")
    thought_process: str = Field(description="Short description of planning logic.")


class AgentOutput(BaseModel):
    agent: Literal["coder", "theoretician"]
    content: str
