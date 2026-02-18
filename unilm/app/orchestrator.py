from pydantic import BaseModel
from typing import TypedDict, Annotated
import operator
import asyncio
from langgraph.graph import StateGraph, START, END

from agents.registry import AgentRegistry
from agents.schemas import Plan, AgentOutput

class AgentState(TypedDict):
    query: str
    plan: Plan
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str


registry = AgentRegistry()

async def plan_node(state: AgentState):
    planner = registry.get_agent(name = "planner")
    plan = await planner.create_plan(state["query"])
    return {"plan": plan}


async def execute_tasks_node(state: AgentState):
    current_plan = state["plan"] 
    new_results = []

    for task in current_plan.tasks:
        agent = registry.get_agent(task.agent)

        output = await agent.run_agent(
            task.description,
            context={"plan_logic": current_plan.thought_process}
        )
        new_results.append(output)

    return {"results": new_results}


workflow = StateGraph(AgentState)

workflow.add_node("planner", plan_node)
workflow.add_node("executor", execute_tasks_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", END)

orchestrator = workflow.compile()


async def test():
    input_state = {
        "query": "Napisz algorytm sortowania bąbelkowego i go wyjaśnij."}
    async for event in orchestrator.astream(input_state):
        print(event)

if __name__ == "__main__":
    asyncio.run(test())