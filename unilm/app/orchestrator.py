from pydantic import BaseModel
from typing import TypedDict, Annotated
import operator
import asyncio
from langgraph.graph import StateGraph, START, END
import litellm
from litellm.caching.caching import Cache

from agents.utils.registry import AgentRegistry
from agents.utils.schemas import Plan, AgentOutput
from memory import RedisMemoryManager

class AgentState(TypedDict):
    query: str
    plan: Plan
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str
    summary: str
    session_id: str | None


registry = AgentRegistry()
memory = RedisMemoryManager(history_threshold=10, agent=registry.get_agent("summerizer"))

async def plan_node(state: AgentState):
    planner = registry.get_agent(name = "planner")
    
    if state.get("session_id"):
        memory.add_message(state["session_id"], "user", state["query"])
    
    plan = await planner.create_plan(state["query"])
    return {"plan": plan}


async def execute_tasks_node(state: AgentState):
    current_plan = state["plan"] 
    session_id = state.get("session_id")

    new_results = []
    for task in current_plan.tasks:
        agent = registry.get_agent(task.agent)
        
        context = {"plan_logic": current_plan.thought_process}
        context["memory"] = memory.get_context(session_id)

        output = await agent.run_agent(
            task.description,
            context=context
        )
        new_results.append(output)
        memory.add_message(session_id, "assistant", output.content, agent=task.agent)

    return {"results": new_results}


workflow = StateGraph(AgentState)

workflow.add_node("planner", plan_node)
workflow.add_node("executor", execute_tasks_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", END)

orchestrator = workflow.compile()

async def test():
    import uuid
    
    session_id = str(uuid.uuid4())
    input_state = {
        "query": "Wygeneruj hello world w javie.",
        "session_id": session_id,
        "plan": None,
        "results": [],
        "final_answer": "",
        "summary": ""
    }
    async for event in orchestrator.astream(input_state):
        print(event)
    input_state['query'] = "O co sie zapytalem poprzednio?"
    async for event in orchestrator.astream(input_state):
        print(event)
    

if __name__ == "__main__":
    asyncio.run(test())