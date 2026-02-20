import json
import logging
from logging import DEBUG
from typing import TypedDict

import litellm
from litellm.caching.caching import Cache
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.utils.registry import AgentRegistry
from agents.utils.schemas import Plan, AgentOutput
from memory import RedisMemoryManager

logger = logging.getLogger(__name__)
logger.setLevel(DEBUG)
logging.basicConfig(filename="../app.log")

class AgentState(TypedDict):
    query: str
    plan: Plan | None
    results: list[AgentOutput]
    final_answer: str
    summary: str
    session_id: str | None


class Orchestrator():
    def __init__(self):
        self.registry = AgentRegistry()
        self.memory = RedisMemoryManager(history_threshold=10, agent=self.registry.get_agent("summerizer"))
        self.checkpointer = MemorySaver()
        self.workflow = self.compile_workflow()
    
    async def plan_node(self, state: AgentState) -> dict[str, dict]:
        planner = self.registry.get_agent(name="planner")
        self.memory.add_message(state["session_id"], "user", state["query"])
        plan = await planner.create_plan(state["query"])
        logger.debug(f"Plan: {plan}")
        return {"plan": plan}
    
    async def execute_node(self, state: AgentState) -> dict[str, list[dict]]:
        current_plan = state["plan"]
        session_id = state.get("session_id")

        new_results = []
        
        for task in current_plan.tasks:
            agent = self.registry.get_agent(task.agent)
            context = {"plan_logic": current_plan.thought_process}
            context["memory"] = self.memory.get_context(session_id)
            logger.debug(f"Context: {context}")
            logger.debug(f"Task: {task.description} | Agent: {task.agent}")
            output = await agent.run_agent(
                task.description,
                context=context
            )
            new_results.append(output)
            self.memory.add_message(session_id, "assistant",
                            output.content, agent=task.agent)
            logger.debug(f"Results: {new_results}")
        return {"results": new_results}
    
    def compile_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("planner", self.plan_node)
        workflow.add_node("executor", self.execute_node)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", END)

        return workflow.compile(checkpointer=self.checkpointer)    

    async def get_stream_response(self, query: str, session_id: str):
        input_state = {
            "query": query,
            "session_id": session_id,
            "plan": None,
            "results": []
        }

        config = {"configurable": {"thread_id": session_id}}

        async for event in self.workflow.astream(
            input_state,
            config=config,
            stream_mode="updates"
        ):
            node_name = list(event.keys())[0]
            node_data = event[node_name]

            if node_name == "executor":
                for result in node_data.get("results", []):
                    if hasattr(result, 'content'):
                        formatted_content = f"\n\n{result.content}\n\n"
                        yield f"data: {json.dumps({'content': formatted_content, 'agent': result.agent})}\n\n"
                    elif node_name == "planner":
                        logger.info(f"Plan created for session {session_id}")
