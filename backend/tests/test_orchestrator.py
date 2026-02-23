"""Tests for the orchestrator."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from unilm.orchestrator import Orchestrator, AgentState
from unilm.agents.utils.schemas import Plan, Task, AgentOutput
from unilm.agents.utils.registry import AgentRegistry
from unilm.memory import RedisMemoryManager


class TestOrchestrator:
    """Test suite for Orchestrator."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock agent registry."""
        registry = MagicMock(spec=AgentRegistry)

        mock_planner = MagicMock()
        mock_planner.create_plan = AsyncMock(
            return_value=Plan(
                tasks=[Task(agent="coder", description="Write code")],
                thought_process="Plan logic",
            )
        )

        mock_coder = MagicMock()
        mock_coder.run_agent = AsyncMock(
            return_value=AgentOutput(agent="coder", content="def test(): pass")
        )

        # Inicjalizujemy jako MagicMock, będziemy nadpisywać side_effect w konkretnych testach
        mock_coder.run_agent_stream = MagicMock()

        def get_agent_side_effect(name):
            if name == "planner":
                return mock_planner
            return mock_coder

        registry.get_agent = MagicMock(side_effect=get_agent_side_effect)
        return registry, {"planner": mock_planner, "coder": mock_coder}

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory manager."""
        memory = MagicMock(spec=RedisMemoryManager)
        memory.add_message = AsyncMock()
        memory.get_context = MagicMock(return_value="Test context")
        return memory

    @pytest.fixture
    def orchestrator(self, mock_registry, mock_memory):
        """Create an Orchestrator instance with mocks."""
        # Używamy pełnych ścieżek do patchowania zgodnie ze strukturą unilm.app
        with patch("unilm.orchestrator.AgentRegistry", return_value=mock_registry[0]):
            with patch(
                "unilm.orchestrator.RedisMemoryManager", return_value=mock_memory
            ):
                orch = Orchestrator()
                return orch, mock_registry[1], mock_memory

    @pytest.mark.asyncio
    async def test_plan_node_basic(self, orchestrator):
        """Test plan_node creates a plan."""
        orch, agents, memory = orchestrator
        state = {
            "query": "Write a test",
            "plan": None,
            "results": [],
            "final_answer": "",
            "summary": "",
            "session_id": "test-session",
        }

        result = await orch.plan_node(state)

        assert "plan" in result
        assert isinstance(result["plan"], Plan)
        memory.add_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_node_single_task(self, orchestrator):
        """Test execute_node with a single task."""
        orch, agents, memory = orchestrator
        plan = Plan(
            tasks=[Task(agent="coder", description="Write code")],
            thought_process="Plan logic",
        )
        state = {
            "query": "Test",
            "plan": plan,
            "results": [],
            "final_answer": "",
            "summary": "",
            "session_id": "test-session",
        }

        result = await orch.execute_node(state)

        assert "results" in result
        assert len(result["results"]) == 1
        assert isinstance(result["results"][0], AgentOutput)

    @pytest.mark.asyncio
    async def test_execute_node_stream(self, orchestrator):
        """Test streaming execution fixing the __aiter__ error."""
        orch, agents, memory = orchestrator

        async def mock_stream_factory(*args, **kwargs):
            yield "Test"
            yield " response"

        agents["coder"].run_agent_stream.side_effect = mock_stream_factory

        plan = Plan(
            tasks=[Task(agent="coder", description="Stream task")],
            thought_process="Test",
        )
        state = {
            "query": "Test",
            "plan": plan,
            "results": [],
            "final_answer": "",
            "summary": "",
            "session_id": "test-session",
        }

        result_chunks = []
        async for chunk in orch.execute_node_stream(state):
            result_chunks.append(chunk)

        assert len(result_chunks) == 2
        assert "Test" in result_chunks[0]["content"]

    @pytest.mark.asyncio
    async def test_get_stream_response(self, orchestrator):
        """Test full streaming response generation with SSE formatting."""
        orch, agents, memory = orchestrator

        async def code_stream(*args, **kwargs):
            yield "Test content"

        agents["coder"].run_agent_stream.side_effect = code_stream

        response_lines = []
        async for line in orch.get_stream_response("Test query", "session-123"):
            response_lines.append(line)

        assert len(response_lines) > 0
        # Weryfikacja formatu SSE (Server-Sent Events)
        assert any("data:" in str(line) for line in response_lines)

    @pytest.mark.asyncio
    async def test_execute_node_stream_with_agent_switch(self, orchestrator):
        """Test streaming with multiple agents."""
        orch, agents, memory = orchestrator

        async def theory_stream(*args, **kwargs):
            yield "Theory"

        async def code_stream(*args, **kwargs):
            yield "Code"

        mock_theoretician = MagicMock()
        mock_theoretician.run_agent_stream.side_effect = theory_stream

        agents["coder"].run_agent_stream.side_effect = code_stream

        def get_agent_side_effect(name):
            if name == "theoretician":
                return mock_theoretician
            return agents["coder"]

        orch.registry.get_agent.side_effect = get_agent_side_effect

        plan = Plan(
            tasks=[
                Task(agent="theoretician", description="Analyze"),
                Task(agent="coder", description="Code"),
            ],
            thought_process="Multi-step",
        )
        state = {
            "query": "Test",
            "plan": plan,
            "results": [],
            "final_answer": "",
            "summary": "",
            "session_id": "test",
        }

        result_chunks = []
        async for chunk in orch.execute_node_stream(state):
            result_chunks.append(chunk)

        assert len(result_chunks) >= 2

    @pytest.mark.asyncio
    async def test_get_stream_response_saves_to_memory(self, orchestrator):
        """Test that streaming response saves final result to memory."""
        orch, agents, memory = orchestrator

        async def code_stream(*args, **kwargs):
            yield "final answer"

        agents["coder"].run_agent_stream.side_effect = code_stream

        async for _ in orch.get_stream_response("Query", "session-123"):
            pass

        # Sprawdzamy czy add_message zostało wywołane dla odpowiedzi asystenta
        assert memory.add_message.called
        # Ostatnie wywołanie powinno być rolą assistant
        last_call_args = memory.add_message.call_args_list[-1]
        assert (
            "assistant" in last_call_args[0]
            or last_call_args[1].get("role") == "assistant"
        )

    def test_agent_state_structure(self):
        """Test AgentState TypedDict structure."""
        state: AgentState = {
            "query": "Test",
            "plan": None,
            "results": [],
            "final_answer": "",
            "summary": "",
            "session_id": "test",
        }
        assert state["query"] == "Test"
        assert isinstance(state["results"], list)
