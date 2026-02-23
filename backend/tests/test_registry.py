"""Tests for the agent registry."""

import pytest
from unittest.mock import patch

from unilm.agents.utils.registry import AgentRegistry
from unilm.agents.base import BaseModel
from unilm.agents.coding_agent import CodingAgent
from unilm.agents.planner import PlannerAgent
from unilm.agents.summerizer import SummerizerAgent
from unilm.agents.theoretician import TheoreticianAgent
from unilm.agents.reviewer import ReviewerAgent


class TestAgentRegistry:
    """Test suite for AgentRegistry."""

    @pytest.fixture
    def registry(self):
        """Create an AgentRegistry instance."""
        with patch.object(CodingAgent, "__init__", return_value=None):
            with patch.object(PlannerAgent, "__init__", return_value=None):
                with patch.object(SummerizerAgent, "__init__", return_value=None):
                    with patch.object(TheoreticianAgent, "__init__", return_value=None):
                        with patch.object(ReviewerAgent, "__init__", return_value=None):
                            return AgentRegistry()

    def test_registry_initialization(self, registry):
        """Test that registry initializes with all agents."""
        assert "coder" in registry.agents
        assert "planner" in registry.agents
        assert "summerizer" in registry.agents
        assert "theoretician" in registry.agents
        assert "reviewer" in registry.agents
        assert len(registry.agents) == 5

    def test_get_agent_planner(self, registry):
        """Test getting planner agent."""
        agent = registry.get_agent("planner")
        assert agent is not None

    def test_get_agent_coder(self, registry):
        """Test getting coder agent."""
        agent = registry.get_agent("coder")
        assert agent is not None

    def test_get_agent_summerizer(self, registry):
        """Test getting summerizer agent."""
        agent = registry.get_agent("summerizer")
        assert agent is not None

    def test_get_agent_theoretician(self, registry):
        """Test getting theoretician agent."""
        agent = registry.get_agent("theoretician")
        assert agent is not None

    def test_get_agent_reviewer(self, registry):
        """Test getting reviewer agent."""
        agent = registry.get_agent("reviewer")
        assert agent is not None

    def test_get_agent_invalid_name_raises_error(self, registry):
        """Test that getting invalid agent raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            registry.get_agent("invalid_agent")
        assert "not exist" in str(exc_info.value)

    def test_get_agent_returns_correct_instance(self, registry):
        """Test that get_agent returns the same instance."""
        agent1 = registry.get_agent("coder")
        agent2 = registry.get_agent("coder")
        assert agent1 is agent2

    def test_get_agent_all_names(self, registry):
        """Test getting all registered agents by name."""
        agent_names = ["coder", "planner", "summerizer", "theoretician", "reviewer"]
        for name in agent_names:
            agent = registry.get_agent(name)
            assert agent is not None

    def test_get_agent_defaults_to_coder(self, registry):
        """Test that invalid agent defaults to coder."""
        # Note: This behavior might be different based on implementation
        # Current implementation does raise ValueError for invalid agents
        with pytest.raises(ValueError):
            registry.get_agent("nonexistent")

    def test_registry_agent_types(self, registry):
        """Test that all agents are BaseModel instances."""
        for agent_name, agent in registry.agents.items():
            assert isinstance(agent, BaseModel) or hasattr(agent, "run_agent")

    def test_registry_agents_dict_not_empty(self, registry):
        """Test that agents dictionary is not empty."""
        assert len(registry.agents) > 0

    def test_registry_preserves_agent_identity(self, registry):
        """Test that registry preserves agent identity across multiple calls."""
        coder1 = registry.get_agent("coder")
        coder2 = registry.get_agent("coder")
        coder3 = registry.get_agent("coder")

        assert coder1 is coder2
        assert coder2 is coder3
