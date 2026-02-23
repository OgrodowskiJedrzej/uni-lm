"""Tests for the memory manager."""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch

from unilm.memory import RedisMemoryManager
from unilm.agents.base import BaseModel


class TestRedisMemoryManager:
    """Test suite for RedisMemoryManager."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis instance."""
        return MagicMock()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock(spec=BaseModel)
        agent.run_agent = AsyncMock(return_value="Summary of conversation")
        return agent

    @pytest.fixture
    def memory_manager(self, mock_redis, mock_agent):
        """Create a RedisMemoryManager with mocked Redis."""
        with patch("unilm.memory.redis.Redis", return_value=mock_redis):
            manager = RedisMemoryManager(
                agent=mock_agent,
                redis_host="localhost",
                redis_port=6379,
                history_threshold=3,
            )
            manager.redis = mock_redis
            return manager

    def test_initialization(self, mock_agent):
        """Test RedisMemoryManager initialization."""
        with patch("unilm.memory.redis.Redis"):
            manager = RedisMemoryManager(agent=mock_agent)
            assert manager.threshold == 10
            assert manager.summerize_agent == mock_agent

    def test_initialization_with_custom_threshold(self, mock_agent):
        """Test initialization with custom threshold."""
        with patch("unilm.memory.redis.Redis"):
            manager = RedisMemoryManager(agent=mock_agent, history_threshold=5)
            assert manager.threshold == 5

    def test_initialization_with_custom_redis_host(self, mock_agent):
        """Test initialization with custom Redis host."""
        with patch("unilm.memory.redis.Redis") as mock_redis_class:
            RedisMemoryManager(
                agent=mock_agent,
                redis_host="custom.host",
                redis_port=6380,
            )
            mock_redis_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_message(self, memory_manager, mock_redis):
        """Test adding a message to memory."""
        session_id = "test-session"
        mock_redis.rpush = MagicMock(return_value=1)
        mock_redis.expire = MagicMock()
        mock_redis.llen = MagicMock(return_value=1)

        await memory_manager.add_message(session_id, "user", "Test message")

        expected_key = f"session:{session_id}:history"
        assert mock_redis.rpush.called
        assert mock_redis.expire.called

    @pytest.mark.asyncio
    async def test_add_message_with_agent(self, memory_manager, mock_redis):
        """Test adding a message with agent information."""
        session_id = "test-session"
        mock_redis.rpush = MagicMock(return_value=1)
        mock_redis.expire = MagicMock()
        mock_redis.llen = MagicMock(return_value=1)

        await memory_manager.add_message(
            session_id, "assistant", "Test response", agent="coder"
        )

        assert mock_redis.rpush.called

    @pytest.mark.asyncio
    async def test_add_message_triggers_summarization(self, memory_manager, mock_redis):
        """Test that adding message triggers summarization when threshold exceeded."""
        session_id = "test-session"
        mock_redis.rpush = MagicMock(return_value=3)
        mock_redis.expire = MagicMock()
        mock_redis.llen = MagicMock(return_value=3)  # At threshold
        mock_redis.lrange = MagicMock(return_value=[
            json.dumps({"role": "user", "content": "msg1", "agent": None}),
            json.dumps({"role": "assistant", "content": "resp1", "agent": "coder"}),
            json.dumps({"role": "user", "content": "msg2", "agent": None}),
        ])

        with patch.object(memory_manager, "_summarize", new_callable=AsyncMock):
            await memory_manager.add_message(session_id, "user", "Message")
            assert memory_manager._summarize.called

    def test_get_history(self, memory_manager, mock_redis):
        """Test getting message history."""
        session_id = "test-session"
        messages = [
            json.dumps({"role": "user", "content": "Question", "agent": None}),
            json.dumps({"role": "assistant", "content": "Answer", "agent": "coder"}),
        ]
        mock_redis.lrange = MagicMock(return_value=messages)

        history = memory_manager.get_history(session_id)

        assert len(history) == 2
        assert history[0]["content"] == "Question"
        assert history[1]["agent"] == "coder"

    def test_get_history_empty(self, memory_manager, mock_redis):
        """Test getting history when empty."""
        session_id = "empty-session"
        mock_redis.lrange = MagicMock(return_value=[])

        history = memory_manager.get_history(session_id)

        assert history == []

    def test_get_context_no_summary(self, memory_manager, mock_redis):
        """Test getting context without summary."""
        session_id = "test-session"
        mock_redis.get = MagicMock(return_value=None)
        mock_redis.lrange = MagicMock(return_value=[
            json.dumps({"role": "user", "content": "Test message", "agent": None}),
        ])

        context = memory_manager.get_context(session_id)

        assert "Recent messages:" in context
        assert "Test message" in context

    def test_get_context_with_summary(self, memory_manager, mock_redis):
        """Test getting context with summary."""
        session_id = "test-session"
        summary_text = "This is a summary"
        mock_redis.get = MagicMock(return_value=summary_text)
        mock_redis.lrange = MagicMock(return_value=[
            json.dumps({"role": "user", "content": "Question", "agent": None}),
        ])

        context = memory_manager.get_context(session_id)

        assert "Summary:" in context
        assert summary_text in context
        assert "Recent messages:" in context

    def test_get_context_empty(self, memory_manager, mock_redis):
        """Test getting context when no history."""
        session_id = "empty-session"
        mock_redis.get = MagicMock(return_value=None)
        mock_redis.lrange = MagicMock(return_value=[])

        context = memory_manager.get_context(session_id)

        assert context == "No context available."

    def test_message_serialization(self, memory_manager, mock_redis):
        """Test that messages are properly serialized."""
        session_id = "test-session"
        role = "user"
        content = "Test content"
        agent = "coder"

        mock_redis.rpush = MagicMock()
        mock_redis.expire = MagicMock()
        mock_redis.llen = MagicMock(return_value=1)

        import asyncio
        asyncio.run(
            memory_manager.add_message(session_id, role, content, agent=agent)
        )

        # Check that rpush was called with JSON serialized data
        call_args = mock_redis.rpush.call_args
        assert call_args is not None

    def test_redis_key_format(self, memory_manager, mock_redis):
        """Test correct Redis key formatting."""
        session_id = "my-session-123"
        mock_redis.lrange = MagicMock(return_value=[])

        memory_manager.get_history(session_id)

        expected_key = f"session:{session_id}:history"
        call_args = mock_redis.lrange.call_args
        assert call_args[0][0] == expected_key
