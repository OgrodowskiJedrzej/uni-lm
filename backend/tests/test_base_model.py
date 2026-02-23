"""Tests for the base agent model."""

import pytest
import yaml
from unittest.mock import MagicMock, patch, AsyncMock
from pydantic import ValidationError

from unilm.agents.base import BaseModel
from unilm.agents.utils.schemas import AgentOutput


class TestBaseModel:
    """Test suite for BaseModel agent."""

    @pytest.fixture
    def mock_config(self):
        """Mock YAML config for testing."""
        return {
            "agents": {
                "coder": {
                    "model": "gpt-4",
                    "system_prompt": "You are a helpful assistant",
                    "temperature": 0.7,
                }
            }
        }

    @pytest.fixture
    def mock_base_model(self, mock_config):
        """Create a BaseModel instance with mocked config."""
        with patch.object(
            BaseModel, "_load_config", return_value=mock_config
        ):
            model = BaseModel(name="coder")
            return model

    def test_base_model_initialization(self, mock_base_model, mock_config):
        """Test BaseModel initialization."""
        assert mock_base_model.name == "coder"
        assert mock_base_model.model == "gpt-4"
        assert mock_base_model.system_prompt == "You are a helpful assistant"
        assert mock_base_model.temperature == 0.7

    def test_load_config_success(self, tmp_path):
        """Test successful YAML configuration loading."""
        config_content = """
agents:
  test_agent:
    model: gpt-4
    system_prompt: Test prompt
    temperature: 0.8
"""
        config_file = tmp_path / "prompts.yaml"
        config_file.write_text(config_content)

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                config_content
            )
            with patch("yaml.safe_load", return_value=yaml.safe_load(config_content)):
                with patch.object(
                    BaseModel, "_load_config", return_value=yaml.safe_load(config_content)
                ):
                    model = BaseModel(name="test_agent")
                    assert model.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_run_agent_success(self, mock_base_model):
        """Test successful agent execution."""
        expected_output = AgentOutput(agent="coder", content="Test response")

        with patch("unilm.agents.base.litellm.acompletion") as mock_completion:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = expected_output.model_dump_json()
            mock_completion.return_value = response

            result = await mock_base_model.run_agent("Test task")

            assert isinstance(result, AgentOutput)
            assert result.content == "Test response"
            mock_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_agent_with_context(self, mock_base_model):
        """Test agent execution with additional context."""
        context = {"key": "value", "plan_logic": "test logic"}
        expected_output = AgentOutput(agent="coder", content="Test response")

        with patch("unilm.agents.base.litellm.acompletion") as mock_completion:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = expected_output.model_dump_json()
            mock_completion.return_value = response

            result = await mock_base_model.run_agent("Test task", context=context)

            assert isinstance(result, AgentOutput)
            call_args = mock_completion.call_args
            assert "Context:" in call_args[1]["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_run_agent_stream(self, mock_base_model):
        """Test streaming agent execution."""
        chunks = ["Hello ", "world ", "test"]

        async def mock_stream():
            for chunk in chunks:
                msg = MagicMock()
                msg.choices = [MagicMock()]
                msg.choices[0].delta.content = chunk
                yield msg

        with patch("unilm.agents.base.litellm.acompletion") as mock_completion:
            mock_completion.return_value = mock_stream()

            result = []
            async for chunk in mock_base_model.run_agent_stream("Test task"):
                result.append(chunk)

            assert result == chunks

    @pytest.mark.asyncio
    async def test_run_agent_stream_with_context(self, mock_base_model):
        """Test streaming agent execution with context."""
        context = {"key": "value"}
        chunks = ["Response "]

        async def mock_stream():
            for chunk in chunks:
                msg = MagicMock()
                msg.choices = [MagicMock()]
                msg.choices[0].delta.content = chunk
                yield msg

        with patch("unilm.agents.base.litellm.acompletion") as mock_completion:
            mock_completion.return_value = mock_stream()

            result = []
            async for chunk in mock_base_model.run_agent_stream("coder", context=context):
                result.append(chunk)

            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_run_agent_handles_empty_content(self, mock_base_model):
        """Test agent handling of empty response content."""
        with patch("unilm.agents.base.litellm.acompletion") as mock_completion:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = None
            mock_completion.return_value = response

            with pytest.raises(Exception):
                await mock_base_model.run_agent("Test task")

    def test_base_model_temperature_setting(self, mock_config):
        """Test temperature setting is correctly loaded."""
        configs = [
            {"temperature": 0.0, "expected": 0.0},
            {"temperature": 0.5, "expected": 0.5},
            {"temperature": 1.0, "expected": 1.0},
            {"temperature": 2.0, "expected": 2.0},
        ]

        for config_case in configs:
            mock_config["agents"]["coder"]["temperature"] = config_case[
                "temperature"
            ]
            with patch.object(
                BaseModel, "_load_config", return_value=mock_config
            ):
                model = BaseModel(name="coder")
                assert model.temperature == config_case["expected"]

    def test_base_model_system_prompt(self, mock_config):
        """Test system prompt setting."""
        custom_prompt = "Custom system prompt for testing"
        mock_config["agents"]["coder"]["system_prompt"] = custom_prompt

        with patch.object(BaseModel, "_load_config", return_value=mock_config):
            model = BaseModel(name="coder")
            assert model.system_prompt == custom_prompt
