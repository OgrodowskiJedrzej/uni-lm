"""Tests for data schemas."""

import pytest
from pydantic import ValidationError

from unilm.agents.utils.schemas import Task, Plan, AgentOutput


class TestTaskSchema:
    """Test suite for Task schema."""

    def test_task_creation_with_coder(self):
        """Test creating Task with coder agent."""
        task = Task(agent="coder", description="Write a function")

        assert task.agent == "coder"
        assert task.description == "Write a function"

    def test_task_creation_with_theoretician(self):
        """Test creating Task with theoretician agent."""
        task = Task(agent="theoretician", description="Analyze the problem")

        assert task.agent == "theoretician"
        assert task.description == "Analyze the problem"

    def test_task_invalid_agent_type(self):
        """Test Task with invalid agent type."""
        with pytest.raises(ValidationError):
            Task(agent="invalid_agent", description="Test")

    def test_task_empty_description(self):
        """Test Task with empty description."""
        task = Task(agent="coder", description="")

        assert task.description == ""

    def test_task_long_description(self):
        """Test Task with very long description."""
        long_desc = "Task description. " * 1000
        task = Task(agent="coder", description=long_desc)

        assert task.description == long_desc

    def test_task_special_characters(self):
        """Test Task with special characters in description."""
        desc = 'Write code to solve @#$%^&*() problem with \'"strings"'
        task = Task(agent="coder", description=desc)

        assert task.description == desc

    def test_task_dict_conversion(self):
        """Test Task conversion to dictionary."""
        task = Task(agent="coder", description="Test")
        task_dict = task.model_dump()

        assert task_dict["agent"] == "coder"
        assert task_dict["description"] == "Test"

    def test_task_json_serialization(self):
        """Test Task JSON serialization."""
        task = Task(agent="coder", description="Test task")
        json_str = task.model_dump_json()

        assert '"agent":"coder"' in json_str or '"agent": "coder"' in json_str
        assert "Test task" in json_str


class TestPlanSchema:
    """Test suite for Plan schema."""

    def test_plan_creation(self):
        """Test creating a Plan."""
        tasks = [
            Task(agent="theoretician", description="Analyze"),
            Task(agent="coder", description="Code"),
        ]
        plan = Plan(tasks=tasks, thought_process="Multi-step approach")

        assert len(plan.tasks) == 2
        assert plan.thought_process == "Multi-step approach"

    def test_plan_single_task(self):
        """Test Plan with single task."""
        tasks = [Task(agent="coder", description="Simple task")]
        plan = Plan(tasks=tasks, thought_process="Simple plan")

        assert len(plan.tasks) == 1

    def test_plan_empty_tasks(self):
        """Test Plan with empty task list."""
        plan = Plan(tasks=[], thought_process="Empty plan")

        assert plan.tasks == []
        assert plan.thought_process == "Empty plan"

    def test_plan_many_tasks(self):
        """Test Plan with many tasks."""
        tasks = [Task(agent="theoretician", description=f"Task {i}") for i in range(10)]
        plan = Plan(tasks=tasks, thought_process="Complex plan")

        assert len(plan.tasks) == 10

    def test_plan_dict_conversion(self):
        """Test Plan conversion to dictionary."""
        tasks = [Task(agent="coder", description="Code")]
        plan = Plan(tasks=tasks, thought_process="Test plan")
        plan_dict = plan.model_dump()

        assert "tasks" in plan_dict
        assert "thought_process" in plan_dict
        assert len(plan_dict["tasks"]) == 1

    def test_plan_json_serialization(self):
        """Test Plan JSON serialization."""
        tasks = [Task(agent="coder", description="Test")]
        plan = Plan(tasks=tasks, thought_process="Plan logic")
        json_str = plan.model_dump_json()

        assert "tasks" in json_str
        assert "thought_process" in json_str

    def test_plan_json_deserialization(self):
        """Test Plan JSON deserialization."""
        json_str = """{
            "tasks": [
                {"agent": "coder", "description": "Write code"}
            ],
            "thought_process": "Implementation plan"
        }"""

        plan = Plan.model_validate_json(json_str)

        assert len(plan.tasks) == 1
        assert plan.tasks[0].agent == "coder"
        assert plan.thought_process == "Implementation plan"

    def test_plan_complex_thought_process(self):
        """Test Plan with complex thought process."""
        complex_thought = (
            "1. Break down the problem\n2. Design solution\n3. Implement\n4. Test"
        )
        tasks = [Task(agent="coder", description="Implementation")]
        plan = Plan(tasks=tasks, thought_process=complex_thought)

        assert complex_thought in plan.thought_process


class TestAgentOutputSchema:
    """Test suite for AgentOutput schema."""

    def test_agent_output_creation_coder(self):
        """Test creating AgentOutput for coder."""
        output = AgentOutput(agent="coder", content="def hello():\n    print('hello')")

        assert output.agent == "coder"
        assert "def hello" in output.content

    def test_agent_output_creation_theoretician(self):
        """Test creating AgentOutput for theoretician."""
        output = AgentOutput(
            agent="theoretician", content="The solution involves analyzing..."
        )

        assert output.agent == "theoretician"

    def test_agent_output_invalid_agent(self):
        """Test AgentOutput with invalid agent type."""
        with pytest.raises(ValidationError):
            AgentOutput(agent="invalid", content="Test")

    def test_agent_output_empty_content(self):
        """Test AgentOutput with empty content."""
        output = AgentOutput(agent="coder", content="")

        assert output.content == ""

    def test_agent_output_long_content(self):
        """Test AgentOutput with very long content."""
        long_content = "Line of code. " * 10000
        output = AgentOutput(agent="coder", content=long_content)

        assert len(output.content) > 100000

    def test_agent_output_multiline_content(self):
        """Test AgentOutput with multiline content."""
        content = """def function():
    line1 = "value"
    line2 = "another"
    return line1 + line2"""

        output = AgentOutput(agent="coder", content=content)

        assert "\n" in output.content
        assert "def function" in output.content

    def test_agent_output_dict_conversion(self):
        """Test AgentOutput conversion to dictionary."""
        output = AgentOutput(agent="coder", content="Test output")
        output_dict = output.model_dump()

        assert output_dict["agent"] == "coder"
        assert output_dict["content"] == "Test output"

    def test_agent_output_json_serialization(self):
        """Test AgentOutput JSON serialization."""
        output = AgentOutput(agent="coder", content="Test")
        json_str = output.model_dump_json()

        assert '"agent":"coder"' in json_str or '"agent": "coder"' in json_str
        assert "Test" in json_str

    def test_agent_output_json_deserialization(self):
        """Test AgentOutput JSON deserialization."""
        json_str = '{"agent": "coder", "content": "Code output"}'

        output = AgentOutput.model_validate_json(json_str)

        assert output.agent == "coder"
        assert output.content == "Code output"

    def test_agent_output_with_special_characters(self):
        """Test AgentOutput with special characters."""
        content = "Code with special chars: @#$%^&*(){}[]<>?:\"'\\"
        output = AgentOutput(agent="coder", content=content)

        assert output.content == content

    def test_agent_output_json_with_special_chars(self):
        """Test AgentOutput JSON serialization with special characters."""
        content = 'String with "quotes" and \\backslashes\\ and newlines\nhere'
        output = AgentOutput(agent="coder", content=content)
        json_str = output.model_dump_json()

        output_parsed = AgentOutput.model_validate_json(json_str)
        assert output_parsed.content == content


class TestSchemasIntegration:
    """Integration tests for schemas."""

    def test_plan_with_agent_output_workflow(self):
        """Test workflow with Plan and AgentOutput."""
        # Create plan
        plan = Plan(
            tasks=[
                Task(agent="theoretician", description="Analyze"),
                Task(agent="coder", description="Implement"),
            ],
            thought_process="Analysis then implementation",
        )

        # Create outputs
        outputs = [
            AgentOutput(agent="theoretician", content="Analysis results"),
            AgentOutput(agent="coder", content="Implementation code"),
        ]

        assert len(plan.tasks) == len(outputs)
        assert plan.tasks[0].agent == outputs[0].agent
        assert plan.tasks[1].agent == outputs[1].agent

    def test_nested_serialization_deserialization(self):
        """Test serialization and deserialization of complex structures."""
        tasks = [
            Task(agent="theoretician", description="Analyze problem space"),
            Task(agent="coder", description="Write implementation"),
        ]
        plan = Plan(tasks=tasks, thought_process="Complete workflow")

        # Serialize
        plan_json = plan.model_dump_json()

        # Deserialize
        plan_restored = Plan.model_validate_json(plan_json)

        assert plan_restored.thought_process == plan.thought_process
        assert len(plan_restored.tasks) == len(plan.tasks)

    def test_all_agent_types_in_output(self):
        """Test AgentOutput with all valid agent types."""
        agent_types = ["coder", "theoretician"]

        for agent_type in agent_types:
            output = AgentOutput(agent=agent_type, content="Test output")
            assert output.agent == agent_type
