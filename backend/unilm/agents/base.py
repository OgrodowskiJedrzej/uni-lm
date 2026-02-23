import yaml
from abc import ABC
import os
from pathlib import Path

import litellm

from unilm.agents.utils.schemas import AgentOutput


class BaseModel(ABC):
    def __init__(self, name: str):
        self.config = self._load_config("prompts.yaml")

        # configs defined in prompts.yaml
        self.name = name
        self.model = self.config["agents"][name]["model"]
        self.system_prompt = self.config["agents"][name]["system_prompt"]
        self.temperature = self.config["agents"][name]["temperature"]

    def _load_config(self, filename="prompts.yaml"):
        env_path = os.getenv("PROMPTS_CONFIG_PATH")
        if env_path:
            path = Path(env_path)
        else:
            base_dir = Path(__file__).resolve().parent.parent.parent
            path = base_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Błąd: Nie znaleziono pliku promptów pod ścieżką: {path.absolute()}"
            )

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    async def run_agent(self, task: str, context: dict | None = None):
        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nTask: {task}"},
            ],
            temperature=self.temperature,
            response_format=AgentOutput,
        )
        content = response.choices[0].message.content
        return AgentOutput.model_validate_json(content)

    async def run_agent_stream(self, task: str, context: dict | None = None):
        async for chunk in await litellm.acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nTask: {task}"},
            ],
            temperature=self.temperature,
            stream=True,
        ):
            if hasattr(chunk.choices[0].delta, "content"):
                yield chunk.choices[0].delta.content
