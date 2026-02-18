import litellm
from abc import ABC, abstractmethod
import yaml

from .schemas import AgentOutput

class BaseModel(ABC):
    def __init__(self, name: str):
        self.config = self._load_config("/Users/jendras/Projects/uni-lm/unilm/app/prompts.yaml")

        self.name = name
        self.model = self.config["agents"][name]["model"]
        self.system_prompt = self.config["agents"][name]["system_prompt"]
        self.temperature = self.config["agents"][name]["temperature"]


    def _load_config(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    async def run_agent(self, task: str, context: dict | None = None):
        response = await litellm.acompletion(
            model = self.model,
            messages = [
                { "role": "system", "content": self.system_prompt },
                { "role": "user", "content": f"Context: {context}\n\nTask: {task}" }
            ],
            temperature = self.temperature,
            response_format=AgentOutput
        )
    
        content = response.choices[0].message.content
        return AgentOutput.model_validate_json(content)
