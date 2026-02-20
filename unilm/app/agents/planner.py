import litellm

from app.agents.base import BaseModel
from app.agents.utils.schemas import Plan

class PlannerAgent(BaseModel):
    def __init__(self):
        super().__init__(name = "planner")
        
    async def create_plan(self, query: str) -> list:
        response = await litellm.acompletion(
            model = self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ],
            response_format=Plan,
            temperature = self.temperature
        )
        content = response.choices[0].message.content
        return Plan.model_validate_json(content)