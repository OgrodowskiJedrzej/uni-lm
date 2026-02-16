import litellm
from typing import Literal
from pydantic import BaseModel, Field

class OrchestratorResponse(BaseModel):
    response: Literal["coding", "reasoning",
                      "tests", "normal"] = Field(default='normal')

class LLMRouter:
    async def classify_intention(self, prompt: str) -> str:
        """
        Basic intention classifier used for task router.
        
        :param prompt: User prompt.
        :type prompt: str
        :return: Predicted intention class.
        :rtype: OrchestratorResponse
        """
        orchestrator_response = litellm.completion(
            model="gpt-4o-mini",
            messages=[
                { "content": "Classify intent of user.", "role": "system" },
                { "content": prompt, "role": "user"}
            ],
            temperature=0,
            response_format=OrchestratorResponse
        )

        raw_content = orchestrator_response.choices[0].message.content

        return OrchestratorResponse.model_validate_json(raw_content).response


