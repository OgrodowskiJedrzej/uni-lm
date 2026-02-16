import litellm
from typing import Literal
from pydantic import BaseModel, Field
from .utils import get_orchestrator_prompt, select_model
import logging

logger = logging.getLogger(__name__)

class OrchestratorResponse(BaseModel):
    response: Literal["coding", "reasoning",
                      "tests", "normal"] = Field(default='normal')

class LLMRouter:
    def __init__(self):
        self.model = select_model("orchestrator")  
    
    async def classify_intention(self, prompt: str) -> str:
        """
        Classify user intent asynchronously.
        
        :param prompt: User prompt to classify.
        :type prompt: str
        :return: Predicted intention class.
        :rtype: str
        :raises ValueError: If intent classification fails.
        """
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt received for classification")
            return "normal"
        
        try:
            system_prompt = get_orchestrator_prompt()
            
            logger.debug(f"Classifying intent for prompt: {prompt[:100]}...")
            
            orchestrator_response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"content": system_prompt, "role": "system"},
                    {"content": prompt, "role": "user"}
                ],
                temperature=0,
                response_format=OrchestratorResponse,
                timeout=30
            )
            
            raw_content = orchestrator_response.choices[0].message.content
            classified = OrchestratorResponse.model_validate_json(raw_content)
            logger.info(f"Intent classified as: {classified.response}")
            return classified.response
        
        except litellm.APIError as e:
            logger.error(f"LLM API error during classification: {e}")
            return "normal"
        except ValueError as e:
            logger.error(f"Invalid response format from orchestrator: {e}")
            return "normal"
        except Exception as e:
            logger.error(f"Unexpected error during intent classification: {e}")
            return "normal"


