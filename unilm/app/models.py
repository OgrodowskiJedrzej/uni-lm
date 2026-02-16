from pydantic import BaseModel
import litellm
from .utils import select_model, get_agent_prompt
import logging

logger = logging.getLogger(__name__)

async def get_final_anwer(intent: str, user_query: str):
    """
    Generate final answer by streaming LLM response.
    
    :param intent: Classified user intent.
    :param user_query: Original user query.
    :param image: Optional image URL for vision tasks.
    :yields: Chunks of the streamed response.
    :raises ValueError: If model selection or prompt retrieval fails.
    """
    try:
        model = select_model(intent)
        prompt = get_agent_prompt(intent)
        
        user_content = [{"type": "text", "text": user_query}]
        
        logger.info(f"Streaming response for intent '{intent}' using model '{model}'")
        
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"content": prompt, "role": "system"},
                {"content": user_content, "role": "user"}
            ],
            temperature= 0 if intent == "tests" else 0.7,
            stream=True,
            timeout=120
        )
        
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None and content != "":
                yield str(content)

    
    except litellm.APIError as e:
        logger.error(f"LLM API error: {e}")
        yield f"Error: Failed to generate response. {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error during response generation: {e}")
        yield f"Error: Unexpected error occurred. {str(e)}"