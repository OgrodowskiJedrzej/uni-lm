from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from .models import get_final_anwer
from .router import LLMRouter
from .utils import load_yaml, _config
import logging
import sys
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load configuration on app startup."""
    try:
        config = load_yaml()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration at startup: {e}")
        raise
    
    yield


app = FastAPI(
    title="uni-lm",
    description="LLM orchestrator for students",
    version="0.1.0",
    lifespan=lifespan
)

llm_router = LLMRouter()


class UserInput(BaseModel):
    user_query: str

    @field_validator("user_query")
    @classmethod
    def validate_query_length(cls, value: str) -> str:
        """Validate that user query is not excessively long."""
        if not value or not value.strip():
            raise ValueError("Query cannot be empty")
        if len(value) > 50000:
            raise ValueError("Query exceeds maximum length of 50,000 characters")
        return value.strip()


@app.post("/ask", response_class=StreamingResponse)
async def ask(input: UserInput):
    """
    Main endpoint to process user queries with intent classification and streaming response.
    
    Args:
        input: UserInput model with user_query and optional image URL.
    
    Returns:
        StreamingResponse with the generated answer.
    """
    try:
        logger.info(f"Processing query from user")
        
        classification = await llm_router.classify_intention(prompt=input.user_query)
        logger.info(f"Query classified as intent: {classification}")
        
        return StreamingResponse(
            get_final_anwer(classification, input.user_query),
            media_type="text/plain"
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in /ask endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}
