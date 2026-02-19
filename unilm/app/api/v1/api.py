from fastapi import APIRouter, Header, Query
from fastapi.responses import StreamingResponse
import uuid

from app.orchestrator import Orchestrator

api_router = APIRouter()
orchestrator = Orchestrator()

@api_router.get("/ask")
async def ask(question: str):
    current_session_id = str(uuid.uuid4())

    return StreamingResponse(
        orchestrator.get_stream_response(
            query=question, session_id=current_session_id),
            media_type="text/event-stream"
        )


@api_router.get("/health")
async def health_check():
    return {"status": "healthy"}
