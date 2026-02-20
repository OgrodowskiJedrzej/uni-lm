import logging

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import uuid

from app.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

api_router = APIRouter()
orchestrator = Orchestrator()


@api_router.get("/ask")
async def ask(question: str, session_id: str = Query(None)):
    current_session_id = session_id or str(uuid.uuid4())
    logging.debug(f"SessionId: {current_session_id}")
    return StreamingResponse(
        orchestrator.get_stream_response(query=question, session_id=current_session_id),
        media_type="text/event-stream",
    )


@api_router.get("/health")
async def health_check():
    return {"status": "healthy"}
