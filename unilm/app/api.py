from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from .models import get_final_anwer
from .router import LLMRouter

load_dotenv()

app = FastAPI()
llm_router = LLMRouter()

@app.post("/ask")
async def ask(user_query, image=None):
    classification = await llm_router.classify_intention(prompt=user_query)
    intent = classification

    return StreamingResponse(
        get_final_anwer(intent, user_query, image),
        media_type="text/plain"
    )
