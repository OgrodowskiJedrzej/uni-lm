from fastapi import FastAPI
from app.api.v1.api import api_router


app = FastAPI(
    title="University Agentic AI API",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "University AI Engine is running. Go to /docs for Swagger UI."}
