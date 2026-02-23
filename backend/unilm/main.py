from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from unilm.api.v1.api import api_router


app = FastAPI(title="University Agentic AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "University AI Engine is running. Go to /docs for Swagger UI."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="0.0.0.0", port=8000)
