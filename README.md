# uni-lm

**LLM orchestrator for students with intent classification, streaming responses, and a user-friendly interface.**

## Description

uni-lm is a platform designed to help students interact with large language models (LLMs). The system automatically classifies user intent, plans tasks, and delegates them to specialized agents (planning, coding, theory, QA, summarization). It provides fast, streaming responses and a simple web interface.

## Architecture

- **Backend:** Python 3.12+, FastAPI, Redis, LangGraph, LiteLLM
  - Orchestrates LLM agents (planning, coding, theory, QA, summarization)
  - Stores and summarizes conversation history in Redis
  - REST API (endpoints: `/api/v1/ask`, `/api/v1/health`)
- **Frontend:** Next.js, React, TailwindCSS
  - Web interface for asking questions and receiving real-time responses

## Quick Start

### Backend

1. Go to the `backend/` directory
2. Install dependencies:
	```
	pip install -r requirements.txt
	```
3. Run the server:
	```
	uvicorn unilm.main:app --reload
	```
4. The API will be available at `http://localhost:8000`

### Frontend

1. Go to the `frontend/` directory
2. Install dependencies:
	```
	npm install
	```
3. Start the app:
	```
	npm run dev
	```
4. The interface will be available at `http://localhost:3000`

### Docker Compose

You can run everything (backend, frontend, Redis) with:
```
docker-compose up --build
```

## Testing

Backend: 
```
pytest
```

## Technologies

- **Backend:** FastAPI, LiteLLM, Redis
- **Frontend:** Next.js, React, TailwindCSS
