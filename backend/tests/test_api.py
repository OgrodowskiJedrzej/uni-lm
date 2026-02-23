"""Tests for the API endpoints."""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch

from fastapi.testclient import TestClient

from unilm.main import app
from unilm.agents.utils.schemas import Plan, Task, AgentOutput


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


class TestAPIEndpoints:
    """Test suite for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct message."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "University AI Engine" in data["message"]

    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    def test_ask_endpoint_without_session(self, client):
        """Test ask endpoint without session_id."""
        with patch("unilm.api.v1.api.orchestrator") as mock_orchestrator:
            async def mock_stream(*args, **kwargs):
                yield b'data: {"content": "test", "agent": "coder"}\n\n'

            mock_orchestrator.get_stream_response.return_value = mock_stream()

            response = client.get("/api/v1/ask?question=Test%20question")

            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

    @pytest.mark.asyncio
    def test_ask_endpoint_with_session(self, client):
        """Test ask endpoint with session_id."""
        with patch("unilm.api.v1.api.orchestrator") as mock_orchestrator:
            async def mock_stream(*args, **kwargs):
                yield b'data: {"content": "response", "agent": "coder"}\n\n'

            mock_orchestrator.get_stream_response.return_value = mock_stream()

            response = client.get(
                "/api/v1/ask?question=Test&session_id=session-123"
            )

            assert response.status_code == 200

    def test_ask_endpoint_empty_question(self, client):
        """Test ask endpoint with empty question."""
        with patch("unilm.api.v1.api.orchestrator") as mock_orchestrator:
            async def mock_stream(*args, **kwargs):
                yield b'data: {"content": "", "agent": "coder"}\n\n'

            mock_orchestrator.get_stream_response.return_value = mock_stream()

            response = client.get("/api/v1/ask?question=")

            assert response.status_code == 200
            

    def test_ask_endpoint_generates_session_if_not_provided(self, client):
        """Test that ask endpoint generates session_id if not provided."""
        with patch("unilm.api.v1.api.orchestrator") as mock_orchestrator:
            with patch("unilm.api.v1.api.uuid.uuid4") as mock_uuid:
                mock_uuid.return_value = "generated-session-id"

                async def mock_stream(*args, **kwargs):
                    yield b'data: {"content": "test", "agent": "coder"}\n\n'

                mock_orchestrator.get_stream_response.return_value = mock_stream()

                response = client.get("/api/v1/ask?question=Test")

                assert response.status_code == 200
                mock_orchestrator.get_stream_response.assert_called()

    @pytest.mark.asyncio
    def test_streaming_response_format(self, client):
        """Test that streaming response has correct format."""
        with patch("unilm.api.v1.api.orchestrator") as mock_orchestrator:
            async def mock_stream(*args, **kwargs):
                data1 = json.dumps({"content": "Part1", "agent": "coder"})
                data2 = json.dumps({"content": "Part2", "agent": "coder"})
                yield f"data: {data1}\n\n"
                yield f"data: {data2}\n\n"

            mock_orchestrator.get_stream_response.return_value = mock_stream()

            response = client.get("/api/v1/ask?question=Test")

            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

    def test_api_router_prefix(self, client):
        """Test that API router has correct prefix."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_middleware_configuration(self, client):
        """Test that CORS middleware is properly configured."""
        response = client.options(
            "/ask", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
        assert response.status_code == 200  

    def test_ask_question_special_characters(self, client):
        """Test ask endpoint with special characters."""
        with patch("unilm.api.v1.api.orchestrator") as mock_orchestrator:
            async def mock_stream(*args, **kwargs):
                yield b'data: {"content": "response", "agent": "coder"}\n\n'

            mock_orchestrator.get_stream_response.return_value = mock_stream()

            question = "What is 1+1? @#$%^&*()"
            response = client.get(f"/api/v1/ask?question={question}")

            assert response.status_code == 200

    def test_health_check_always_works(self, client):
        """Test health check is always available."""
        for _ in range(5):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

    @pytest.mark.asyncio
    def test_ask_with_long_question(self, client):
        """Test ask endpoint with very long question."""
        with patch("unilm.api.v1.api.orchestrator") as mock_orchestrator:
            async def mock_stream(*args, **kwargs):
                yield b'data: {"content": "response", "agent": "coder"}\n\n'

            mock_orchestrator.get_stream_response.return_value = mock_stream()

            long_question = "Test? " * 1000
            response = client.get(f"/api/v1/ask?question={long_question[:500]}")

            assert response.status_code == 200


class TestAPIErrorHandling:
    """Test error handling in API."""

    def test_ask_without_question_parameter(self, client):
        """Test ask endpoint without question parameter."""
        response = client.get("/api/v1/ask")

        # Should fail validation
        assert response.status_code == 422

    def test_invalid_endpoint_returns_404(self, client):
        """Test invalid endpoint returns 404."""
        response = client.get("/api/v1/nonexistent")

        assert response.status_code == 404

    def test_invalid_http_method(self, client):
        """Test invalid HTTP method."""
        response = client.post("/api/v1/health")

        assert response.status_code == 405
