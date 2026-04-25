import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import app


@pytest.fixture
async def auth_headers() -> dict[str, str]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/auth/token",
            data={"username": settings.auth_username, "password": "changeme"},
        )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
