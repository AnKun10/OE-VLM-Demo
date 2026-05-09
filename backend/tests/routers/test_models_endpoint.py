"""Test that /api/models exposes capabilities."""
from __future__ import annotations


def test_models_endpoint_returns_capabilities(client, fake_manager):
    response = client.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    assert "models" in body
    for entry in body["models"]:
        assert "id" in entry
        assert "name" in entry
        assert "capabilities" in entry
        assert "vision" in entry["capabilities"]
        assert isinstance(entry["capabilities"]["vision"], bool)
