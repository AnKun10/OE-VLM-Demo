"""Shared fixtures for backend tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def fake_manager():
    """A MagicMock standing in for VLMManager. Tests overwrite `.stream`
    with their own async generator function as needed.
    """
    manager = MagicMock()
    manager.list_models.return_value = [
        {"id": "fake-vision", "name": "Fake Vision",
         "capabilities": {"vision": True}},
        {"id": "fake-text", "name": "Fake Text",
         "capabilities": {"vision": False}},
    ]
    return manager


@pytest.fixture
def client(fake_manager):
    """TestClient that does NOT enter the lifespan context (which would
    call real VLMManager.load()). We set `app.state.vlm_manager` directly.

    Tests must be run from `backend/` so the static mount on `/images`
    points at the existing `backend/images/` directory. Tests that touch
    file storage monkeypatch `app.services.files.IMAGES_DIR` per-test.
    """
    from app.main import app
    app.state.vlm_manager = fake_manager
    return TestClient(app)
