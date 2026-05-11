# Qwen3-VL-8B Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Qwen3-VL-8B into the backend as a new selectable model served by a remote vLLM HTTP server, via a decomposed provider package (`config` / `transforms` / `provider`) that future model packages can mirror.

**Architecture:** New provider package `backend/app/models/vlm/providers/qwen_vllm/`. Reuses the OpenAI SDK but adds a pre-send transform layer for Qwen quirks (image-token stripping, `min_pixels`/`max_pixels` injection) and a 1-retry policy on connection errors. The FastAPI router, manager plumbing, and frontend stay untouched except for one `PROVIDER_MAP` line and the YAML registry.

**Tech Stack:** Python 3.11+, FastAPI, `openai` SDK (client only — server is remote vLLM), PyYAML, pytest (new), httpx (test-only dependency, already a transitive dep of `openai`).

**Spec:** `qwen3_vl_design.md`.

---

## File Structure

| Path | Role |
|---|---|
| `backend/app/models/vlm/providers/qwen_vllm/__init__.py` | Export `QwenVLLMProvider`. |
| `backend/app/models/vlm/providers/qwen_vllm/config.py` | Qwen-specific constants. Only file to touch when tuning. |
| `backend/app/models/vlm/providers/qwen_vllm/transforms.py` | Pure message transforms. No SDK, no I/O. |
| `backend/app/models/vlm/providers/qwen_vllm/provider.py` | `VLMProvider` subclass. Only file that imports `openai`. |
| `backend/app/models/vlm/manager.py` | +1 `PROVIDER_MAP` entry, +optional YAML overrides into provider kwargs. |
| `backend/app/models/vlm/models.yaml` | Remove LLaVA entry; add Qwen3-VL-8B entry. |
| `backend/requirements.txt` | Add `pytest`. |
| `backend/pytest.ini` | Configure pytest rootdir + `pythonpath`. |
| `backend/tests/vlm/providers/qwen_vllm/test_transforms.py` | Unit tests for pure transforms. |
| `backend/tests/vlm/providers/qwen_vllm/test_provider.py` | Unit tests for provider retry + transform orchestration. |
| `backend/scripts/smoke_qwen3_vl.py` | Manual end-to-end smoke test. |
| `qwen_development.md` | Operational notes + deferred work. |

---

## Task 1: Set up pytest infrastructure

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/pytest.ini`

- [ ] **Step 1: Add pytest to requirements**

Edit `backend/requirements.txt`. Current contents:

```
fastapi
uvicorn[standard]
pydantic
pydantic-settings
pillow
openai
pyyaml
vllm
bitsandbytes
```

Append `pytest` so the final file is:

```
fastapi
uvicorn[standard]
pydantic
pydantic-settings
pillow
openai
pyyaml
vllm
bitsandbytes
pytest
```

- [ ] **Step 2: Create `backend/pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 3: Install pytest in the active venv**

Run (from `backend/` with venv activated):
```
pip install pytest
```

Expected: `Successfully installed pytest-...`.

- [ ] **Step 4: Verify pytest runs (no tests yet)**

Run (from `backend/`):
```
pytest
```

Expected: exit code 5 — "no tests ran" or similar. This confirms pytest is installed and configured.

- [ ] **Step 5: Commit**

```
git add backend/requirements.txt backend/pytest.ini
git commit -m "chore: add pytest for backend unit tests"
```

---

## Task 2: Scaffold `qwen_vllm/` package and `config.py`

**Files:**
- Create: `backend/app/models/vlm/providers/qwen_vllm/__init__.py`
- Create: `backend/app/models/vlm/providers/qwen_vllm/config.py`

No TDD for pure constants — `config.py` holds values, not behavior.

- [ ] **Step 1: Create an empty-stub `__init__.py`**

`backend/app/models/vlm/providers/qwen_vllm/__init__.py` (one line — the real export is added in Task 5 once `provider.py` exists, so Tasks 3 and 4 can import `transforms` without triggering a missing-`provider` ImportError):

```python
# Populated in Task 5 once provider.py exists.
```

- [ ] **Step 2: Create `config.py`**

`backend/app/models/vlm/providers/qwen_vllm/config.py`:

```python
"""Constants and defaults for the Qwen vLLM provider.

This is the ONLY file in this package that should be edited to tune
hyperparameters for Qwen-family models served via vLLM.
"""

# Regex patterns stripped from user text (quirk i — image-token leakage).
IMAGE_TOKEN_PATTERNS: tuple[str, ...] = (
    r"<image>",
    r"<\|image_pad\|>",
    r"<\|vision_start\|>",
    r"<\|vision_end\|>",
)

# Default pixel bounds for the Qwen processor (quirk ii).
# Per development_roadmap.md: min = 256*28*28 = 200704; working max = 1605632.
DEFAULT_MIN_PIXELS: int = 200_704
DEFAULT_MAX_PIXELS: int = 1_605_632

# HTTP policy (quirk iv).
REQUEST_TIMEOUT_S: float = 120.0
MAX_RETRIES: int = 1
RETRY_BACKOFF_S: float = 1.0
```

- [ ] **Step 3: Sanity-import from a Python REPL**

From `backend/` with venv active:
```
python -c "from app.models.vlm.providers.qwen_vllm import config; print(config.DEFAULT_MIN_PIXELS, config.DEFAULT_MAX_PIXELS)"
```

Expected output: `200704 1605632`.

- [ ] **Step 4: Commit**

```
git add backend/app/models/vlm/providers/qwen_vllm/
git commit -m "feat(vlm): scaffold qwen_vllm package with config constants"
```

---

## Task 3: TDD `transforms.strip_image_tokens`

**Files:**
- Create: `backend/tests/vlm/providers/qwen_vllm/test_transforms.py`
- Create: `backend/app/models/vlm/providers/qwen_vllm/transforms.py`

- [ ] **Step 1: Write failing tests for `strip_image_tokens`**

Create `backend/tests/vlm/providers/qwen_vllm/test_transforms.py`:

```python
import copy

from app.models.vlm.providers.qwen_vllm.transforms import strip_image_tokens


def test_strip_image_tokens_removes_image_tag():
    msgs = [{"role": "user", "content": "Hello <image> world"}]
    out = strip_image_tokens(msgs)
    assert out[0]["content"] == "Hello  world"


def test_strip_image_tokens_removes_all_patterns():
    msgs = [{
        "role": "user",
        "content": "<image><|image_pad|><|vision_start|><|vision_end|>hi",
    }]
    out = strip_image_tokens(msgs)
    assert out[0]["content"] == "hi"


def test_strip_image_tokens_leaves_image_url_parts_untouched():
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            {"type": "text", "text": "look <image>"},
        ],
    }]
    out = strip_image_tokens(msgs)
    assert out[0]["content"][0]["image_url"]["url"] == "data:image/png;base64,AAA"
    assert out[0]["content"][1]["text"] == "look "


def test_strip_image_tokens_leaves_non_user_turns_untouched():
    msgs = [
        {"role": "assistant", "content": "I see <image> there"},
        {"role": "system", "content": "<image>"},
    ]
    out = strip_image_tokens(msgs)
    assert out[0]["content"] == "I see <image> there"
    assert out[1]["content"] == "<image>"


def test_strip_image_tokens_idempotent_on_clean_text():
    msgs = [{"role": "user", "content": "Hello world"}]
    out = strip_image_tokens(msgs)
    assert out[0]["content"] == "Hello world"


def test_strip_image_tokens_does_not_mutate_input():
    msgs = [{"role": "user", "content": "Hello <image>"}]
    snapshot = copy.deepcopy(msgs)
    strip_image_tokens(msgs)
    assert msgs == snapshot
```

- [ ] **Step 2: Run tests — verify they fail**

From `backend/`:
```
pytest tests/vlm/providers/qwen_vllm/test_transforms.py -v
```

Expected: all tests fail with `ModuleNotFoundError: No module named 'app.models.vlm.providers.qwen_vllm.transforms'`.

- [ ] **Step 3: Create `transforms.py` with the minimal implementation**

`backend/app/models/vlm/providers/qwen_vllm/transforms.py`:

```python
"""Pure message transforms for the Qwen vLLM provider.

All functions in this module are pure: they take OpenAI-format message
lists and return new lists. They do not perform I/O, mutate inputs, or
import the OpenAI SDK.
"""
from __future__ import annotations

import re
from copy import deepcopy

from .config import IMAGE_TOKEN_PATTERNS

_COMPILED_TOKEN_PATTERNS = tuple(re.compile(p) for p in IMAGE_TOKEN_PATTERNS)


def strip_image_tokens(messages: list[dict]) -> list[dict]:
    """Return a copy of messages with Qwen image placeholder tokens
    removed from user text content.
    """
    out = deepcopy(messages)
    for msg in out:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = _strip(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = _strip(part.get("text", ""))
    return out


def _strip(text: str) -> str:
    for pat in _COMPILED_TOKEN_PATTERNS:
        text = pat.sub("", text)
    return text
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/vlm/providers/qwen_vllm/test_transforms.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```
git add backend/tests/vlm/providers/qwen_vllm/test_transforms.py backend/app/models/vlm/providers/qwen_vllm/transforms.py
git commit -m "feat(vlm): add strip_image_tokens transform for qwen_vllm"
```

---

## Task 4: TDD `transforms.inject_pixel_bounds`

**Files:**
- Modify: `backend/tests/vlm/providers/qwen_vllm/test_transforms.py`
- Modify: `backend/app/models/vlm/providers/qwen_vllm/transforms.py`

- [ ] **Step 1: Append failing tests to `test_transforms.py`**

Add these to the bottom of `backend/tests/vlm/providers/qwen_vllm/test_transforms.py`. Also update the import line at the top.

Replace the existing top-of-file import line:
```python
from app.models.vlm.providers.qwen_vllm.transforms import strip_image_tokens
```
with:
```python
from app.models.vlm.providers.qwen_vllm.transforms import (
    inject_pixel_bounds,
    strip_image_tokens,
)
```

Append these tests at the end of the file:

```python
def test_inject_pixel_bounds_attaches_bounds_to_image_parts():
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "text", "text": "hi"},
        ],
    }]
    out = inject_pixel_bounds(msgs, 100, 200)
    assert out[0]["content"][0]["image_url"]["min_pixels"] == 100
    assert out[0]["content"][0]["image_url"]["max_pixels"] == 200


def test_inject_pixel_bounds_noop_for_text_only():
    msgs = [{"role": "user", "content": "hello"}]
    out = inject_pixel_bounds(msgs, 100, 200)
    assert out == msgs


def test_inject_pixel_bounds_does_not_mutate_input():
    msgs = [{
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "x"}}],
    }]
    snapshot = copy.deepcopy(msgs)
    inject_pixel_bounds(msgs, 100, 200)
    assert msgs == snapshot
```

- [ ] **Step 2: Run tests — verify new ones fail**

```
pytest tests/vlm/providers/qwen_vllm/test_transforms.py -v
```

Expected: 3 new tests fail with `ImportError: cannot import name 'inject_pixel_bounds'`. 6 existing tests still pass (after import line fix, they all fail on the ImportError — that's also acceptable; the point is the module doesn't export the symbol).

- [ ] **Step 3: Add `inject_pixel_bounds` to `transforms.py`**

Append to `backend/app/models/vlm/providers/qwen_vllm/transforms.py`:

```python
def inject_pixel_bounds(
    messages: list[dict],
    min_pixels: int,
    max_pixels: int,
) -> list[dict]:
    """Return a copy of messages with min_pixels/max_pixels attached to
    every image_url content part. No-op for text-only messages.
    """
    out = deepcopy(messages)
    for msg in out:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    image_url["min_pixels"] = min_pixels
                    image_url["max_pixels"] = max_pixels
    return out
```

- [ ] **Step 4: Run tests — verify all pass**

```
pytest tests/vlm/providers/qwen_vllm/test_transforms.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```
git add backend/tests/vlm/providers/qwen_vllm/test_transforms.py backend/app/models/vlm/providers/qwen_vllm/transforms.py
git commit -m "feat(vlm): add inject_pixel_bounds transform for qwen_vllm"
```

---

## Task 5: TDD `QwenVLLMProvider`

**Files:**
- Create: `backend/tests/vlm/providers/qwen_vllm/test_provider.py`
- Create: `backend/app/models/vlm/providers/qwen_vllm/provider.py`
- Modify: `backend/app/models/vlm/providers/qwen_vllm/__init__.py`

- [ ] **Step 1: Write failing tests for `QwenVLLMProvider`**

Create `backend/tests/vlm/providers/qwen_vllm/test_provider.py`:

```python
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError

from app.models.vlm.providers.qwen_vllm.provider import QwenVLLMProvider


def _make_api_connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "http://fake"))


def _make_response(text: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response


def test_generate_returns_content_on_success():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1",
        api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    )
    with patch.object(
        provider._client.chat.completions,
        "create",
        return_value=_make_response("hello"),
    ) as mock_create:
        result = provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=0,
        )
    assert result == "hello"
    assert mock_create.call_count == 1


def test_generate_retries_once_on_connection_error_then_succeeds():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1",
        api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    )
    with patch.object(
        provider._client.chat.completions,
        "create",
        side_effect=[_make_api_connection_error(), _make_response("ok")],
    ) as mock_create, patch(
        "app.models.vlm.providers.qwen_vllm.provider.time.sleep"
    ):
        result = provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=0,
        )
    assert result == "ok"
    assert mock_create.call_count == 2


def test_generate_raises_connection_error_after_retry_exhaustion():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1",
        api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
    )
    with patch.object(
        provider._client.chat.completions,
        "create",
        side_effect=[
            _make_api_connection_error(),
            _make_api_connection_error(),
        ],
    ), patch("app.models.vlm.providers.qwen_vllm.provider.time.sleep"):
        with pytest.raises(ConnectionError) as excinfo:
            provider.generate(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
                temperature=0,
            )
    assert "Qwen/Qwen3-VL-8B-Instruct" in str(excinfo.value)
    assert "http://fake/v1" in str(excinfo.value)


def test_generate_applies_transforms_before_call():
    provider = QwenVLLMProvider(
        base_url="http://fake/v1",
        api_key="none",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        min_pixels=111,
        max_pixels=222,
    )
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _make_response("ok")

    with patch.object(
        provider._client.chat.completions, "create", side_effect=fake_create
    ):
        provider.generate(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "x"}},
                    {"type": "text", "text": "see <image> this"},
                ],
            }],
            max_tokens=10,
            temperature=0,
        )

    sent = captured["messages"]
    text_part = sent[0]["content"][1]
    img_part = sent[0]["content"][0]
    assert text_part["text"] == "see  this"
    assert img_part["image_url"]["min_pixels"] == 111
    assert img_part["image_url"]["max_pixels"] == 222
```

- [ ] **Step 2: Run tests — verify they fail**

```
pytest tests/vlm/providers/qwen_vllm/test_provider.py -v
```

Expected: all 4 tests fail with `ModuleNotFoundError: No module named 'app.models.vlm.providers.qwen_vllm.provider'`.

- [ ] **Step 3: Create `provider.py`**

`backend/app/models/vlm/providers/qwen_vllm/provider.py`:

```python
"""Qwen vLLM provider.

Wraps the OpenAI SDK client with Qwen-specific input transforms and
a one-retry policy on connection errors.
"""
from __future__ import annotations

import time

from openai import APIConnectionError, OpenAI

from ..base import VLMProvider
from . import config, transforms


class QwenVLLMProvider(VLMProvider):
    """Provider for Qwen-family VL models served via vLLM HTTP."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=config.REQUEST_TIMEOUT_S,
        )
        self._model_id = model_id
        self._base_url = base_url
        self._min_pixels = (
            min_pixels if min_pixels is not None else config.DEFAULT_MIN_PIXELS
        )
        self._max_pixels = (
            max_pixels if max_pixels is not None else config.DEFAULT_MAX_PIXELS
        )

    def generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        messages = transforms.strip_image_tokens(messages)
        messages = transforms.inject_pixel_bounds(
            messages, self._min_pixels, self._max_pixels
        )

        last_exc: APIConnectionError | None = None
        for attempt in range(config.MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                return (content or "").strip()
            except APIConnectionError as exc:
                last_exc = exc
                if attempt < config.MAX_RETRIES:
                    time.sleep(config.RETRY_BACKOFF_S)

        raise ConnectionError(
            f"Cannot connect to model '{self._model_id}' at {self._base_url}. "
            f"Is the vLLM server running? ({last_exc})"
        )
```

- [ ] **Step 4: Update `__init__.py` to export the provider**

Replace `backend/app/models/vlm/providers/qwen_vllm/__init__.py` contents with:

```python
from .provider import QwenVLLMProvider

__all__ = ["QwenVLLMProvider"]
```

- [ ] **Step 5: Run tests — verify all pass**

```
pytest tests/vlm/providers/qwen_vllm/ -v
```

Expected: all 13 tests pass (9 transforms + 4 provider).

- [ ] **Step 6: Commit**

```
git add backend/tests/vlm/providers/qwen_vllm/test_provider.py backend/app/models/vlm/providers/qwen_vllm/provider.py backend/app/models/vlm/providers/qwen_vllm/__init__.py
git commit -m "feat(vlm): add QwenVLLMProvider with retry and transform orchestration"
```

---

## Task 6: Wire provider into manager and swap YAML entry

**Files:**
- Modify: `backend/app/models/vlm/manager.py`
- Modify: `backend/app/models/vlm/models.yaml`

- [ ] **Step 1: Register `QwenVLLMProvider` in `PROVIDER_MAP`**

Edit `backend/app/models/vlm/manager.py`.

Replace the existing import + `PROVIDER_MAP` block:

```python
from .providers.base import VLMProvider
from .providers.openai_compatible import OpenAICompatibleProvider

# Maps provider name in YAML -> provider class
PROVIDER_MAP: dict[str, type[VLMProvider]] = {
    "openai_compatible": OpenAICompatibleProvider,
}
```

with:

```python
from .providers.base import VLMProvider
from .providers.openai_compatible import OpenAICompatibleProvider
from .providers.qwen_vllm import QwenVLLMProvider

# Maps provider name in YAML -> provider class
PROVIDER_MAP: dict[str, type[VLMProvider]] = {
    "openai_compatible": OpenAICompatibleProvider,
    "qwen_vllm": QwenVLLMProvider,
}
```

- [ ] **Step 2: Plumb optional `min_pixels` / `max_pixels` through to the provider constructor**

In `backend/app/models/vlm/manager.py`, replace the existing provider-instantiation block:

```python
            provider_cls = PROVIDER_MAP[provider_name]
            provider = provider_cls(
                base_url=entry["base_url"],
                api_key=api_key,
                model_id=entry["model_id"],
            )
```

with:

```python
            provider_cls = PROVIDER_MAP[provider_name]
            provider_kwargs: dict[str, Any] = {
                "base_url": entry["base_url"],
                "api_key": api_key,
                "model_id": entry["model_id"],
            }
            if provider_name == "qwen_vllm":
                if "min_pixels" in entry:
                    provider_kwargs["min_pixels"] = entry["min_pixels"]
                if "max_pixels" in entry:
                    provider_kwargs["max_pixels"] = entry["max_pixels"]
            provider = provider_cls(**provider_kwargs)
```

- [ ] **Step 3: Update `models.yaml` — remove LLaVA, add Qwen3-VL-8B**

Edit `backend/app/models/vlm/models.yaml`.

Delete the entire `llava-1.5-7b-local` block (lines 13–22 currently):

```yaml
  - id: "llava-1.5-7b-local"         # Unique identifier — sent by frontend to select this model
    name: "LLaVA 1.5 7B (Local)"     # Display name shown in the frontend dropdown
    provider: "openai_compatible"     # Provider class to use (maps to providers/ module)
    base_url: "http://localhost:8002/v1"  # API endpoint for this provider
    model_id: "llava-hf/llava-1.5-7b-hf"  # Model name sent in API requests
    api_key_env: null                 # .env variable holding the API key (null = no auth, uses "none")
    system_prompt: >                  # System message prepended to every conversation
      You are a helpful AI assistant!
    max_tokens: 256                   # Maximum tokens for model generation
    temperature: 0                    # Sampling temperature (0 = deterministic)
```

Replace that block with:

```yaml
  - id: "qwen3-vl-8b-vllm"           # Unique identifier — sent by frontend to select this model
    name: "Qwen3-VL 8B (vLLM)"       # Display name shown in the frontend dropdown
    provider: "qwen_vllm"             # Provider class to use (maps to providers/ module)
    base_url: "http://localhost:8003/v1"  # SSH-tunneled endpoint to remote vLLM server
    model_id: "Qwen/Qwen3-VL-8B-Instruct"  # Model name sent in API requests
    api_key_env: null                 # .env variable holding the API key (null = no auth, uses "none")
    system_prompt: >                  # System message prepended to every conversation
      You are a helpful AI assistant!
    max_tokens: 512                   # Maximum tokens for model generation
    temperature: 0                    # Sampling temperature (0 = deterministic)
    min_pixels: 200704                # Qwen processor lower bound (optional — falls back to config.py)
    max_pixels: 1605632               # Qwen processor upper bound (optional — falls back to config.py)
```

- [ ] **Step 4: Verify the manager loads without errors**

From `backend/`:
```
python -c "from app.models.vlm import VLMManager; m = VLMManager(); m.load(); print(m.list_models())"
```

Expected output (order may vary):
```
[VLMManager] Loaded 3 model(s): ['qwen-vl-local', 'qwen3-vl-8b-vllm', 'gpt-5.4-mini']
[{'id': 'qwen-vl-local', 'name': 'Qwen 2.5 VL 3B (Local)'}, {'id': 'qwen3-vl-8b-vllm', 'name': 'Qwen3-VL 8B (vLLM)'}, {'id': 'gpt-5.4-mini', 'name': 'GPT-5.4 Mini (OpenAI)'}]
```

If the first line warns `Unknown provider 'qwen_vllm'`, re-check Step 1.

- [ ] **Step 5: Run the full test suite**

```
pytest -v
```

Expected: all 13 tests pass.

- [ ] **Step 6: Commit**

```
git add backend/app/models/vlm/manager.py backend/app/models/vlm/models.yaml
git commit -m "feat(vlm): register qwen_vllm provider and swap LLaVA for Qwen3-VL-8B"
```

---

## Task 7: Smoke test script

**Files:**
- Create: `backend/scripts/smoke_qwen3_vl.py`

No TDD — this is a manual operational script, not library code.

- [ ] **Step 1: Create the smoke script**

`backend/scripts/smoke_qwen3_vl.py`:

```python
"""Manual smoke test for the Qwen3-VL-8B model.

Run after `vllm serve` is up on the remote host (tunneled to :8003) and
the backend is running on :8000.

Usage:
    cd backend
    python scripts/smoke_qwen3_vl.py path/to/image.jpg "What is in this image?"
"""
from __future__ import annotations

import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_URL = "http://localhost:8000/api/chat"
MODEL_ID = "qwen3-vl-8b-vllm"
ERROR_PREFIXES = ("Lỗi kết nối", "Xin lỗi")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} <image-path> <question>", file=sys.stderr)
        return 1

    image_path = Path(argv[1])
    question = argv[2]

    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode()
    data_url = f"data:{mime};base64,{data}"

    payload = {
        "message": question,
        "history": [],
        "image_urls": [data_url],
        "model_id": MODEL_ID,
    }

    req = urllib.request.Request(
        BACKEND_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    reply = body.get("reply", "")
    print(f"Reply: {reply}")

    if not reply:
        print("FAIL: empty reply", file=sys.stderr)
        return 1
    for prefix in ERROR_PREFIXES:
        if reply.startswith(prefix):
            print(f"FAIL: reply starts with error prefix '{prefix}'", file=sys.stderr)
            return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 2: Verify the script compiles (syntax check only; no live server needed)**

```
python -m py_compile backend/scripts/smoke_qwen3_vl.py
```

Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```
git add backend/scripts/smoke_qwen3_vl.py
git commit -m "feat(vlm): add manual smoke script for qwen3-vl-8b"
```

---

## Task 8: Write `qwen_development.md`

**Files:**
- Create: `qwen_development.md` (repo root)

- [ ] **Step 1: Create `qwen_development.md` at repo root**

`qwen_development.md`:

````markdown
# Qwen3-VL-8B — Operational Notes & Deferred Work

Companion to `qwen3_vl_design.md`. Covers vast.ai operations, deferred quirks, future phases, and manual verification.

## Operational Notes

### vLLM start command (on vast.ai GPU host)

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8003 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --limit-mm-per-prompt image=4
```

### SSH tunnel (from local dev machine)

Extend the pattern in `vast_ai.txt`:

```bash
ssh -L 5173:localhost:5173 \
    -L 8000:localhost:8000 \
    -L 8003:localhost:8003 \
    -p <SSH_PORT> root@<VAST_IP>
```

### Port allocation

- `8001` — Qwen 2.5-VL 3B (existing)
- `8002` — reserved (freed by LLaVA removal)
- `8003` — Qwen3-VL 8B (this change)

## Deferred Quirks

Numbering matches the brainstorming-scope question. Each item maps back to a gotcha in `development_roadmap.md` Phase 1.

### (iii) Stop-token sanitization

Qwen uses `\n\n` as a stop token, which can truncate mid-paragraph. Add a stop-sanitizer to `transforms.py` if/when `models.yaml` starts passing a `stop` knob through to the provider. Not needed today.

### (v) Video input with `max_num_frames` cap

vLLM can accept video; the Qwen processor needs explicit frame sampling. Add when the frontend begins uploading video (`image_urls` generalizes to `media_urls`). Wire a YAML-driven `max_num_frames` into `config.py` and a new transform.

### (vi) Per-request system prompt override

Today `manager.py` prepends the YAML `system_prompt` on every request. If the playground UI adds a system-prompt field, route it through the `/api/chat` body and let it override the YAML value per request.

## Future Phases (from `development_roadmap.md` Phase 3)

- Streaming (`stream=true`) — vLLM supports it; the backend router + frontend need to handle Server-Sent Events.
- Multi-turn history trimming to a token budget per model.
- Per-request metrics on `/metrics` (TTFT, tokens/sec).
- Bearer-token auth on the vLLM endpoint.

## Manual Verification Checklist

After deploy:

- [ ] `/api/models` returns the `qwen3-vl-8b-vllm` entry.
- [ ] Text-only chat via `/playground` returns a non-empty reply.
- [ ] Text + image chat via `/playground` returns a reply that references the image.
- [ ] Kill the vLLM server mid-request → frontend shows "Lỗi kết nối: …" (not a 500).
- [ ] `python backend/scripts/smoke_qwen3_vl.py <image> "<question>"` exits 0.

## Rationale Pointers

| Quirk | Handled in | Source |
|---|---|---|
| Image-token leakage (i) | `transforms.strip_image_tokens` | `development_roadmap.md` Phase 1 gotcha 2 |
| `min_pixels` / `max_pixels` (ii) | `transforms.inject_pixel_bounds`, `config.DEFAULT_*` | `development_roadmap.md` Phase 1 gotcha 3 |
| Connection retry / timeout (iv) | `provider.QwenVLLMProvider`, `config.REQUEST_TIMEOUT_S` | `development_roadmap.md` Phase 1 gotcha 9 (partial) |
````

- [ ] **Step 2: Commit**

```
git add qwen_development.md
git commit -m "docs: add qwen_development.md with ops notes and deferred work"
```

---

## Final Verification

- [ ] **Step 1: Run the full test suite**

```
cd backend && pytest -v
```

Expected: 13 tests pass (9 transforms + 4 provider).

- [ ] **Step 2: Start the backend locally and hit `/api/models`**

```
cd backend && uvicorn app.main:app --reload --port 8000
```

In another terminal:
```
curl http://localhost:8000/api/models
```

Expected JSON contains entries with `"id": "qwen3-vl-8b-vllm"`, `"id": "qwen-vl-local"`, `"id": "gpt-5.4-mini"`. No LLaVA entry.

- [ ] **Step 3: Stop the backend (no live vLLM needed for Step 2).**

Live end-to-end verification (with vLLM server up on vast.ai + SSH tunnel) is covered by the manual checklist in `qwen_development.md`.
