# Phase 5 — Image-aware history compression + thinking log Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `open-webui/vast-templates/qwen3-vl-8b/functions/qwenvl_image_compress.py` into our backend as a service module, then surface its progress and reasoning in the frontend. After this phase, conversations can have arbitrarily many images: the compressor captions every image once (cached by SHA-256), then rewrites history so only the relevant turn carries pixels, and the user sees a live status banner + a collapsible "thinking log" `<details>` block alongside each assistant reply.

**Architecture:** New `backend/app/services/image_compressor/` package. `CaptionCache` (aiosqlite WAL) keyed by image-bytes SHA-256, with `INSERT OR IGNORE` to make concurrent puts safe. `messages.py` holds the pure helpers (`iter_image_parts`, `find_latest_image_turn`, `text_of`, `has_images`, `hash_image_url`, `rewrite_messages`). `engine.py` orchestrates: scan history → cache lookup → caption misses in parallel via `manager.generate_raw()` → if the latest turn is text-only and earlier turns hold images, run a router LLM call → strip pixel bytes from non-`keep_idx` turns and replace with `[Past image #N: <caption>]`. `compress()` is an async generator that yields `StatusEvent`s while it works and one terminal `CompressionResult` (or a passthrough on any internal exception). `chat_stream` consumes the generator: status events become `data: {"type":"status",...}` SSE frames, the result's `thinking_md` is prepended to the model's first delta as a `<details>` block. Frontend gains `rehype-raw` so the block renders, an optional `onStatus` callback on `useChatStream`, a transient `StatusBanner` pill, and a simpler per-turn `checkAttachmentCap`. The dumb Phase 1 `enforce_image_cap(max=4)` stays as a defensive backstop in case the compressor itself crashes.

**Tech Stack:** Python 3.11+ FastAPI + aiosqlite (NEW dep). TypeScript + React 18 + `rehype-raw@^7` (NEW dep). Existing test infra: pytest + pytest-asyncio, Vitest + jsdom, Playwright (Chromium).

**Spec:** `docs/superpowers/specs/2026-05-09-playground-qwen-vl-parity-design.md`. Phase 5 sections: B.7 (backend, line ~415), C.11 (frontend, line ~755), Phase 5 subsection of Phase Breakdown (line ~966).

**Phase 4 plan (sibling):** `docs/superpowers/plans/2026-05-11-playground-qwen-vl-parity-phase-4.md`. Phase 4 finalised the persistence layer this phase reuses (no schema changes — Phase 5 doesn't persist a new field; thinking log lives inside `msg.text`, status events are ephemeral).

**Reference filter:** `open-webui/vast-templates/qwen3-vl-8b/functions/qwenvl_image_compress.py`. Read once for context, but **do not** copy verbatim — our integration uses `manager.generate_raw()` (we don't ship the filter plugin system) and our SSE protocol (we don't have Open WebUI's `__event_emitter__`).

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| **Task 1 — Bootstrap (deps + skeleton)** |||
| Modify | `backend/requirements.txt` | Add `aiosqlite` |
| Modify | `.gitignore` | Add `backend/data/` |
| Modify | `backend/app/models/vlm/models.yaml` | Add top-level `compressor:` block |
| Create | `backend/app/services/image_compressor/__init__.py` | Re-exports `ImageCompressorEngine` (placeholder for Task 6) |
| Create | `backend/app/services/image_compressor/types.py` | `Scanned`, `StatusEvent`, `CompressionResult`, `CompressionEvent` |
| Create | `backend/app/services/image_compressor/prompts.py` | `CAPTION_SYSTEM_PROMPT`, `CAPTION_USER_TEXT`, `ROUTER_SYSTEM_PROMPT`, `ROUTER_USER_TEMPLATE` |
| **Task 2 — `VLMManager.generate_raw`** |||
| Modify | `backend/app/models/vlm/manager.py` | Add `generate_raw(model_id, messages, *, max_tokens, temperature)` that streams via provider WITHOUT prepending the model's system prompt |
| Create | `backend/tests/test_manager_generate_raw.py` | Test: generate_raw skips system prompt; honors max_tokens/temperature override |
| **Task 3 — `CaptionCache`** |||
| Create | `backend/app/services/image_compressor/cache.py` | `CaptionCache` (aiosqlite WAL, sha256-keyed); `init`, `get`, `get_many`, `put`, `put_many` |
| Create | `backend/tests/test_image_compressor_cache.py` | T5.B1 round-trip + get_many partial + put_many INSERT OR IGNORE; A5.B3 concurrent put |
| **Task 4 — `messages.py` helpers** |||
| Create | `backend/app/services/image_compressor/messages.py` | `has_images`, `iter_image_parts`, `find_latest_image_turn`, `text_of`, `hash_image_url`, `rewrite_messages` |
| Create | `backend/tests/test_image_compressor_messages.py` | T5.B2 iter; T5.B3 rewrite; T5.B4 hash; A5.B4 hash relative path |
| **Task 5 — Engine units** |||
| Create | `backend/app/services/image_compressor/engine.py` (partial) | `ImageCompressorEngine.__init__` + `caption_one`, `route`, `ensure_captions` |
| Create | `backend/tests/test_image_compressor_engine_units.py` | T5.B5 caption; T5.B6 route happy + JSON error fail-open; A5.B1 caption error fail-open; A5.B6 dedup-by-hash |
| **Task 6 — Engine `compress()` orchestrator** |||
| Modify | `backend/app/services/image_compressor/engine.py` | Add `compress()` async generator (with self-catching outer try/except → passthrough) |
| Create | `backend/tests/test_image_compressor_engine_compress.py` | T5.B7 compress with images; A5.B5 engine self-catch passthrough |
| **Task 7 — Lifespan + chat_stream wiring** |||
| Modify | `backend/app/models/vlm/manager.py` | Add `compressor_config()` method (reads `compressor:` block, validates ids exist) |
| Modify | `backend/app/main.py` | Lifespan instantiates `CaptionCache` + `ImageCompressorEngine`; stores on `app.state.compressor_engine` |
| Modify | `backend/app/routers/chat.py` | Add `_sse_status` helper; thread engine into `chat_stream` (status events + thinking_md prefix + safety net behind it) |
| Modify | `backend/app/services/messages.py` | Update `enforce_image_cap` doc-comment to reflect Phase 5 fallback role (no behavior change) |
| Create | `backend/tests/test_chat_stream_compressor.py` | T5.B8 with-images integration; T5.B9 no-images fast path; A5.11 compressor disabled (no yaml block) |
| **Task 8 — Frontend SSE + chatStream** |||
| Modify | `frontend/src/playground/lib/sseParser.ts` | Recognize `{type:"status",message,done}` → yield `{type:"status",message,statusDone}` |
| Modify | `frontend/src/playground/lib/sseParser.test.ts` | T5.1 + T5.2 + T5.3 status event parsing + forward-compat |
| Modify | `frontend/src/playground/lib/chatStream.ts` | Accept `onStatus?(message, done)` in callbacks; route status events |
| Modify | `frontend/src/playground/lib/chatStream.test.ts` | T5.4 onStatus invocation |
| **Task 9 — StatusBanner + PlaygroundPage + cap relax + rehype-raw** |||
| Modify | `frontend/package.json` | Add `rehype-raw@^7` |
| Create | `frontend/src/playground/components/StatusBanner.tsx` | Pill component; auto-clears 1500ms after `done=true` |
| Create | `frontend/src/playground/components/StatusBanner.test.tsx` | T5.5 render + auto-dismiss with fake timers |
| Modify | `frontend/src/playground/lib/fileValidate.ts` | Simplify `checkAttachmentCap(currentCount)` (drop history param) |
| Modify | `frontend/src/playground/lib/fileValidate.test.ts` | Update T2.10 to new signature |
| Modify | `frontend/src/playground/components/MessageBubble.tsx` | Add `rehypeRaw` to plugin chain (after `rehypeHighlight`) |
| Modify | `frontend/src/playground/components/ComposerBar.tsx` | Drop `historyImageCount` prop (no longer needed) |
| Modify | `frontend/src/playground/hooks/useChatStream.ts` | Forward optional `onStatus` from `SendArgs` to `streamChat` |
| Modify | `frontend/src/pages/PlaygroundPage.tsx` | `useState<StatusBannerState\|null>` + `useRef<number\|null>` for timeout; `onStatus` handler; render `<StatusBanner status={...}/>` above MessageList; remove `historyImageCount` plumbing |
| **Task 10 — Playwright E2E** |||
| Modify | `frontend/tests/e2e/fixtures/sseFixture.ts` | Add `mockChatStreamWithStatus(page, statuses, deltas)` |
| Create | `frontend/tests/e2e/playground-compressor.spec.ts` | E5.1 status banner + E5.2 thinking-log `<details>` |
| **Task 11 — Manual smoke + final pass** |||
| — | — | Run all suites; manual A5.4–A5.7, A5.10 walkthroughs |

---

## Tasks

### Task 1: Bootstrap (deps + package skeleton + yaml + types + prompts)

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `.gitignore`
- Modify: `backend/app/models/vlm/models.yaml`
- Create: `backend/app/services/image_compressor/__init__.py`
- Create: `backend/app/services/image_compressor/types.py`
- Create: `backend/app/services/image_compressor/prompts.py`

This task ships pure scaffolding — no logic, no tests. Subsequent tasks depend on the types and prompts existing.

- [ ] **Step 1: Add `aiosqlite` to backend deps**

Add a single line to `backend/requirements.txt`:

```
aiosqlite
```

The full file should now read:

```
fastapi
uvicorn[standard]
pydantic
pydantic-settings
pillow
openai
pyyaml
pytest
pytest-asyncio
python-multipart
aiosqlite
```

- [ ] **Step 2: Install the new dep**

Run:
```
cd backend
. .venv/bin/activate    # (or `.\.venv\Scripts\Activate.ps1` on PowerShell)
pip install aiosqlite
```

Expected: aiosqlite-X.Y.Z installed without errors.

- [ ] **Step 3: Add `backend/data/` to `.gitignore`**

Append to `.gitignore` under the `# Python` section, after `backend/images/`:

```
backend/data/
```

- [ ] **Step 4: Add the `compressor:` block to `models.yaml`**

Append the following to the END of `backend/app/models/vlm/models.yaml`:

```yaml

compressor:
  caption_model_id: "qwen3-vl-8b-vllm"
  router_model_id: "qwen3-vl-8b-vllm"
  cache_db_path: "backend/data/img_captions.db"
  caption_max_tokens: 80
  router_max_tokens: 60
  caption_timeout_s: 30
  router_timeout_s: 15
  router_failopen_keep: true
  webui_internal_base: "http://127.0.0.1:8000"
```

- [ ] **Step 5: Create the package `__init__.py`**

Write `backend/app/services/image_compressor/__init__.py` with a deferred re-export (Task 6 will land the engine class — for now this just makes the package importable):

```python
"""Image-aware history compressor: ports open-webui's qwenvl_image_compress
filter into our backend's service layer. The full surface area (CaptionCache,
ImageCompressorEngine) is built across Tasks 3-6 of the Phase 5 plan.
"""
```

- [ ] **Step 6: Create `types.py`**

Write `backend/app/services/image_compressor/types.py`:

```python
"""Type aliases and dataclasses for the image-aware compressor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

# (msg_idx, content_idx, url, sha256_hex, raw_bytes)
Scanned = tuple[int, int, str, str, bytes]


@dataclass(frozen=True)
class StatusEvent:
    message: str
    done: bool = False


@dataclass(frozen=True)
class CompressionResult:
    messages: list[dict]
    thinking_md: str


CompressionEvent = Union[StatusEvent, CompressionResult]
```

- [ ] **Step 7: Create `prompts.py`**

Write `backend/app/services/image_compressor/prompts.py`. Verbatim from reference filter (Vietnamese — keep exactly):

```python
"""Vietnamese system prompts for caption + router LLM calls.

Sourced from open-webui/vast-templates/qwen3-vl-8b/functions/qwenvl_image_compress.py.
Kept verbatim so the routing/captioning behaviour matches the reference filter.
"""

CAPTION_SYSTEM_PROMPT = """\
Bạn là image captioner. Mô tả ảnh trong 1-2 câu khách quan, không quá 60 từ.
Cần nêu:
  - Chủ thể chính (người/vật/cảnh).
  - Văn bản nhìn thấy trong ảnh, copy nguyên văn nếu ngắn.
  - Bố cục/màu sắc nổi bật nếu liên quan.
KHÔNG suy diễn cảm xúc, KHÔNG khen chê, KHÔNG bịa chi tiết.
Trả về DUY NHẤT phần caption, không prefix \"Caption:\" hay markdown."""

CAPTION_USER_TEXT = "Mô tả ảnh này."

ROUTER_SYSTEM_PROMPT = """\
Bạn là router cho 1 hệ thống chat đa phương thức.
Cho 1 câu hỏi text-only của user và mô tả các ảnh user đã upload trước đó,
quyết định xem có cần gửi PIXEL của các ảnh đó cho LLM trả lời không.

Trả LLM cần nhìn pixel khi:
  - Câu hỏi tham chiếu trực tiếp ảnh: \"ảnh đó\", \"cái này\", \"hình thứ N\", \"trên màn hình\".
  - Câu hỏi đòi visual detail: màu, vị trí, đếm, OCR chính xác, so sánh ảnh.
  - Câu hỏi tiếp tục chủ đề liên quan đến nội dung ảnh.

Trả LLM KHÔNG cần pixel khi:
  - Câu hỏi đổi sang chủ đề mới không liên quan ảnh.
  - Câu hỏi tổng quát không có đại từ chỉ ảnh và caption đã đủ context.

Output DUY NHẤT 1 JSON object, không markdown:
  {\"need_images\": true|false, \"reason\": \"<1 câu ngắn tiếng Việt>\"}"""

ROUTER_USER_TEMPLATE = (
    "Ảnh đã upload trước đó (theo thứ tự):\n"
    "{captions_block}\n\n"
    "Câu hỏi mới của user:\n"
    "\"\"\"\n{user_text}\n\"\"\"\n"
)
```

- [ ] **Step 8: Verify package imports**

Run a smoke import (no test file yet — just verify nothing is broken):

```
cd backend
python -c "from app.services.image_compressor import types, prompts; print(types.Scanned, prompts.CAPTION_USER_TEXT)"
```

Expected output: `tuple[int, int, str, str, bytes] Mô tả ảnh này.`

- [ ] **Step 9: Commit**

```
git add backend/requirements.txt .gitignore backend/app/models/vlm/models.yaml backend/app/services/image_compressor/__init__.py backend/app/services/image_compressor/types.py backend/app/services/image_compressor/prompts.py
git commit -m "feat(backend): scaffold image_compressor package + add aiosqlite"
```

---

### Task 2: `VLMManager.generate_raw()`

**Files:**
- Modify: `backend/app/models/vlm/manager.py`
- Create: `backend/tests/test_manager_generate_raw.py`

**Why:** The compressor's caption + router calls have their own system prompts (in `prompts.py`). Calling `manager.generate(model_id, messages)` would prepend the model's generic `system_prompt` ("You are a helpful AI assistant!"), which dilutes the captioner's strict format. We need a path that bypasses `_prepare_messages` AND lets the caller override `max_tokens` / `temperature` (different defaults for caption vs router).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_manager_generate_raw.py`:

```python
"""Test VLMManager.generate_raw — Phase 5 entry point for compressor calls."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from tests._helpers import make_async_stream_mock


@pytest.fixture
def real_manager():
    """A real VLMManager with one Qwen-VL provider mocked at the SDK boundary.
    We don't load YAML here; we hand-build a minimal config.
    """
    from app.models.vlm.manager import VLMManager
    from app.models.vlm.providers.openai_compatible import OpenAICompatibleProvider

    m = VLMManager()
    m.models["fake-model"] = {
        "id": "fake-model",
        "name": "Fake Model",
        "model_id": "fake-model",
        "system_prompt": "You are SYS.",
        "max_tokens": 256,
        "temperature": 0,
    }
    m.providers["fake-model"] = OpenAICompatibleProvider(
        base_url="http://fake/v1", api_key="none", model_id="fake-model",
    )
    m.default_model = "fake-model"
    return m


@pytest.mark.asyncio
async def test_generate_raw_skips_system_prompt(real_manager):
    """generate_raw must NOT prepend the model's system_prompt; caller's
    messages list is sent through verbatim."""
    sent_messages: list[list[dict]] = []

    def capture(*args, **kwargs):
        sent_messages.append(list(kwargs["messages"]))
        return make_async_stream_mock(["ok"])(*args, **kwargs)

    provider = real_manager.providers["fake-model"]
    with patch.object(provider.client.chat.completions, "create", side_effect=capture):
        result = await real_manager.generate_raw(
            "fake-model",
            [{"role": "user", "content": "hi"}],
            max_tokens=42, temperature=0.7,
        )

    assert result == "ok"
    assert len(sent_messages) == 1
    assert sent_messages[0] == [{"role": "user", "content": "hi"}]
    # NOTE: no {"role":"system","content":"You are SYS."} prepended.


@pytest.mark.asyncio
async def test_generate_raw_honors_max_tokens_override(real_manager):
    """Caller-supplied max_tokens reaches the SDK call (not the yaml default)."""
    captured_kwargs: list[dict] = []

    def capture(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return make_async_stream_mock(["ok"])(*args, **kwargs)

    provider = real_manager.providers["fake-model"]
    with patch.object(provider.client.chat.completions, "create", side_effect=capture):
        await real_manager.generate_raw(
            "fake-model",
            [{"role": "user", "content": "hi"}],
            max_tokens=42, temperature=0.7,
        )

    # The SDK accepts either max_completion_tokens (modern) or max_tokens
    # (legacy fallback path). For the happy first call, our provider uses
    # max_completion_tokens=42 + temperature=0.7.
    assert captured_kwargs[0]["max_completion_tokens"] == 42
    assert captured_kwargs[0]["temperature"] == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run from `backend/`:
```
pytest tests/test_manager_generate_raw.py -v
```

Expected: FAIL with `AttributeError: 'VLMManager' object has no attribute 'generate_raw'`.

- [ ] **Step 3: Implement `generate_raw`**

In `backend/app/models/vlm/manager.py`, append two new methods AFTER the existing `generate()` method (after line 104):

```python
    async def stream_raw(
        self,
        model_id: str | None,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Stream tokens from the provider WITHOUT prepending the model's
        per-yaml system prompt. The caller supplies its own messages list
        verbatim. Used by the image compressor (caption + router calls have
        their own system prompts).
        """
        provider, _ = self._resolve(model_id)
        async for delta in provider.stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield delta

    async def generate_raw(
        self,
        model_id: str | None,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Non-streaming wrapper around `stream_raw`. Returns the joined+
        stripped text for the call."""
        chunks: list[str] = []
        async for delta in self.stream_raw(
            model_id, messages,
            max_tokens=max_tokens, temperature=temperature,
        ):
            chunks.append(delta)
        return "".join(chunks).strip()
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_manager_generate_raw.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Run the full backend suite to confirm no regressions**

```
pytest -q
```

Expected: 67 PASSED (65 baseline + 2 new).

- [ ] **Step 6: Commit**

```
git add backend/app/models/vlm/manager.py backend/tests/test_manager_generate_raw.py
git commit -m "feat(backend): add VLMManager.generate_raw (system-prompt-bypass for compressor)"
```

---

### Task 3: `CaptionCache` (aiosqlite WAL)

**Files:**
- Create: `backend/app/services/image_compressor/cache.py`
- Create: `backend/tests/test_image_compressor_cache.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_image_compressor_cache.py`:

```python
"""Unit tests for CaptionCache (Phase 5)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.services.image_compressor.cache import CaptionCache


@pytest.fixture
def cache_path(tmp_path: Path) -> str:
    return str(tmp_path / "captions.db")


@pytest.mark.asyncio
async def test_T5B1_round_trip(cache_path: str) -> None:
    """T5.B1: put → get round-trips. get_many partial hits return only hits.
    put_many with duplicate hash leaves first writer's caption (INSERT OR IGNORE).
    """
    cache = CaptionCache(cache_path)
    await cache.init()

    await cache.put("h1", "First caption.", model="m", bytes_size=100, user_id="u")
    assert await cache.get("h1") == "First caption."
    assert await cache.get("missing") is None

    # get_many: 2 hits, 1 miss
    await cache.put("h2", "Second caption.", model="m")
    out = await cache.get_many(["h1", "h2", "h3"])
    assert out == {"h1": "First caption.", "h2": "Second caption."}

    # put_many duplicate: first writer wins
    await cache.put_many([
        ("h1", "OVERWRITE attempt.", "m", None, None),  # collision with h1
        ("h4", "Fourth.", "m", None, None),             # new
    ])
    assert await cache.get("h1") == "First caption."  # unchanged
    assert await cache.get("h4") == "Fourth."


@pytest.mark.asyncio
async def test_A5B3_concurrent_put(cache_path: str) -> None:
    """A5.B3: 5 concurrent put_many calls with the SAME hash → no
    IntegrityError; cache has exactly 1 row for that hash."""
    cache = CaptionCache(cache_path)
    await cache.init()

    async def writer(i: int) -> None:
        await cache.put_many([
            ("same_hash", f"caption from writer {i}.", "m", None, None),
        ])

    await asyncio.gather(*(writer(i) for i in range(5)))

    # Exactly one row, one of the 5 captions (whichever won the INSERT race).
    cached = await cache.get("same_hash")
    assert cached is not None
    assert cached.startswith("caption from writer ")


@pytest.mark.asyncio
async def test_init_idempotent(cache_path: str) -> None:
    """Calling init() twice on the same path is safe (CREATE IF NOT EXISTS)."""
    cache_a = CaptionCache(cache_path)
    cache_b = CaptionCache(cache_path)
    await cache_a.init()
    await cache_b.init()
    await cache_a.put("h1", "x", model="m")
    assert await cache_b.get("h1") == "x"
```

- [ ] **Step 2: Run to verify failure**

```
cd backend
pytest tests/test_image_compressor_cache.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.image_compressor.cache'`.

- [ ] **Step 3: Implement `cache.py`**

Create `backend/app/services/image_compressor/cache.py`:

```python
"""Aiosqlite-backed cache for image captions, keyed by SHA-256 of image bytes.

WAL mode + INSERT OR IGNORE makes concurrent writes safe across asyncio tasks.
Mirrors the schema used by the open-webui reference filter so the same DB file
could be reused if/when we run both side-by-side.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS captions (
    img_hash    TEXT PRIMARY KEY,
    caption     TEXT NOT NULL,
    model       TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    bytes_size  INTEGER,
    user_id     TEXT
);
CREATE INDEX IF NOT EXISTS idx_created ON captions(created_at);
"""


class CaptionCache:
    """Async SQLite cache. One DB file shared across all callers."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def init(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            async with aiosqlite.connect(self.path) as db:
                await db.execute("PRAGMA journal_mode = WAL")
                await db.execute("PRAGMA synchronous = NORMAL")
                await db.execute("PRAGMA busy_timeout = 5000")
                await db.executescript(SCHEMA)
                await db.commit()
            self._initialized = True

    async def get(self, h: str) -> Optional[str]:
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                "SELECT caption FROM captions WHERE img_hash = ?", (h,),
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else None

    async def get_many(self, hashes: list[str]) -> dict[str, str]:
        if not hashes:
            return {}
        placeholders = ",".join("?" * len(hashes))
        sql = f"SELECT img_hash, caption FROM captions WHERE img_hash IN ({placeholders})"
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(sql, hashes) as cur:
                return {h: c async for h, c in cur}

    async def put(
        self, h: str, caption: str, *, model: str,
        bytes_size: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> None:
        now_ms = int(time.time() * 1000)
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO captions"
                "(img_hash, caption, model, created_at, bytes_size, user_id)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (h, caption, model, now_ms, bytes_size, user_id),
            )
            await db.commit()

    async def put_many(
        self,
        items: list[tuple[str, str, str, Optional[int], Optional[str]]],
    ) -> None:
        """items: list of (hash, caption, model, bytes_size, user_id)."""
        if not items:
            return
        now_ms = int(time.time() * 1000)
        rows = [(h, c, m, now_ms, sz, uid) for h, c, m, sz, uid in items]
        async with aiosqlite.connect(self.path) as db:
            await db.executemany(
                "INSERT OR IGNORE INTO captions"
                "(img_hash, caption, model, created_at, bytes_size, user_id)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            await db.commit()
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/test_image_compressor_cache.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Run full backend suite**

```
pytest -q
```

Expected: 70 PASSED (67 + 3 new).

- [ ] **Step 6: Commit**

```
git add backend/app/services/image_compressor/cache.py backend/tests/test_image_compressor_cache.py
git commit -m "feat(backend): add CaptionCache (aiosqlite WAL, sha256-keyed)"
```

---

### Task 4: `messages.py` helpers

**Files:**
- Create: `backend/app/services/image_compressor/messages.py`
- Create: `backend/tests/test_image_compressor_messages.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_image_compressor_messages.py`:

```python
"""Unit tests for image_compressor.messages helpers (Phase 5)."""
from __future__ import annotations

import base64
import hashlib

import httpx
import pytest

from app.services.image_compressor.messages import (
    find_latest_image_turn,
    has_images,
    hash_image_url,
    iter_image_parts,
    rewrite_messages,
    text_of,
)


def _png_data_url() -> tuple[str, bytes]:
    """A 1x1 transparent PNG as data URL + its raw bytes."""
    raw = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452"
        "0000000100000001080600000001f15c"
        "4890000000d49444154789c63fc0f0000"
        "0301010029a4c20000000049454e44ae426082"
    )
    b64 = base64.b64encode(raw).decode()
    return f"data:image/png;base64,{b64}", raw


# ---- T5.B2 iter / has_images / find_latest -----------------------------------

def test_T5B2_iter_image_parts_empty() -> None:
    assert list(iter_image_parts([])) == []


def test_T5B2_iter_image_parts_str_content_msg() -> None:
    msgs = [{"role": "user", "content": "hello"}]
    assert list(iter_image_parts(msgs)) == []


def test_T5B2_iter_image_parts_mixed() -> None:
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "u1"}},
            {"type": "image_url", "image_url": {"url": "u2"}},
        ]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "u3"}},
        ]},
    ]
    assert list(iter_image_parts(msgs)) == [
        (0, 1, "u1"), (0, 2, "u2"), (2, 0, "u3"),
    ]


def test_has_images() -> None:
    assert has_images({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "u1"}},
    ]})
    assert not has_images({"role": "user", "content": "hello"})
    assert not has_images({"role": "user", "content": [
        {"type": "text", "text": "hi"},
    ]})


def test_find_latest_image_turn() -> None:
    msgs = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u1"}}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "no images here"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u2"}}]},
        {"role": "user", "content": "also text only"},
    ]
    assert find_latest_image_turn(msgs) == 3
    assert find_latest_image_turn([]) is None
    assert find_latest_image_turn([{"role": "user", "content": "hi"}]) is None


def test_text_of() -> None:
    assert text_of({"role": "user", "content": "hi"}) == "hi"
    assert text_of({"role": "user", "content": [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "u"}},
    ]}) == "hello"
    assert text_of({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "u"}},
    ]}) == ""


# ---- T5.B3 rewrite_messages --------------------------------------------------

def test_T5B3_rewrite_keeps_only_keep_idx() -> None:
    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uA"}},
            {"type": "text", "text": "first"},
        ]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uB"}},
        ]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "uC"}},
            {"type": "text", "text": "third"},
        ]},
    ]
    captions = {"uA": "alpha caption", "uB": "beta caption", "uC": "gamma caption"}

    out = rewrite_messages(msgs, keep_idx=2, captions_by_url=captions)

    # Msg 0 lost its image; text coalesced with [Past image #1: ...]
    assert out[0]["content"] == [
        {"type": "text", "text": "first\n[Past image #1: alpha caption]"},
    ]
    # Msg 1 was image-only; collapses to a string content with placeholder.
    assert out[1]["content"] == [
        {"type": "text", "text": "[Past image #1: beta caption]"},
    ]
    # Msg 2 (keep_idx) untouched.
    assert out[2]["content"] == msgs[2]["content"]


def test_T5B3_rewrite_drop_all_when_keep_idx_none() -> None:
    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "u1"}},
            {"type": "text", "text": "ask"},
        ]},
    ]
    out = rewrite_messages(msgs, keep_idx=None, captions_by_url={"u1": "the cat"})
    assert out[0]["content"] == [
        {"type": "text", "text": "ask\n[Past image #1: the cat]"},
    ]


def test_T5B3_rewrite_does_not_mutate_input() -> None:
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "u"}},
    ]}]
    snapshot = [dict(m) for m in msgs]
    rewrite_messages(msgs, keep_idx=None, captions_by_url={"u": "x"})
    assert msgs == snapshot


# ---- T5.B4 / A5.B4 hash_image_url --------------------------------------------

@pytest.mark.asyncio
async def test_T5B4_hash_data_url() -> None:
    url, raw = _png_data_url()
    h, body = await hash_image_url(url, fetch_base="http://x")
    assert h == hashlib.sha256(raw).hexdigest()
    assert body == raw


@pytest.mark.asyncio
async def test_T5B4_hash_bad_scheme() -> None:
    with pytest.raises(ValueError):
        await hash_image_url("ftp://x.png", fetch_base="http://x")


@pytest.mark.asyncio
async def test_T5B4_hash_empty_data_url() -> None:
    with pytest.raises(ValueError):
        await hash_image_url("data:image/png;base64,", fetch_base="http://x")


@pytest.mark.asyncio
async def test_A5B4_hash_relative_path(monkeypatch) -> None:
    """A5.B4: /api/files/<id> resolves against fetch_base via httpx."""
    _, raw = _png_data_url()

    class _Resp:
        def __init__(self, content: bytes) -> None:
            self.content = content
        def raise_for_status(self) -> None: pass

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, _url): return _Resp(raw)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    h, body = await hash_image_url(
        "/api/files/abc", fetch_base="http://127.0.0.1:8000",
    )
    assert h == hashlib.sha256(raw).hexdigest()
    assert body == raw
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_image_compressor_messages.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.image_compressor.messages'`.

- [ ] **Step 3: Implement `messages.py`**

Create `backend/app/services/image_compressor/messages.py`:

```python
"""Pure helpers for the image compressor: scanning, hashing, rewriting."""
from __future__ import annotations

import base64
import copy
import hashlib
from typing import Iterator, Optional

import httpx


def has_images(msg: dict) -> bool:
    """True if `msg` carries at least one image_url part with non-empty url."""
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        p.get("type") == "image_url" and p.get("image_url", {}).get("url")
        for p in content
    )


def iter_image_parts(msgs: list[dict]) -> Iterator[tuple[int, int, str]]:
    """Yield (msg_idx, content_idx, url) for every image_url part in `msgs`.
    Messages whose content is a string are skipped silently.
    """
    for i, msg in enumerate(msgs):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, part in enumerate(content):
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url:
                    yield i, j, url


def find_latest_image_turn(msgs: list[dict]) -> Optional[int]:
    """Index of the latest USER turn that carries images; None if none."""
    latest: Optional[int] = None
    for i, msg in enumerate(msgs):
        if msg.get("role") == "user" and has_images(msg):
            latest = i
    return latest


def text_of(msg: dict) -> str:
    """Pull the first text part out of a message (string content → as-is,
    list content → first {type:'text'} entry, else empty)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                return part.get("text") or ""
    return ""


async def hash_image_url(
    url: str,
    fetch_base: str,
    fetch_timeout_s: int = 10,
) -> tuple[str, bytes]:
    """Resolve `url` to raw bytes and SHA-256-hash them.

    Supports `data:`, absolute http(s)://, and relative paths (resolved
    against `fetch_base`, e.g. http://127.0.0.1:8000). Empty / malformed
    inputs raise ValueError; non-2xx http responses raise via `raise_for_status`.
    """
    if url.startswith("data:"):
        if "," not in url:
            raise ValueError("malformed data URL")
        raw = base64.b64decode(url.split(",", 1)[1])
        if not raw:
            raise ValueError("data URL payload is empty")
    elif url.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=fetch_timeout_s) as client:
            r = await client.get(url)
            r.raise_for_status()
            raw = r.content
    elif url.startswith("/"):
        full = f"{fetch_base.rstrip('/')}{url}"
        async with httpx.AsyncClient(timeout=fetch_timeout_s) as client:
            r = await client.get(full)
            r.raise_for_status()
            raw = r.content
    else:
        raise ValueError(f"Unsupported image URL scheme: {url[:32]!r}")
    return hashlib.sha256(raw).hexdigest(), raw


def rewrite_messages(
    msgs: list[dict],
    keep_idx: Optional[int],
    captions_by_url: dict[str, str],
) -> list[dict]:
    """Deep-copy `msgs`; strip `image_url` parts at every turn except `keep_idx`,
    and append `[Past image #N: <caption>]` text where they were stripped.

    Per-message numbering: each message's stripped images get #1, #2, ...
    starting from 1 again — so the model can correlate captions to positions
    inside the same turn.
    """
    out = copy.deepcopy(msgs)
    for i, msg in enumerate(out):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        if i == keep_idx:
            continue
        new_parts: list[dict] = []
        stripped_captions: list[str] = []
        img_n = 0
        for part in content:
            if part.get("type") == "image_url":
                img_n += 1
                url = part.get("image_url", {}).get("url", "")
                cap = captions_by_url.get(url) or "(no caption)"
                stripped_captions.append(f"[Past image #{img_n}: {cap}]")
            else:
                new_parts.append(part)
        if stripped_captions:
            extra_text = "\n".join(stripped_captions)
            if new_parts and new_parts[-1].get("type") == "text":
                new_parts[-1]["text"] = (
                    new_parts[-1]["text"] + "\n" + extra_text
                )
            else:
                new_parts.append({"type": "text", "text": extra_text})
        msg["content"] = new_parts
    return out
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/test_image_compressor_messages.py -v
```

Expected: 12 PASSED.

- [ ] **Step 5: Confirm full suite**

```
pytest -q
```

Expected: 82 PASSED (70 + 12).

- [ ] **Step 6: Commit**

```
git add backend/app/services/image_compressor/messages.py backend/tests/test_image_compressor_messages.py
git commit -m "feat(backend): add image_compressor message helpers (hash, iter, rewrite)"
```

---

### Task 5: Engine units — `caption_one`, `route`, `ensure_captions`

**Files:**
- Create: `backend/app/services/image_compressor/engine.py` (partial — just the units; `compress()` lands in Task 6)
- Create: `backend/tests/test_image_compressor_engine_units.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_image_compressor_engine_units.py`:

```python
"""Unit tests for engine helpers: caption_one, route, ensure_captions."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.image_compressor.cache import CaptionCache
from app.services.image_compressor.engine import ImageCompressorEngine


def _engine(cache: CaptionCache, *, generate_raw: AsyncMock) -> ImageCompressorEngine:
    """Engine with a fake VLMManager whose `generate_raw` is a script."""
    fake_mgr = MagicMock()
    fake_mgr.generate_raw = generate_raw
    return ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
        webui_internal_base="http://localhost:8000",
    )


# ---- T5.B5 caption_one happy path -------------------------------------------

@pytest.mark.asyncio
async def test_T5B5_caption_one_returns_trimmed(tmp_path: Path) -> None:
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()
    gen = AsyncMock(return_value="  Một con mèo đen.  \n")
    engine = _engine(cache, generate_raw=gen)
    assert await engine.caption_one("data:image/png;base64,iVBORw==") == "Một con mèo đen."

    # Check it called manager with caption-shaped messages + max_tokens override.
    args, kwargs = gen.call_args
    assert kwargs["max_tokens"] == 80
    assert kwargs["messages"][0]["role"] == "system"
    assert "image captioner" in kwargs["messages"][0]["content"].lower()
    assert kwargs["messages"][1]["role"] == "user"


# ---- T5.B6 route happy + JSON parse fail-open --------------------------------

@pytest.mark.asyncio
async def test_T5B6_route_happy(tmp_path: Path) -> None:
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()
    gen = AsyncMock(return_value='{"need_images": true, "reason": "câu hỏi đề cập ảnh"}')
    engine = _engine(cache, generate_raw=gen)

    decision, reason = await engine.route("Cái này là gì?", ["caption A"])
    assert decision is True
    assert reason == "câu hỏi đề cập ảnh"


@pytest.mark.asyncio
async def test_T5B6_route_non_json_fails_open_keep(tmp_path: Path) -> None:
    """When the router returns garbage, we fall back to router_failopen_keep
    (default True)."""
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()
    gen = AsyncMock(return_value="LOL not JSON")
    engine = _engine(cache, generate_raw=gen)

    decision, reason = await engine.route("Q?", ["cap"])
    assert decision is True  # default failopen_keep
    assert "router failure" in reason


@pytest.mark.asyncio
async def test_T5B6_route_failopen_drop(tmp_path: Path) -> None:
    """If router_failopen_keep is False, parse error → drop images."""
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()
    gen = AsyncMock(return_value="not json")
    fake_mgr = MagicMock(); fake_mgr.generate_raw = gen
    engine = ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
        router_failopen_keep=False,
    )

    decision, _ = await engine.route("Q?", ["cap"])
    assert decision is False


# ---- A5.B1 ensure_captions: per-image fail-open + cache fill -----------------

@pytest.mark.asyncio
async def test_A5B1_ensure_captions_partial_failure(tmp_path: Path) -> None:
    """ensure_captions: call 3 captions; one raises → it's OMITTED from the
    result; other two succeed and are written to cache."""
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()

    call_count = {"n": 0}

    async def fake_generate_raw(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("boom")
        return f"caption #{call_count['n']}"

    fake_mgr = MagicMock(); fake_mgr.generate_raw = fake_generate_raw
    engine = ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
    )

    scanned = [
        (0, 0, "data:image/png;base64,A==", "h1", b"\x00"),
        (0, 1, "data:image/png;base64,B==", "h2", b"\x01"),
        (0, 2, "data:image/png;base64,C==", "h3", b"\x02"),
    ]
    out = await engine.ensure_captions(scanned)

    assert "data:image/png;base64,A==" in out
    assert "data:image/png;base64,C==" in out
    assert "data:image/png;base64,B==" not in out  # the failure is omitted

    # Cache filled with 2 captions (the survivors), keyed by hash:
    assert await cache.get("h1") is not None
    assert await cache.get("h2") is None
    assert await cache.get("h3") is not None


# ---- A5.B6 dedup-by-hash ------------------------------------------------------

@pytest.mark.asyncio
async def test_A5B6_ensure_captions_cache_hit(tmp_path: Path) -> None:
    """Hash-based dedup: 2 entries with same hash → captioner called once."""
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()
    await cache.put("h_shared", "cached caption", model="m")

    gen = AsyncMock(return_value="UNCALLED")
    fake_mgr = MagicMock(); fake_mgr.generate_raw = gen
    engine = ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
    )

    scanned = [
        (0, 0, "url-A", "h_shared", b"\x00"),
        (1, 0, "url-B", "h_shared", b"\x00"),
    ]
    out = await engine.ensure_captions(scanned)
    assert out == {"url-A": "cached caption", "url-B": "cached caption"}
    gen.assert_not_called()
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_image_compressor_engine_units.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.image_compressor.engine'`.

- [ ] **Step 3: Implement `engine.py` (units only)**

Create `backend/app/services/image_compressor/engine.py`:

```python
"""ImageCompressorEngine: orchestrates caption + router calls and history
rewriting. Built across Tasks 5-6 of the Phase 5 plan.

Task 5 (this commit): __init__, caption_one, route, ensure_captions.
Task 6 lands `compress()` — the async generator that drives chat_stream.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from app.services.image_compressor.cache import CaptionCache
from app.services.image_compressor.prompts import (
    CAPTION_SYSTEM_PROMPT,
    CAPTION_USER_TEXT,
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_TEMPLATE,
)
from app.services.image_compressor.types import Scanned

log = logging.getLogger("image_compressor")


class ImageCompressorEngine:
    def __init__(
        self,
        cache: CaptionCache,
        vlm_manager,  # type: ignore[no-untyped-def]  # circular-import dodge
        *,
        caption_model_id: str,
        router_model_id: str,
        webui_internal_base: str = "http://127.0.0.1:8000",
        caption_max_tokens: int = 80,
        router_max_tokens: int = 60,
        caption_timeout_s: int = 30,
        router_timeout_s: int = 15,
        router_failopen_keep: bool = True,
    ) -> None:
        self.cache = cache
        self.manager = vlm_manager
        self.caption_model_id = caption_model_id
        self.router_model_id = router_model_id
        self.webui_internal_base = webui_internal_base
        self.caption_max_tokens = caption_max_tokens
        self.router_max_tokens = router_max_tokens
        self.caption_timeout_s = caption_timeout_s
        self.router_timeout_s = router_timeout_s
        self.router_failopen_keep = router_failopen_keep

    async def caption_one(self, data_url: str) -> str:
        """Caption a single image. Returns trimmed caption text."""
        messages = [
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": CAPTION_USER_TEXT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]
        text = await self.manager.generate_raw(
            self.caption_model_id, messages,
            max_tokens=self.caption_max_tokens, temperature=0.2,
        )
        return text.strip()

    async def route(
        self, user_text: str, captions: list[str],
    ) -> tuple[bool, str]:
        """Decide whether the LLM needs pixels for this user turn.

        Returns (need_images, reason). On any error (HTTP, JSON parse,
        missing key) returns (router_failopen_keep, "router failure: ...").
        """
        captions_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(captions))
        user_content = ROUTER_USER_TEMPLATE.format(
            captions_block=captions_block, user_text=user_text,
        )
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            raw = await self.manager.generate_raw(
                self.router_model_id, messages,
                max_tokens=self.router_max_tokens, temperature=0.0,
            )
            parsed = json.loads(raw)
            decision = bool(parsed.get("need_images"))
            reason = str(parsed.get("reason", ""))[:200]
            return decision, reason
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            log.warning("router parse failed: %s", e)
            return self.router_failopen_keep, f"router failure: {type(e).__name__}"
        except Exception as e:  # network / runtime
            log.warning("router call failed: %s", e)
            return self.router_failopen_keep, f"router failure: {type(e).__name__}"

    async def ensure_captions(
        self, scanned: list[Scanned], *, user_id: Optional[str] = None,
    ) -> dict[str, str]:
        """Return {url: caption} for every image in `scanned`. Cache hits
        served first; misses captioned in parallel and persisted. A single
        caption-call failure omits that url from the result and does NOT
        bring down the others.
        """
        if not scanned:
            return {}

        hashes = [h for _, _, _, h, _ in scanned]
        hits = await self.cache.get_many(hashes)

        out: dict[str, str] = {}
        misses: list[tuple[str, str, bytes]] = []
        for _, _, url, h, raw in scanned:
            if h in hits:
                out[url] = hits[h]
            else:
                misses.append((url, h, raw))

        if not misses:
            return out

        async def _one(url: str) -> tuple[str, Optional[str]]:
            try:
                cap = await self.caption_one(url)
                return url, cap or None
            except Exception as e:
                log.warning("caption failed for url=%s err=%s", url[:60], e)
                return url, None

        results = await asyncio.gather(*(_one(url) for url, _, _ in misses))

        new_rows: list[tuple[str, str, str, Optional[int], Optional[str]]] = []
        for (url, h, raw), (_, cap) in zip(misses, results):
            if cap:
                out[url] = cap
                new_rows.append((h, cap, self.caption_model_id, len(raw), user_id))

        if new_rows:
            await self.cache.put_many(new_rows)

        return out
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/test_image_compressor_engine_units.py -v
```

Expected: 7 PASSED.

- [ ] **Step 5: Confirm full suite**

```
pytest -q
```

Expected: 89 PASSED (82 + 7).

- [ ] **Step 6: Commit**

```
git add backend/app/services/image_compressor/engine.py backend/tests/test_image_compressor_engine_units.py
git commit -m "feat(backend): add caption_one + route + ensure_captions to compressor engine"
```

---

### Task 6: Engine `compress()` orchestrator

**Files:**
- Modify: `backend/app/services/image_compressor/engine.py`
- Create: `backend/tests/test_image_compressor_engine_compress.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_image_compressor_engine_compress.py`:

```python
"""Integration tests for ImageCompressorEngine.compress (Phase 5)."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.image_compressor.cache import CaptionCache
from app.services.image_compressor.engine import ImageCompressorEngine
from app.services.image_compressor.types import (
    CompressionResult, StatusEvent,
)


def _png_data_url(byte: int) -> str:
    """Distinct 1-byte payloads → distinct hashes."""
    raw = bytes([byte])
    b64 = base64.b64encode(raw).decode()
    return f"data:image/png;base64,{b64}"


def _expected_hash(byte: int) -> str:
    return hashlib.sha256(bytes([byte])).hexdigest()


# ---- T5.B7 happy path with images --------------------------------------------

@pytest.mark.asyncio
async def test_T5B7_compress_with_images(tmp_path: Path) -> None:
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()

    gen = AsyncMock(side_effect=[
        "Caption for image one.",
        "Caption for image two.",
        '{"need_images": false, "reason": "không liên quan ảnh"}',
    ])
    fake_mgr = MagicMock(); fake_mgr.generate_raw = gen
    engine = ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
    )

    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _png_data_url(1)}},
            {"type": "text", "text": "first turn"},
        ]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _png_data_url(2)}},
            {"type": "text", "text": "second turn"},
        ]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "third turn (text-only)"},
    ]

    events: list = []
    async for ev in engine.compress(msgs):
        events.append(ev)

    # At least one StatusEvent and exactly one terminal CompressionResult.
    statuses = [e for e in events if isinstance(e, StatusEvent)]
    results = [e for e in events if isinstance(e, CompressionResult)]
    assert len(statuses) >= 2          # captioning + routing + done
    assert any(s.done for s in statuses)
    assert len(results) == 1

    result = results[0]
    # Router said no-pixels → both image turns get `[Past image #1: ...]` inserts.
    text_0 = result.messages[0]["content"][-1]["text"]
    assert "[Past image #1: Caption for image one.]" in text_0
    text_2 = result.messages[2]["content"][-1]["text"]
    assert "[Past image #1: Caption for image two.]" in text_2
    # Last user turn untouched (no images).
    assert result.messages[4] == msgs[4]
    # Thinking md present and mentions reasoning.
    assert "🧠 Image compressor reasoning" in result.thinking_md
    assert "<details>" in result.thinking_md and "</details>" in result.thinking_md


# ---- T5.B7-keep: latest turn has images → keep_idx = latest ------------------

@pytest.mark.asyncio
async def test_T5B7_compress_latest_has_images(tmp_path: Path) -> None:
    """If the latest user turn carries images, we skip the router and keep
    those pixels (decision_label='kept new upload')."""
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()

    gen = AsyncMock(return_value="Caption for older image.")
    fake_mgr = MagicMock(); fake_mgr.generate_raw = gen
    engine = ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
    )

    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _png_data_url(1)}},
            {"type": "text", "text": "old"},
        ]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _png_data_url(2)}},
            {"type": "text", "text": "new"},
        ]},
    ]

    events: list = [ev async for ev in engine.compress(msgs)]
    result = next(e for e in events if isinstance(e, CompressionResult))

    # Old turn stripped + caption insert; new turn UNTOUCHED.
    assert "[Past image #1: Caption for older image.]" in result.messages[0]["content"][-1]["text"]
    assert result.messages[2]["content"] == msgs[2]["content"]
    # We did NOT call the router (gen called once for the old image only).
    assert gen.call_count == 1


# ---- T5.B7-fast: 0 images → fast path ---------------------------------------

@pytest.mark.asyncio
async def test_T5B7_compress_no_images_fast_path(tmp_path: Path) -> None:
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()
    gen = AsyncMock()  # should NEVER be called
    fake_mgr = MagicMock(); fake_mgr.generate_raw = gen
    engine = ImageCompressorEngine(
        cache=cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
    )

    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    events = [ev async for ev in engine.compress(msgs)]

    # Exactly one CompressionResult, zero StatusEvents.
    assert len(events) == 1
    assert isinstance(events[0], CompressionResult)
    assert events[0].messages == msgs
    assert events[0].thinking_md == ""
    gen.assert_not_called()


# ---- A5.B5 engine self-catch passthrough -------------------------------------

@pytest.mark.asyncio
async def test_A5B5_compress_self_catches_exceptions(tmp_path: Path) -> None:
    """Any exception inside compress is caught; result is a passthrough
    with thinking_md=''. Caller never sees the exception."""
    cache = CaptionCache(str(tmp_path / "c.db")); await cache.init()

    # Inject a bug: the cache.get_many on the engine raises.
    class BoomCache(CaptionCache):
        async def get_many(self, hashes):  # type: ignore[override]
            raise RuntimeError("DB unavailable")

    bad_cache = BoomCache(str(tmp_path / "c.db"))
    await bad_cache.init()

    gen = AsyncMock(return_value="caption")
    fake_mgr = MagicMock(); fake_mgr.generate_raw = gen
    engine = ImageCompressorEngine(
        cache=bad_cache, vlm_manager=fake_mgr,
        caption_model_id="m", router_model_id="m",
    )

    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _png_data_url(1)}},
        ]},
    ]
    events = [ev async for ev in engine.compress(msgs)]

    # Last event MUST be a passthrough CompressionResult.
    assert isinstance(events[-1], CompressionResult)
    assert events[-1].messages == msgs   # untouched
    assert events[-1].thinking_md == ""  # empty on failure
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_image_compressor_engine_compress.py -v
```

Expected: FAIL — `AttributeError: 'ImageCompressorEngine' object has no attribute 'compress'`.

- [ ] **Step 3: Implement `compress()` and the thinking-log builder**

Append to `backend/app/services/image_compressor/engine.py` (inside the `ImageCompressorEngine` class, AFTER `ensure_captions`):

```python
    def _estimate_image_tokens(self, raw: bytes) -> int:
        """Cheap heuristic: ~800 tokens minimum, +1 token per 800 bytes."""
        return max(800, len(raw) // 800)

    def _build_thinking_log(
        self, *,
        n_images: int, n_misses: int, decision_label: str,
        captions_used: list[tuple[str, str]],
        user_text: Optional[str],
        route_reason: Optional[str],
        tokens_saved: int,
    ) -> str:
        """Build the <details>-wrapped reasoning markdown shown above the
        assistant reply."""
        lines = [
            "<details>",
            f"<summary>🧠 Image compressor reasoning ({n_images} ảnh, "
            f"{n_misses} caption mới, {decision_label})</summary>",
            "",
            "**Step 1 — Image scan**",
            f"- Tổng {n_images} ảnh; cache miss: {n_misses}, "
            f"hit: {n_images - n_misses}",
            "",
        ]
        if captions_used:
            lines.append("**Step 2 — Captions in use**")
            for h_short, cap in captions_used:
                lines.append(f"- `{h_short}` → \"{cap[:120]}\"")
            lines.append("")
        if user_text is not None:
            lines.append("**Step 3 — Router**")
            lines.append(f"- User: \"{user_text[:200]}\"")
            lines.append(f"- {decision_label}")
            if route_reason:
                lines.append(f"- Reason: *{route_reason}*")
            lines.append("")
        lines.append("**Step 4 — Rewrite**")
        if tokens_saved > 0:
            lines.append(f"- Token estimate saved: ~{tokens_saved}")
        else:
            lines.append("- Images preserved; no tokens saved")
        lines.append("</details>")
        lines.append("")
        return "\n".join(lines)

    async def compress(self, messages: list[dict]):
        """Async generator. Yields zero or more StatusEvents while working,
        then exactly one terminal CompressionResult. Self-catches every
        internal exception → on failure, yields passthrough(messages, '')."""
        try:
            async for ev in self._compress_impl(messages):
                yield ev
        except Exception as e:
            log.exception("compressor crash; falling through to passthrough: %s", e)
            yield CompressionResult(messages=messages, thinking_md="")

    async def _compress_impl(self, messages: list[dict]):
        from app.services.image_compressor.messages import (
            find_latest_image_turn, has_images, hash_image_url,
            iter_image_parts, rewrite_messages, text_of,
        )

        url_list = list(iter_image_parts(messages))
        if not url_list:
            yield CompressionResult(messages=messages, thinking_md="")
            return

        # Hash each image (fail-soft).
        scanned: list[Scanned] = []
        for msg_idx, c_idx, url in url_list:
            try:
                h, raw = await hash_image_url(url, self.webui_internal_base)
                scanned.append((msg_idx, c_idx, url, h, raw))
            except Exception as e:
                log.warning("hash skipped url=%s err=%s", url[:60], e)

        if not scanned:
            yield CompressionResult(messages=messages, thinking_md="")
            return

        # Cache lookup → count misses for the status banner + thinking log.
        existing = await self.cache.get_many([h for *_, h, _ in scanned])
        n_misses = sum(1 for s in scanned if s[3] not in existing)
        if n_misses > 0:
            yield StatusEvent(
                message=f"🖼️ Captioning {n_misses} new image(s)..."
            )

        captions_by_url = await self.ensure_captions(scanned)

        # Decide keep_idx.
        last_msg = messages[-1] if messages else None
        if last_msg and has_images(last_msg):
            keep_idx: Optional[int] = len(messages) - 1
            decision_label = "kept new upload"
            user_text_for_log: Optional[str] = None
            route_reason: Optional[str] = None
        else:
            latest_idx = find_latest_image_turn(messages)
            if latest_idx is None:
                # No usable image turn → pure-text history; nothing to compress.
                yield CompressionResult(messages=messages, thinking_md="")
                return
            yield StatusEvent(message="🧭 Routing: do we need pixels?")
            captions_for_router = [
                captions_by_url.get(u, "(no caption)")
                for (mi, _, u, _, _) in scanned if mi == latest_idx
            ]
            user_text_for_log = text_of(last_msg) if last_msg else ""
            decision, route_reason = await self.route(
                user_text_for_log, captions_for_router,
            )
            if decision:
                keep_idx = latest_idx
                decision_label = "🎯 Router: keep images"
            else:
                keep_idx = None
                decision_label = "🎯 Router: drop images"
            yield StatusEvent(message=decision_label)

        # Token-saving estimate.
        if keep_idx is None:
            tokens_saved = sum(
                self._estimate_image_tokens(raw) for *_, raw in scanned
            )
        else:
            tokens_saved = sum(
                self._estimate_image_tokens(raw)
                for (mi, _, _, _, raw) in scanned if mi != keep_idx
            )

        new_messages = rewrite_messages(messages, keep_idx, captions_by_url)

        captions_used = [
            (h[:8], captions_by_url.get(url, "(no caption)"))
            for (_, _, url, h, _) in scanned
        ]
        thinking_md = self._build_thinking_log(
            n_images=len(scanned), n_misses=n_misses,
            decision_label=decision_label, captions_used=captions_used,
            user_text=user_text_for_log, route_reason=route_reason,
            tokens_saved=tokens_saved,
        )

        yield StatusEvent(message="✅ Compressor done", done=True)
        yield CompressionResult(messages=new_messages, thinking_md=thinking_md)
```

Also add `from app.services.image_compressor.types import CompressionResult` at the top of the file (next to the existing `Scanned` import).

- [ ] **Step 4: Run to verify pass**

```
pytest tests/test_image_compressor_engine_compress.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Confirm full suite**

```
pytest -q
```

Expected: 93 PASSED (89 + 4).

- [ ] **Step 6: Commit**

```
git add backend/app/services/image_compressor/engine.py backend/tests/test_image_compressor_engine_compress.py
git commit -m "feat(backend): add ImageCompressorEngine.compress orchestrator"
```

---

### Task 7: Lifespan + chat_stream wiring

**Files:**
- Modify: `backend/app/models/vlm/manager.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/routers/chat.py`
- Modify: `backend/app/services/messages.py`
- Modify: `backend/app/services/image_compressor/__init__.py`
- Create: `backend/tests/test_chat_stream_compressor.py`

This is the largest task. We:
1. Re-export `ImageCompressorEngine` from the package (so `from app.services.image_compressor import ImageCompressorEngine` works in `main.py`).
2. Teach `VLMManager` to read the `compressor:` block.
3. Wire the engine into `lifespan()`.
4. Thread it into `chat_stream` with the new SSE shape.
5. Update the dumb `enforce_image_cap` doc-comment to flag its new fallback role.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_chat_stream_compressor.py`:

```python
"""Integration tests: chat_stream + compressor engine (Phase 5)."""
from __future__ import annotations

import json

import pytest

from app.services.image_compressor.types import (
    CompressionResult, StatusEvent,
)


@pytest.fixture
def fake_engine():
    """A fake engine that scripts a fixed sequence of events for compress()."""
    class _FakeEngine:
        def __init__(self, *, events):
            self.events = events
            self.compress_calls = 0

        async def compress(self, messages):
            self.compress_calls += 1
            for ev in self.events:
                yield ev

    return _FakeEngine


def _parse_sse(body: bytes) -> list[dict]:
    """Parse SSE body into a list of decoded JSON payloads."""
    out = []
    for chunk in body.split(b"\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith(b"data:"):
            payload = chunk[5:].strip()
            out.append(json.loads(payload))
    return out


def test_T5B8_chat_stream_with_compressor(client, fake_manager, fake_engine):
    """T5.B8: 1 image in history → compressor emits status + thinking-log
    delta, then model deltas, then done."""
    from app.main import app

    img_url = "data:image/png;base64,iVBORw0KGgo="
    rewritten = [{"role": "user", "content": [
        {"type": "text", "text": "describe\n[Past image #1: a cat]"},
    ]}]

    app.state.compressor_engine = fake_engine(events=[
        StatusEvent(message="🖼️ Captioning 1 new image(s)..."),
        StatusEvent(message="✅ Compressor done", done=True),
        CompressionResult(
            messages=rewritten,
            thinking_md="<details><summary>🧠 reasoning</summary>...</details>",
        ),
    ])

    async def fake_stream(model_id, messages):
        # Compressor's rewritten messages reach the provider.
        assert messages == rewritten
        for delta in ["Hello ", "world."]:
            yield delta
    fake_manager.stream = fake_stream

    resp = client.post(
        "/api/chat/stream",
        json={
            "model_id": "fake-vision",
            "messages": [{
                "role": "user", "text": "describe",
                "attachments": [],
            }],
        },
    )
    assert resp.status_code == 200
    payloads = _parse_sse(resp.content)

    types_seen = []
    for p in payloads:
        if p.get("type") == "status":
            types_seen.append(("status", p.get("message"), p.get("done")))
        elif "delta" in p:
            types_seen.append(("delta", p["delta"], p.get("done")))
        else:
            types_seen.append(("error", p))

    assert types_seen[0] == ("status", "🖼️ Captioning 1 new image(s)...", False)
    assert types_seen[1] == ("status", "✅ Compressor done", True)
    # Next event is the thinking-log delta.
    assert types_seen[2][0] == "delta"
    assert "<details>" in types_seen[2][1]
    # Followed by model deltas.
    assert types_seen[3] == ("delta", "Hello ", False)
    assert types_seen[4] == ("delta", "world.", False)
    # Terminal done.
    assert types_seen[5] == ("delta", "", True)


def test_T5B9_chat_stream_no_images_fast_path(client, fake_manager, fake_engine):
    """T5.B9: 0 images → engine fast-path returns CompressionResult only;
    no status events, no thinking-log delta in stream."""
    from app.main import app
    app.state.compressor_engine = fake_engine(events=[
        CompressionResult(messages=[
            {"role": "user", "content": "hello"},
        ], thinking_md=""),
    ])

    async def fake_stream(model_id, messages):
        for delta in ["Hi ", "there."]:
            yield delta
    fake_manager.stream = fake_stream

    resp = client.post(
        "/api/chat/stream",
        json={
            "model_id": "fake-text",
            "messages": [{
                "role": "user", "text": "hello",
                "attachments": [],
            }],
        },
    )
    payloads = _parse_sse(resp.content)
    deltas = [p["delta"] for p in payloads if "delta" in p]
    statuses = [p for p in payloads if p.get("type") == "status"]

    # No status events, no thinking-log delta → just the model output + terminal.
    assert statuses == []
    assert deltas == ["Hi ", "there.", ""]


def test_A5_11_compressor_disabled_passthrough(client, fake_manager):
    """A5.11: when compressor_engine is None on app.state, chat_stream
    behaves exactly like Phase 4."""
    from app.main import app
    app.state.compressor_engine = None

    async def fake_stream(model_id, messages):
        for delta in ["Plain ", "answer."]:
            yield delta
    fake_manager.stream = fake_stream

    resp = client.post(
        "/api/chat/stream",
        json={
            "model_id": "fake-text",
            "messages": [{
                "role": "user", "text": "hi",
                "attachments": [],
            }],
        },
    )
    payloads = _parse_sse(resp.content)
    statuses = [p for p in payloads if p.get("type") == "status"]
    deltas = [p["delta"] for p in payloads if "delta" in p]

    assert statuses == []
    assert deltas == ["Plain ", "answer.", ""]
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_chat_stream_compressor.py -v
```

Expected: FAIL — `app.state.compressor_engine` does not exist (or AttributeError, depending on order).

- [ ] **Step 3: Re-export the engine**

Update `backend/app/services/image_compressor/__init__.py`:

```python
"""Image-aware history compressor: ports open-webui's qwenvl_image_compress
filter into our backend's service layer.
"""
from app.services.image_compressor.engine import ImageCompressorEngine

__all__ = ["ImageCompressorEngine"]
```

- [ ] **Step 4: Add `compressor_config()` to `VLMManager`**

In `backend/app/models/vlm/manager.py`, append a new method to the class (after `generate_raw`):

```python
    def compressor_config(self) -> dict | None:
        """Return the `compressor:` block from models.yaml, or None if absent
        or malformed. Validates that referenced model ids exist.
        """
        yaml_path = Path(__file__).parent / "models.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        block = config.get("compressor")
        if not block:
            return None

        caption_id = block.get("caption_model_id")
        router_id = block.get("router_model_id")
        if not caption_id or caption_id not in self.providers:
            print(f"[VLMManager] compressor.caption_model_id '{caption_id}' "
                  f"not in providers; compressor disabled.")
            return None
        if not router_id or router_id not in self.providers:
            print(f"[VLMManager] compressor.router_model_id '{router_id}' "
                  f"not in providers; compressor disabled.")
            return None

        return block
```

- [ ] **Step 5: Wire the engine in `main.py` lifespan**

Replace `backend/app/main.py` lifespan:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.models.vlm import VLMManager
from app.routers import chat, files
from app.services.image_compressor import ImageCompressorEngine
from app.services.image_compressor.cache import CaptionCache


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = VLMManager()
    manager.load()
    app.state.vlm_manager = manager

    cfg = manager.compressor_config()
    if cfg:
        cache = CaptionCache(cfg["cache_db_path"])
        await cache.init()
        app.state.compressor_engine = ImageCompressorEngine(
            cache=cache, vlm_manager=manager,
            caption_model_id=cfg["caption_model_id"],
            router_model_id=cfg["router_model_id"],
            webui_internal_base=cfg.get("webui_internal_base", "http://127.0.0.1:8000"),
            caption_max_tokens=cfg.get("caption_max_tokens", 80),
            router_max_tokens=cfg.get("router_max_tokens", 60),
            caption_timeout_s=cfg.get("caption_timeout_s", 30),
            router_timeout_s=cfg.get("router_timeout_s", 15),
            router_failopen_keep=cfg.get("router_failopen_keep", True),
        )
        print(f"[lifespan] compressor enabled "
              f"(caption={cfg['caption_model_id']}, router={cfg['router_model_id']})")
    else:
        app.state.compressor_engine = None
        print("[lifespan] compressor disabled (no `compressor:` yaml block)")
    yield


app = FastAPI(
    title="OE-VLM Shop API",
    description="E-commerce Chatbot API powered by FastAPI and VLM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(files.router)

app.mount("/images", StaticFiles(directory="images"), name="images")


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 6: Add `_sse_status` + thread the engine through `chat_stream`**

Modify `backend/app/routers/chat.py`. After the existing `_sse_error` helper (around line 123), add:

```python
def _sse_status(message: str, done: bool) -> str:
    return f"data: {json.dumps({'type': 'status', 'message': message, 'done': done}, ensure_ascii=False)}\n\n"
```

Then replace the body of the `event_stream` async generator inside `chat_stream` (currently lines ~130-152) with:

```python
    async def event_stream():
        try:
            openai_messages = build_openai_messages(body.messages)
        except FileNotFoundError as exc:
            yield _sse_error("file_missing", str(exc))
            return

        # Phase 5: image-aware compressor. Engine self-catches its own errors
        # and returns a passthrough on failure. The outer try/except below is a
        # second-line guard for unexpected iteration-level issues only.
        engine = getattr(request.app.state, "compressor_engine", None)
        thinking_md = ""
        if engine is not None:
            try:
                from app.services.image_compressor.types import (
                    CompressionResult, StatusEvent,
                )
                async for event in engine.compress(openai_messages):
                    if isinstance(event, StatusEvent):
                        yield _sse_status(event.message, event.done)
                    elif isinstance(event, CompressionResult):
                        openai_messages = event.messages
                        thinking_md = event.thinking_md
                        break
            except Exception:
                traceback.print_exc()

        # Safety net: even if compressor passed-through with too many images
        # (or there's no compressor at all), enforce vLLM's hard limit of 4.
        openai_messages = enforce_image_cap(openai_messages, max_images=4)

        try:
            first_emitted = False
            async for delta in manager.stream(body.model_id, openai_messages):
                if await request.is_disconnected():
                    return
                if not first_emitted and thinking_md:
                    yield _sse_delta(thinking_md + delta)
                else:
                    yield _sse_delta(delta)
                first_emitted = True
            if not first_emitted and thinking_md:
                # The model produced zero deltas but we still want the
                # thinking-log visible (e.g., model 200 with empty body).
                yield _sse_delta(thinking_md)
            yield _sse_done()
        except ConnectionError as exc:
            yield _sse_error("connection", str(exc))
        except BadRequestError as exc:
            yield _sse_error("bad_request", str(exc))
        except RuntimeError as exc:
            yield _sse_error("bad_request", str(exc))
        except Exception:
            traceback.print_exc()
            yield _sse_error("internal", "Internal error")
```

Replace `manager` from the line above to be `request.app.state.vlm_manager` (the existing local was `manager = request.app.state.vlm_manager` on line 128 — keep it; just confirm the new body still references it).

Also ensure these imports exist near the top of `chat.py`:

```python
from app.services.messages import build_openai_messages, enforce_image_cap
```

(They already do — but verify after the edit.)

- [ ] **Step 7: Update `enforce_image_cap` docstring**

Modify `backend/app/services/messages.py:38-44` (the existing `enforce_image_cap` docstring). Replace it with:

```python
def enforce_image_cap(messages: list[dict], max_images: int = 4) -> list[dict]:
    """Defensive backstop: cap the total `image_url` parts at `max_images` by
    replacing the OLDEST images with a dumb placeholder string.

    From Phase 5 onward this is a fallback for when `ImageCompressorEngine`
    crashes or is disabled — the compressor strips history images down to ≤1
    pixel-bearing turn before the request reaches here, so under normal
    operation this function is a no-op. Default `max_images=4` matches vLLM's
    `--limit-mm-per-prompt image=4` hard cap.
    """
```

- [ ] **Step 8: Run the new tests**

```
cd backend
pytest tests/test_chat_stream_compressor.py -v
```

Expected: 3 PASSED.

- [ ] **Step 9: Run the full backend suite**

```
pytest -q
```

Expected: 96 PASSED (93 + 3). If any of the existing `test_chat_stream.py` tests fail, the typical cause is the new `app.state.compressor_engine` attribute being unset — they should default to `None` via the `getattr(..., None)` in `event_stream`, but if your conftest's `client` fixture creates the app fresh each time, ensure the lifespan still sets `compressor_engine` (it does, in Step 5 above). Otherwise add `app.state.compressor_engine = None` in the `client` fixture of `conftest.py`.

If you see "compressor_engine is not set" failures: append to `backend/tests/conftest.py`'s `client` fixture, just before `return TestClient(app)`:

```python
    if not hasattr(app.state, "compressor_engine"):
        app.state.compressor_engine = None
```

- [ ] **Step 10: Commit**

```
git add backend/app/services/image_compressor/__init__.py backend/app/models/vlm/manager.py backend/app/main.py backend/app/routers/chat.py backend/app/services/messages.py backend/tests/test_chat_stream_compressor.py backend/tests/conftest.py
git commit -m "feat(backend): wire image_compressor into chat_stream + lifespan"
```

---

### Task 8: Frontend SSE parser + chatStream `onStatus`

**Files:**
- Modify: `frontend/src/playground/lib/sseParser.ts`
- Modify: `frontend/src/playground/lib/sseParser.test.ts`
- Modify: `frontend/src/playground/lib/chatStream.ts`
- Modify: `frontend/src/playground/lib/chatStream.test.ts`

- [ ] **Step 1: Write the failing parser tests**

Append to `frontend/src/playground/lib/sseParser.test.ts`:

```ts
describe("Phase 5 status events", () => {
  it("T5.1 — parses {type:'status', message, done:false} → status event", () => {
    const buf =
      `data: {"type":"status","message":"Captioning...","done":false}\n\n`;
    const { events, rest } = drainEvents(buf);
    expect(rest).toBe("");
    expect(events).toEqual([
      { type: "status", message: "Captioning...", statusDone: false },
    ]);
  });

  it("T5.2 — parses status event with done:true", () => {
    const buf =
      `data: {"type":"status","message":"Done","done":true}\n\n`;
    const { events } = drainEvents(buf);
    expect(events).toEqual([
      { type: "status", message: "Done", statusDone: true },
    ]);
  });

  it("T5.3 — ignores unknown {type} shape; subsequent delta still parsed", () => {
    const buf =
      `data: {"type":"future_unknown","whatever":1}\n\n` +
      `data: {"delta":"hi","done":false}\n\n`;
    const { events } = drainEvents(buf);
    expect(events).toEqual([{ type: "delta", delta: "hi" }]);
  });
});
```

- [ ] **Step 2: Run parser tests to verify failure**

```
cd frontend
npm run test:run -- src/playground/lib/sseParser.test.ts
```

Expected: 3 new tests FAIL — parser doesn't recognize the `status` shape.

- [ ] **Step 3: Update `sseParser.ts`**

Replace `frontend/src/playground/lib/sseParser.ts` entirely:

```ts
/**
 * Pure SSE parser for the /api/chat/stream wire format.
 *
 * Backend emits one JSON payload per `data:` frame, separated by `\n\n`:
 *   {"delta":"...","done":false}                      normal token
 *   {"delta":"","done":true}                          end of stream
 *   {"error":"...","message":"..."}                   terminal error
 *   {"type":"status","message":"...","done":bool}     ephemeral status (Phase 5)
 *
 * Forward-compat: unknown JSON shapes (e.g. future Phase 6 meta events) are
 * silently ignored — drainEvents emits nothing for them.
 */

export type SseEvent =
  | { type: "delta"; delta: string }
  | { type: "done" }
  | { type: "error"; errorKind: string; message: string }
  | { type: "status"; message: string; statusDone: boolean }
  | { type: "parse_error"; raw: string };

const FRAME_DELIMITER = /\r?\n\r?\n/;

function parsePayload(raw: string): SseEvent | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { type: "parse_error", raw };
  }
  if (typeof parsed !== "object" || parsed === null) return null;
  const obj = parsed as Record<string, unknown>;

  // Status events are checked first — they share the wire with delta/done/error
  // but use a `type` discriminator. Forward-compat: any future `type:` other
  // than "status" is ignored.
  if (obj.type === "status" && typeof obj.message === "string") {
    return {
      type: "status",
      message: obj.message,
      statusDone: obj.done === true,
    };
  }
  if (typeof obj.error === "string") {
    return {
      type: "error",
      errorKind: obj.error,
      message: typeof obj.message === "string" ? obj.message : "",
    };
  }
  if (obj.done === true) return { type: "done" };
  if (typeof obj.delta === "string" && obj.done === false) {
    return { type: "delta", delta: obj.delta };
  }
  // Unknown shape: forward-compat, ignore.
  return null;
}

export function drainEvents(buffer: string): {
  events: SseEvent[];
  rest: string;
} {
  const blocks = buffer.split(FRAME_DELIMITER);
  const rest = blocks.pop() ?? "";

  const events: SseEvent[] = [];
  for (const block of blocks) {
    const trimmed = block.trim();
    if (!trimmed.startsWith("data:")) continue;
    const payload = trimmed.slice(5).trim();
    if (!payload) continue;
    const ev = parsePayload(payload);
    if (ev) events.push(ev);
  }
  return { events, rest };
}
```

- [ ] **Step 4: Run parser tests to verify pass**

```
npm run test:run -- src/playground/lib/sseParser.test.ts
```

Expected: all sseParser tests PASS (8 baseline + 3 new = 11).

- [ ] **Step 5: Write the failing chatStream test**

Append to `frontend/src/playground/lib/chatStream.test.ts`:

```ts
describe("Phase 5 onStatus callback", () => {
  it("T5.4 — invokes onStatus(message, done) when status events arrive", async () => {
    const body =
      `data: {"type":"status","message":"Captioning...","done":false}\n\n` +
      `data: {"type":"status","message":"Done","done":true}\n\n` +
      `data: {"delta":"hi","done":false}\n\n` +
      `data: {"delta":"","done":true}\n\n`;

    const fetchMock = vi.fn(async () =>
      new Response(body, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const onStatus = vi.fn();
    const onDelta = vi.fn();
    const onDone = vi.fn();
    const onError = vi.fn();

    await streamChat({
      signal: new AbortController().signal,
      messages: [],
      modelId: null,
      onDelta, onDone, onError, onStatus,
    });

    expect(onStatus.mock.calls).toEqual([
      ["Captioning...", false],
      ["Done", true],
    ]);
    expect(onDelta).toHaveBeenCalledWith("hi");
    expect(onDone).toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 6: Run chatStream test to verify failure**

```
npm run test:run -- src/playground/lib/chatStream.test.ts
```

Expected: T5.4 FAILS — `onStatus` is not a known field of ChatStreamCallbacks.

- [ ] **Step 7: Add `onStatus` to chatStream**

Replace `frontend/src/playground/lib/chatStream.ts`:

```ts
import { drainEvents } from "./sseParser";
import type { ChatMessageWithAttachments, ErrorKind } from "../types";

export type ChatStreamCallbacks = {
  onDelta: (delta: string) => void;
  onDone: () => void;
  onError: (e: { errorKind: ErrorKind; message: string }) => void;
  /** Phase 5: ephemeral status events from the image compressor. */
  onStatus?: (message: string, done: boolean) => void;
};

export type ChatStreamArgs = ChatStreamCallbacks & {
  signal: AbortSignal;
  messages: ChatMessageWithAttachments[];
  modelId: string | null;
};

/**
 * Pure async streaming worker. Owns the fetch call + reader loop,
 * but does NOT manage AbortController lifetime — caller passes signal.
 * AbortError is treated as a silent success (no callback fires).
 */
export async function streamChat(args: ChatStreamArgs): Promise<void> {
  const { signal, messages, modelId, onDelta, onDone, onError, onStatus } = args;

  let resp: Response;
  try {
    resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, model_id: modelId }),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    onError({ errorKind: "network", message: String(e) });
    return;
  }

  if (!resp.ok || !resp.body) {
    onError({ errorKind: "http", message: `HTTP ${resp.status}` });
    return;
  }

  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";

  const abortPromise = new Promise<never>((_, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
    } else {
      signal.addEventListener(
        "abort",
        () => reject(new DOMException("Aborted", "AbortError")),
        { once: true },
      );
    }
  });

  try {
    while (true) {
      const { done, value } = await Promise.race([reader.read(), abortPromise]);
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const { events, rest } = drainEvents(buf);
      buf = rest;
      for (const ev of events) {
        if (ev.type === "delta") onDelta(ev.delta);
        else if (ev.type === "status") onStatus?.(ev.message, ev.statusDone);
        else if (ev.type === "done") {
          onDone();
          return;
        } else if (ev.type === "error") {
          onError({
            errorKind: (ev.errorKind as ErrorKind) ?? "internal",
            message: ev.message,
          });
          return;
        } else if (ev.type === "parse_error") {
          // eslint-disable-next-line no-console
          console.warn("[chatStream] parse_error:", ev.raw);
        }
      }
    }
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    onError({ errorKind: "stream", message: String(e) });
  } finally {
    reader.releaseLock();
  }
}
```

- [ ] **Step 8: Run chatStream test to verify pass**

```
npm run test:run -- src/playground/lib/chatStream.test.ts
```

Expected: all chatStream tests PASS (6 baseline + 1 new = 7).

- [ ] **Step 9: Run full Vitest to confirm no regressions**

```
npm run test:run
```

Expected: 48 PASSED (44 baseline + 3 sseParser + 1 chatStream).

- [ ] **Step 10: Commit**

```
git add frontend/src/playground/lib/sseParser.ts frontend/src/playground/lib/sseParser.test.ts frontend/src/playground/lib/chatStream.ts frontend/src/playground/lib/chatStream.test.ts
git commit -m "feat(playground): parse status SSE events + onStatus callback"
```

---

### Task 9: StatusBanner + PlaygroundPage + cap relax + rehype-raw

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/src/playground/components/StatusBanner.tsx`
- Create: `frontend/src/playground/components/StatusBanner.test.tsx`
- Modify: `frontend/src/playground/lib/fileValidate.ts`
- Modify: `frontend/src/playground/lib/fileValidate.test.ts`
- Modify: `frontend/src/playground/components/MessageBubble.tsx`
- Modify: `frontend/src/playground/components/ComposerBar.tsx`
- Modify: `frontend/src/playground/hooks/useChatStream.ts`
- Modify: `frontend/src/pages/PlaygroundPage.tsx`

- [ ] **Step 1: Install `rehype-raw`**

```
cd frontend
npm install rehype-raw@^7
```

Expected: `package.json` gains `"rehype-raw": "^7.x.x"` under `dependencies`. `package-lock.json` updates.

- [ ] **Step 2: Write the failing StatusBanner test**

Create `frontend/src/playground/components/StatusBanner.test.tsx`:

```tsx
import { render, screen, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { StatusBanner } from "./StatusBanner";

describe("StatusBanner", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("T5.5a — renders message with spinner when status is pending", () => {
    render(<StatusBanner status={{ message: "Captioning...", done: false }} />);
    expect(screen.getByText("Captioning...")).toBeInTheDocument();
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("T5.5b — calls onClear after 1500ms when done=true", () => {
    const onClear = vi.fn();
    render(
      <StatusBanner
        status={{ message: "Done", done: true }}
        onClear={onClear}
      />,
    );
    expect(onClear).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(1499));
    expect(onClear).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(1));
    expect(onClear).toHaveBeenCalledTimes(1);
  });

  it("T5.5c — replacing status before timeout cancels the prior timer", () => {
    const onClear = vi.fn();
    const { rerender } = render(
      <StatusBanner
        status={{ message: "Done A", done: true }}
        onClear={onClear}
      />,
    );
    act(() => vi.advanceTimersByTime(1000));
    rerender(
      <StatusBanner
        status={{ message: "Working B", done: false }}
        onClear={onClear}
      />,
    );
    act(() => vi.advanceTimersByTime(2000));
    // The first timer was cancelled by re-render; new status is non-done.
    expect(onClear).not.toHaveBeenCalled();
  });

  it("T5.5d — renders nothing when status is null", () => {
    const { container } = render(<StatusBanner status={null} />);
    expect(container.firstChild).toBeNull();
  });
});
```

- [ ] **Step 3: Run StatusBanner test to verify failure**

```
npm run test:run -- src/playground/components/StatusBanner.test.tsx
```

Expected: FAIL — `Cannot find module './StatusBanner'`.

- [ ] **Step 4: Implement `StatusBanner.tsx`**

Create `frontend/src/playground/components/StatusBanner.tsx`:

```tsx
import { useEffect, useRef } from "react";
import { Loader2 } from "lucide-react";

export type StatusBannerState = {
  message: string;
  done: boolean;
};

type Props = {
  status: StatusBannerState | null;
  /** Fires 1500ms after a `done:true` status arrives. The parent uses this
   * to dismiss the banner. */
  onClear?: () => void;
};

const AUTO_CLEAR_MS = 1500;

export function StatusBanner({ status, onClear }: Props) {
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    // On every status change: cancel any pending timer.
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (status?.done && onClear) {
      timerRef.current = window.setTimeout(() => {
        timerRef.current = null;
        onClear();
      }, AUTO_CLEAR_MS);
    }
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [status, onClear]);

  if (!status) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-2 mx-auto mt-3 px-3 py-1.5 rounded-full text-xs"
      style={{
        background: "#eff6ff",
        color: "#1e40af",
        border: "1px solid #bfdbfe",
        width: "fit-content",
        maxWidth: "90%",
      }}
    >
      {!status.done && (
        <Loader2 size={12} className="animate-spin" aria-hidden="true" />
      )}
      <span>{status.message}</span>
    </div>
  );
}
```

- [ ] **Step 5: Run StatusBanner test to verify pass**

```
npm run test:run -- src/playground/components/StatusBanner.test.tsx
```

Expected: 4 PASSED.

- [ ] **Step 6: Update `fileValidate.checkAttachmentCap` signature**

Replace `frontend/src/playground/lib/fileValidate.ts`:

```ts
import { FriendlyError } from "./errors";

export const ALLOWED_MIME = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
] as const;

export const MAX_BYTES = 10 * 1024 * 1024; // 10 MiB

/**
 * Per-turn image cap. The user can attach at most MAX_IMAGES images to the
 * tin nhắn currently being composed. History images are NOT counted —
 * Phase 5's image compressor strips them down to ≤1 pixel-bearing turn at
 * the backend, so a 50-message conversation can still accept a fresh
 * 4-image upload on the next turn.
 */
export const MAX_IMAGES = 4;

export function validateFile(f: File): void {
  if (!ALLOWED_MIME.includes(f.type as (typeof ALLOWED_MIME)[number])) {
    throw new FriendlyError("unsupported_mime", f.type || "unknown");
  }
  if (f.size === 0) throw new FriendlyError("empty_file");
  if (f.size > MAX_BYTES) throw new FriendlyError("too_large");
}

/** True if you can still add an attachment without breaking the per-turn cap. */
export function checkAttachmentCap(currentCount: number): boolean {
  return currentCount < MAX_IMAGES;
}
```

- [ ] **Step 7: Update fileValidate test**

Replace the existing `T2.10` test in `frontend/src/playground/lib/fileValidate.test.ts`. Find:

```ts
  it("T2.10 — rejects when current + history >= MAX_IMAGES", () => {
    expect(checkAttachmentCap(2, 2)).toBe(false);
    expect(checkAttachmentCap(3, 1)).toBe(false);
    expect(checkAttachmentCap(0, 4)).toBe(false);
    expect(checkAttachmentCap(2, 1)).toBe(true);
  });
```

Replace with:

```ts
  it("T2.10 (Phase 5) — rejects when current >= MAX_IMAGES (per-turn only)", () => {
    expect(checkAttachmentCap(0)).toBe(true);
    expect(checkAttachmentCap(3)).toBe(true);
    expect(checkAttachmentCap(4)).toBe(false);
    expect(checkAttachmentCap(5)).toBe(false);
  });
```

- [ ] **Step 8: Update `useChatStream` hook to forward `onStatus`**

Replace `frontend/src/playground/hooks/useChatStream.ts`:

```ts
import { useCallback, useRef } from "react";
import { streamChat, type ChatStreamCallbacks } from "../lib/chatStream";
import type { ChatMessageWithAttachments } from "../types";

export type SendArgs = ChatStreamCallbacks & {
  messages: ChatMessageWithAttachments[];
  modelId: string | null;
};

export function useChatStream(): {
  send: (args: SendArgs) => Promise<void>;
  abort: () => void;
} {
  const ctrlRef = useRef<AbortController | null>(null);

  const send = useCallback(async (args: SendArgs) => {
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    await streamChat({
      signal: ctrl.signal,
      messages: args.messages,
      modelId: args.modelId,
      onDelta: args.onDelta,
      onDone: args.onDone,
      onError: args.onError,
      onStatus: args.onStatus,
    });
  }, []);

  const abort = useCallback(() => {
    ctrlRef.current?.abort();
  }, []);

  return { send, abort };
}
```

- [ ] **Step 9: Add `rehype-raw` to MessageBubble plugin chain**

In `frontend/src/playground/components/MessageBubble.tsx`, find the existing import block:

```tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";
```

Add the new import (after `rehype-highlight`):

```tsx
import rehypeRaw from "rehype-raw";
```

Then find each `<ReactMarkdown ...>` in the file (currently just one in `AssistantBubble`) and update the `rehypePlugins` prop:

```tsx
            rehypePlugins={[rehypeRaw, rehypeHighlight]}
```

(Order matters: `rehypeRaw` must come BEFORE `rehypeHighlight` so the raw HTML is parsed before code blocks get tokenized.)

- [ ] **Step 10: Drop `historyImageCount` from ComposerBar**

In `frontend/src/playground/components/ComposerBar.tsx`:

Find:

```tsx
  historyImageCount: number;
```

Delete that line from the `ComposerBarProps` type.

Find:

```tsx
    historyImageCount,
```

Delete that destructured field from `function ComposerBar(props: ComposerBarProps) { const { ... } = props; ... }`.

Find every reference to `historyImageCount` in the file (there will be a `checkAttachmentCap(attachments.length, historyImageCount)` call inside `handleFiles`). Replace with `checkAttachmentCap(attachments.length + added)`. The relevant block becomes:

```tsx
      let added = 0;
      for (const f of files) {
        if (!checkAttachmentCap(attachments.length + added)) {
          toast.push(`Tối đa ${MAX_IMAGES} ảnh trong một tin nhắn.`, "error");
          break;
        }
```

Also update the Vietnamese in the toast string from `"Tối đa ${MAX_IMAGES} ảnh trong một cuộc trò chuyện."` (Phase 4) to `"Tối đa ${MAX_IMAGES} ảnh trong một tin nhắn."` (Phase 5 — per turn).

- [ ] **Step 11: Wire StatusBanner into PlaygroundPage**

In `frontend/src/pages/PlaygroundPage.tsx`:

Add to the imports near the top (after the existing `Toaster` import):

```tsx
import { StatusBanner, type StatusBannerState } from "../playground/components/StatusBanner";
```

Inside `function PlaygroundInner()`, after the existing `useState` calls (e.g., after `const [editingId, setEditingId] = useState<string | null>(null);` on line ~57), add:

```tsx
  const [status, setStatus] = useState<StatusBannerState | null>(null);
```

In the existing `runStream` function, add an `onStatus` callback to the `send({...})` call. Find the existing call:

```tsx
    await send({
      messages: wireMessages,
      modelId: effectiveModelId || null,
      onDelta: (delta) => ...,
      onDone: () => ...,
      onError: (e) => ...,
    });
```

Add inside the object literal (anywhere among the callbacks):

```tsx
      onStatus: (message, done) => setStatus({ message, done }),
```

Find `historyImageCount` plumbing:

```tsx
  const historyImageCount = useMemo(
    () =>
      messages.reduce((n, m) => n + (m.attachments?.length ?? 0), 0),
    [messages],
  );
```

Delete that block (no longer used).

Find the `<ComposerBar ... />` JSX and remove the `historyImageCount={historyImageCount}` prop from it.

Find the place where `<MessageList ... />` is rendered (a few lines above ComposerBar). Render `<StatusBanner ...>` IMMEDIATELY ABOVE the existing `<MessageList ...>`:

```tsx
            <StatusBanner status={status} onClear={() => setStatus(null)} />
            <MessageList
              messages={messages}
              actions={messageActions}
            />
```

Also add a `setStatus(null)` call inside `selectConversation` (alongside the existing `setText("")` etc.) and inside `handleStop`:

```tsx
  function selectConversation(id: string) {
    abort();
    if (active) {
      const streaming = active.messages.find((m) => m.status === "streaming");
      if (streaming) {
        dispatch({
          type: "MARK_STOPPED",
          conversationId: active.id,
          messageId: streaming.id,
        });
      }
    }
    dispatch({ type: "SELECT_CONVERSATION", id });
    setText("");
    setAttachments([]);
    setEditingId(null);
    setStatus(null);     // <-- add
  }

  function handleStop() {
    if (!activeId) return;
    abort();
    setStatus(null);     // <-- add
    const streamingMsg = [...messages].reverse().find((m) => m.status === "streaming");
    if (streamingMsg) {
      dispatch({
        type: "MARK_STOPPED",
        conversationId: activeId,
        messageId: streamingMsg.id,
      });
    }
  }
```

- [ ] **Step 12: Run all frontend Vitest**

```
npm run test:run
```

Expected: 52 PASSED (48 + 4 StatusBanner). The fileValidate suite stays at 7 (one renamed; total unchanged).

If `tsc` complains about unused `historyImageCount` parameters that you missed in PlaygroundPage / ComposerBar, fix those compile errors first; then re-run tests.

- [ ] **Step 13: TypeScript clean check**

```
npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 14: Commit**

```
git add frontend/package.json frontend/package-lock.json frontend/src/playground/components/StatusBanner.tsx frontend/src/playground/components/StatusBanner.test.tsx frontend/src/playground/lib/fileValidate.ts frontend/src/playground/lib/fileValidate.test.ts frontend/src/playground/hooks/useChatStream.ts frontend/src/playground/components/MessageBubble.tsx frontend/src/playground/components/ComposerBar.tsx frontend/src/pages/PlaygroundPage.tsx
git commit -m "feat(playground): add StatusBanner + relax cap to per-turn + rehype-raw"
```

---

### Task 10: Playwright E2E (E5.1, E5.2)

**Files:**
- Modify: `frontend/tests/e2e/fixtures/sseFixture.ts`
- Create: `frontend/tests/e2e/playground-compressor.spec.ts`

- [ ] **Step 1: Extend the fixtures with `mockChatStreamWithStatus`**

Append to `frontend/tests/e2e/fixtures/sseFixture.ts`:

```ts
/**
 * Phase 5: emit a sequence of status events, then a single thinking-log
 * delta, then the model deltas, then done.
 */
export async function mockChatStreamWithStatus(
  page: Page,
  statuses: Array<{ message: string; done: boolean }>,
  thinkingMd: string,
  deltas: string[],
) {
  await page.route("**/api/chat/stream", (route: Route) => {
    const parts: string[] = [];
    for (const s of statuses) {
      parts.push(
        `data: ${JSON.stringify({ type: "status", message: s.message, done: s.done })}\n\n`,
      );
    }
    if (thinkingMd) {
      parts.push(
        `data: ${JSON.stringify({ delta: thinkingMd, done: false })}\n\n`,
      );
    }
    for (const d of deltas) {
      parts.push(
        `data: ${JSON.stringify({ delta: d, done: false })}\n\n`,
      );
    }
    parts.push(`data: ${JSON.stringify({ delta: "", done: true })}\n\n`);
    route.fulfill({
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
      body: parts.join(""),
    });
  });
}
```

- [ ] **Step 2: Write the E2E spec**

Create `frontend/tests/e2e/playground-compressor.spec.ts`:

```ts
import { test, expect } from "@playwright/test";
import {
  mockModels,
  mockFileUploads,
  mockChatStreamWithStatus,
} from "./fixtures/sseFixture";

test.describe("Playground compressor (Phase 5)", () => {
  test("E5.1 — status banner appears then auto-clears", async ({ page }) => {
    await mockModels(page);
    await mockFileUploads(page);
    await mockChatStreamWithStatus(
      page,
      [
        { message: "🖼️ Captioning 1 new image(s)...", done: false },
        { message: "✅ Compressor done", done: true },
      ],
      "",
      ["Hello world."],
    );
    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    await page.locator("textarea").fill("hi");
    await page.getByLabel("Gửi").click();

    // Banner appears with the captioning message.
    await expect(
      page.getByText("🖼️ Captioning 1 new image(s)..."),
    ).toBeVisible({ timeout: 5000 });
    // Banner auto-clears after the done event (1.5s grace).
    await expect(page.getByText("Hello world.")).toBeVisible({ timeout: 5000 });
    await expect(
      page.getByText("🖼️ Captioning 1 new image(s)..."),
    ).not.toBeVisible({ timeout: 4000 });
  });

  test("E5.2 — thinking log <details> renders and expands", async ({ page }) => {
    await mockModels(page);
    await mockFileUploads(page);
    const thinkingMd =
      "<details><summary>🧠 Image compressor reasoning (1 ảnh, 1 caption mới, kept new upload)</summary>\n\n" +
      "**Step 1 — Image scan**\n- Tổng 1 ảnh; cache miss: 1, hit: 0\n\n" +
      "</details>\n\n";
    await mockChatStreamWithStatus(
      page,
      [{ message: "✅ Compressor done", done: true }],
      thinkingMd,
      ["Câu trả lời cho ảnh."],
    );
    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    await page.locator("textarea").fill("describe");
    await page.getByLabel("Gửi").click();

    // Summary text renders.
    await expect(
      page.getByText(/Image compressor reasoning/),
    ).toBeVisible({ timeout: 5000 });
    // Closed by default — Step 1 text not visible.
    await expect(page.getByText(/cache miss: 1/)).not.toBeVisible();

    // Click summary to expand.
    await page.getByText(/Image compressor reasoning/).click();
    await expect(page.getByText(/cache miss: 1/)).toBeVisible();
    // Final answer also rendered.
    await expect(page.getByText("Câu trả lời cho ảnh.")).toBeVisible();
  });
});
```

- [ ] **Step 3: Run Playwright**

```
cd frontend
npm run test:e2e
```

Expected: 8 tests pass total — 2 from `playground.spec.ts` (Phase 2) + 4 from `playground-controls.spec.ts` (Phase 3) + 0 explicit Phase 4 file (E4 was bundled into the controls spec or a separate file — adjust expected count to whatever your tree shows on `dev/AnKun10` HEAD) + **2 NEW** from `playground-compressor.spec.ts`.

If the banner-disappears assertion (`E5.1`) is flaky because the model deltas race in faster than 1.5s, bump the timeout from `4000` to `6000`. If the `<details>` block expands by default (browser-dependent), the spec already targets it via summary click which is idempotent — no change needed.

- [ ] **Step 4: Commit**

```
git add frontend/tests/e2e/fixtures/sseFixture.ts frontend/tests/e2e/playground-compressor.spec.ts
git commit -m "test(playground): add E2E for compressor status banner + thinking log"
```

---

### Task 11: Manual smoke + final pass

**Files:** none modified — verification only.

- [ ] **Step 1: Run the full Vitest suite**

```
cd frontend
npm run test:run
```

Expected: 52 PASSED.

- [ ] **Step 2: Run the full Playwright suite**

```
npm run test:e2e
```

Expected: ALL pass (Phase 2 + 3 + 4 + the 2 new Phase 5 tests).

- [ ] **Step 3: Run the full backend pytest suite**

```
cd ../backend
. .venv/bin/activate    # or PowerShell equivalent
pytest -q
```

Expected: 96 PASSED.

- [ ] **Step 4: TypeScript check**

```
cd ../frontend
npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 5: Manual browser smoke (with backend + vLLM up)**

Start backend:
```
cd backend
. .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

The startup log MUST contain (in order):
```
[VLMManager] Loaded N model(s): [...]
[lifespan] compressor enabled (caption=qwen3-vl-8b-vllm, router=qwen3-vl-8b-vllm)
```

Start frontend:
```
cd frontend
npm run dev
```

Open `http://localhost:5173/playground`. Walk through each scenario; if any fails, return to the relevant earlier task, write a regression test, fix, re-run all tests.

**A. Cold-start with images (covers A5.4 cache hit, A5.7 SSE buffering)**
1. Upload 2 distinct images in one turn + send "describe these".
2. DevTools Network tab → `/api/chat/stream` → EventStream pane shows status events arriving over time (not in one batch).
3. Status banner shows "🖼️ Captioning 2 new image(s)..." then "✅ Compressor done".
4. Final assistant bubble has a collapsed `<details>` block above the answer; click → expands, shows captions and decision.
5. Click "Tạo lại" → new round trip → DevTools shows ZERO calls to caption logic (since cache is warm). Banner shows only "✅ Compressor done" (or skips entirely on the 0-miss fast path).

**B. Restart-survives (covers A5.5)**
1. With (A) done, stop uvicorn (Ctrl+C). Confirm `backend/data/img_captions.db` exists on disk.
2. Restart uvicorn. Send another regenerate of the same conversation → cache hits as before (no caption calls).

**C. Edit-truncate preserves cache (covers A5.6)**
1. In a 2-image conversation, click "Chỉnh sửa" on the first user message → save → next stream uses the same hashes → cache hits → 0 caption calls.

**D. Per-upload cap (covers A5.10)**
1. Send 4 images in a turn. Wait for response. In the next turn, attach 4 MORE images. Composer accepts all 4 (per-turn cap, history not counted).
2. Try to attach a 5th in the same turn → toast "Tối đa 4 ảnh trong một tin nhắn."

**E. Compressor disabled fallback (covers A5.11 + R23 mitigation)**
1. Stop the server. Edit `backend/app/models/vlm/models.yaml` to remove the `compressor:` block (comment it out).
2. Restart. Log says: `[lifespan] compressor disabled (no compressor: yaml block)`.
3. Send a message with 1 image → no status banner, no thinking-log block, plain answer streams. Behavior identical to Phase 4.
4. Restore the `compressor:` block.

**F. SSE forward-compat (visual confirmation only)**
1. With the compressor running, observe that the unknown-shape fallback in `sseParser` doesn't break anything in DevTools. (No specific assertion — just confirm no `[chatStream] parse_error` in console.)

- [ ] **Step 6: No commit unless fixes were applied**

```
git status
```

Expected: clean working tree.

---

## Coverage Mapping

| Test ID | Description | Task |
|---------|-------------|------|
| T5.B1 | CaptionCache round-trip + get_many partial + put_many INSERT OR IGNORE | Task 3 |
| T5.B2 | iter_image_parts / has_images / find_latest_image_turn / text_of helpers | Task 4 |
| T5.B3 | rewrite_messages: keep_idx semantics + immutable input | Task 4 |
| T5.B4 | hash_image_url: data URL decoded; bad scheme raises | Task 4 |
| T5.B5 | engine.caption_one returns trimmed caption + uses caption max_tokens | Task 5 |
| T5.B6 | engine.route happy + JSON parse fail-open | Task 5 |
| T5.B7 | engine.compress: with images + latest-has-images + no-images fast path | Task 6 |
| T5.B8 | chat_stream integration: status SSE + thinking-log delta + model deltas | Task 7 |
| T5.B9 | chat_stream no-image fast path: zero status events | Task 7 |
| A5.B1 | ensure_captions: per-image fail-open omits failed url, others succeed | Task 5 |
| A5.B2 | route fail-open on JSON parse error (covered by T5.B6 second case) | Task 5 |
| A5.B3 | concurrent put_many: INSERT OR IGNORE leaves one row | Task 3 |
| A5.B4 | hash_image_url for relative path → fetched via webui_internal_base | Task 4 |
| A5.B5 | engine.compress self-catches all exceptions → passthrough result | Task 6 |
| A5.B6 | ensure_captions dedup-by-hash: 2 different urls, same hash → 1 caption call | Task 5 |
| T5.1 | sseParser parses {type:"status", message, done:false} | Task 8 |
| T5.2 | sseParser parses {type:"status", message, done:true} | Task 8 |
| T5.3 | sseParser ignores unknown {type:"future_unknown"} (forward-compat) | Task 8 |
| T5.4 | chatStream invokes onStatus callback when status events arrive | Task 8 |
| T5.5 | StatusBanner render + auto-dismiss-on-done with fake timers | Task 9 |
| E5.1 | Status banner appears + clears (Playwright) | Task 10 |
| E5.2 | Thinking-log `<details>` renders + expands (Playwright) | Task 10 |
| A5.1 | Caption call fails for one image → image kept | Covered by A5.B1 |
| A5.2 | Router fails → failopen_keep=True → images preserved | Covered by T5.B6 |
| A5.3 | Cache write contention: INSERT OR IGNORE | Covered by A5.B3 |
| A5.4 | Cache hit on regenerate → 0 caption calls | Task 11 manual smoke A.5 |
| A5.5 | DB file survives backend restart | Task 11 manual smoke B |
| A5.6 | Edit-truncate preserves cached captions | Task 11 manual smoke C |
| A5.7 | SSE proxy buffering: status events arrive over time | Task 11 manual smoke A.2 |
| A5.8 | <details> collapsed-by-default + click expands | Task 10 (E5.2) + Task 11 manual smoke A.4 |
| A5.9 | XSS check: raw `<script>` in markdown does NOT execute | Deferred — see Risks (R22). Not added to Phase 5 because we never echo user-supplied HTML. |
| A5.10 | Per-upload cap relax: 4-images-per-turn allowed irrespective of history | Task 11 manual smoke D |
| A5.11 | Compressor disabled (no yaml block) → behavior identical to Phase 4 | Task 7 (`test_A5_11_compressor_disabled_passthrough`) + Task 11 manual smoke E |

A5.9 is intentionally NOT exercised in Phase 5 because the only HTML in `msg.text` is the engine's own `<details>` block — the model never produces user-supplied raw HTML in our pipeline. If a future phase introduces user-authored markdown that may contain HTML, revisit.

---

## Risks & deferrals (Phase 5)

| ID | Risk | Mitigation in this plan |
|----|------|------------------------|
| R5.1 | Caption call latency on cold cache (~3-5s/image × 4 in parallel = 12-20s) blocks first model token | `asyncio.gather` parallelizes (Task 5 `ensure_captions`). Status banner gives real-time feedback (Task 9). Cache hits make subsequent turns instant. Document expected cold-start latency in README (deferred to docs commit if desired). |
| R5.2 | aiosqlite WAL is single-writer; multi-uvicorn-worker would briefly block on writes | Document single-worker deploy. If horizontal scale needed → switch to Postgres or per-process cache (loses dedup). |
| R5.3 | Caption hallucination — captioner LLM may invent details not present in image | System prompt explicitly constrains ("KHÔNG suy diễn cảm xúc, KHÔNG bịa") in `prompts.py`. Caption sloppy → answer sloppy; same risk as the reference filter. |
| R5.4 | Router LLM returns non-JSON despite system-prompt instruction | `engine.route` catches `JSONDecodeError` → falls back to `router_failopen_keep`. Tested by Task 5 (`test_T5B6_route_non_json_fails_open_keep`). |
| R5.5 | SSE buffering between Vite proxy and uvicorn could batch status events | `StreamingResponse` with `text/event-stream` is chunked by uvicorn; Vite passes through. Verified by Task 11 manual smoke A.2. |
| R5.6 | `rehype-raw` allowing raw HTML opens XSS surface if model output ever contains `<script>` | Backend never echoes user-supplied HTML; only the engine's own `<details>` block. Re-test if a future phase adds user-authored markdown that may contain HTML. |
| R5.7 | Conversations from Phase 4 (pre-compressor) reload with >4 images → first compressor run hammers caption LLM | One-time cost per unique image. Cache fills; subsequent runs hit. User can opt out via `compressor:` block removal (Task 7 step 5 demonstrates the path). |

**Defer to a possible Phase 6:**
- User valves UI toggles (force_keep_all_images, disable per-conversation). Hardcoded always-on in this phase.
- Cache eviction policy (TTL / max-size). Manual `DELETE FROM captions WHERE created_at < ?` for now.
- Smart cache-aware safety net (read caption from cache when `enforce_image_cap` strips). Stays dumb in this phase.
- Separate (smaller) text-only router model. Defaults to the same Qwen3-VL model as the captioner.
- A5.9 explicit XSS regression test if/when user-authored HTML enters `msg.text`.
