# AI Playground — Qwen-VL Parity Design

## Problem

The AI Playground at `/playground` is a lightweight React chat that calls a sync FastAPI backend for non-streaming completions, supports a single image per turn (sent inline as a data URI), drops images from history when constructing prompts, renders assistant text as plain pre-wrapped strings, and stores conversations only in React state. Open WebUI's Qwen-VL chat — used as the reference for "what good feels like" — streams tokens, supports multi-image upload via a file service, preserves multimodal history across turns, renders markdown with code highlighting, and persists conversations.

The gap is wide enough that small ad-hoc improvements would compound into inconsistency. We need one coherent design.

## Goal

Bring the AI Playground to functional parity with Open WebUI's Qwen-VL chat for the **scope tier** the user selected (Tier B — parity + UX core). After this work, a user opening `/playground`:

- Picks a vision-capable model and sees streaming text token-by-token.
- Uploads up to 4 images per request (drag/drop, paste, or picker), with images preserved across turns.
- Reads markdown-rendered assistant replies with syntax-highlighted code blocks.
- Stops, regenerates, or edits messages with predictable linear semantics.
- Reloads the page and finds their conversations intact.

Without ballooning into a full Open WebUI clone (no settings panel, no message tree, no math rendering, no auth, no DB, no tool calling).

## Scope

### In scope (v1, Tier B)

- SSE streaming end-to-end (vLLM → backend forward → frontend token render).
- File upload service: `POST /api/files` + `GET /api/files/{id}`, server resolves IDs to base64 at chat time.
- Multi-image input (≤ 4 cap including history), drag/drop + paste + picker.
- Multimodal history preserved across turns (image_url parts kept in assistant context).
- Markdown rendering with code highlighting (react-markdown + remark-gfm + rehype-highlight).
- Vision capability flag in `models.yaml` exposed via `/api/models`; frontend gates image attachment on it.
- Stop generation via AbortController + backend client-disconnect handling.
- Regenerate last assistant reply.
- Edit user message (linear: truncate everything after the edit, then resend).
- localStorage persistence for conversations and messages (file IDs only, not data URIs).

### Out of scope (defer)

- Per-conversation settings UI (temperature, max_tokens, top_p, system prompt override).
- Math rendering (KaTeX).
- Client-side image compression.
- Token counter, conversation export, search, keyboard shortcuts.
- Message-tree branching (siblings on edit/regenerate).
- Tool calling / function calling.
- File TTL or cleanup cron.
- Auth, rate limiting, multi-user collaboration.
- Server-side conversation DB.
- Mobile-first responsive redesign.
- i18n (keep Vietnamese hardcoded).

## Architecture

### Streaming chat flow

```
PlaygroundPage (React)
  └─ user composes msg + N file IDs
     └─ fetch POST /api/chat/stream { messages, model_id }
        └─ FastAPI async handler
           ├─ build_openai_messages: resolve attachments → base64 data URI
           ├─ ImageCompressorEngine.compress(messages, model_id)         ← Phase 5
           │  ├─ hash + cache lookup per image; caption misses (parallel)
           │  ├─ if latest turn text-only with prior images → router LLM call
           │  ├─ rewrite_messages: strip pixels except keep_idx → "[Past image #N: caption]"
           │  └─ yields StatusEvents → handler emits SSE {type:"status",...}
           │     yields CompressionResult.thinking_md → handler prepends as 1st delta
           ├─ enforce_image_cap (safety net, max=4): only kicks in if compressor passthrough
           ├─ VLMManager.stream(model_id, rewritten_messages)
           │  └─ QwenVLLMProvider.stream
           │     ├─ apply transforms (strip_image_tokens, inject_pixel_bounds)
           │     └─ AsyncOpenAI client.chat.completions.create(stream=True)
           │        → async iterator of ChatCompletionChunk
           ├─ wrap chunks → SSE: data: {"delta":"...","done":false}\n\n
           └─ on client disconnect → break loop → cleanup async generator
PlaygroundPage
  └─ ReadableStream reader → sseParser → routes by event shape:
      ├─ {delta} → reducer APPEND_DELTA
      ├─ {type:"status", message, done} → setStatus (transient banner, not persisted)
      ├─ {done:true} → reducer MARK_DONE
      └─ {error} → reducer MARK_ERROR
```

### File upload flow

```
user picks/drops/pastes image
  └─ client validate MIME + size
     └─ POST /api/files multipart
        ├─ server validate Content-Type ∈ ALLOWED_MIME
        ├─ stream-read body, abort if > MAX_UPLOAD_BYTES
        ├─ PIL magic-byte verify
        ├─ uuid4().hex → write images/<id>.<ext> atomically
        └─ return { id, url: "/api/files/<id>", mime, size, original_name }
  ← attachments[] state
  └─ on send: include attachment.id in chat request
  └─ on edit/regenerate: file IDs stay valid (file disk-resident)
```

### Persistence flow (localStorage)

```
state ConversationsState {
  schemaVersion: 1
  conversations: Record<id, Conversation>
  activeId: string | null
}
Conversation {
  id, title, modelId, messages: Message[], createdAt, updatedAt
}
Message {
  id, role: "user"|"assistant", text,
  attachments?: AttachmentRef[],
  status?: "streaming"|"done"|"stopped"|"error",
  errorKind?: "connection"|"file_missing"|"internal"
}
AttachmentRef { id, url, mime, originalName }
```

On reload: hydrate from localStorage; coerce `streaming` → `stopped`; render `<img src={attachment.url}>`; on 404 fallback to "Ảnh đã hết hạn" placeholder.

## Backend Design

### B.1 Async migration

`providers/base.py` — add `stream()`, keep `generate()` as a wrapper for backward compat:

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator

class VLMProvider(ABC):
    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        """Yield text deltas as they arrive from upstream."""
        ...

    async def generate(self, messages, max_tokens, temperature) -> str:
        chunks = []
        async for delta in self.stream(messages, max_tokens, temperature):
            chunks.append(delta)
        return "".join(chunks).strip()
```

`providers/openai_compatible.py` — swap `OpenAI` → `AsyncOpenAI`, set `stream=True`:

```python
async def stream(self, messages, max_tokens, temperature):
    try:
        result = await self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=True,
            **self._token_kwargs(max_tokens),
            temperature=temperature,
        )
        async for chunk in result:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
    except APIConnectionError as exc:
        raise ConnectionError(str(exc)) from exc
```

`providers/qwen_vllm/provider.py` — apply transforms before stream; retry only pre-first-token:

```python
async def stream(self, messages, max_tokens, temperature):
    transformed = inject_pixel_bounds(
        strip_image_tokens(messages), self._min_pixels, self._max_pixels,
    )
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await self._client.chat.completions.create(
                model=self.model_id, messages=transformed,
                stream=True, max_tokens=max_tokens, temperature=temperature,
            )
            async for chunk in result:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
            return
        except APIConnectionError as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_BACKOFF_S)
                continue
            raise ConnectionError(
                f"Cannot reach {self.model_id} at {self._base_url}: {exc}"
            ) from exc
```

Retry only fires before the first chunk yields. After any chunk has reached the client, the iterator simply propagates upstream errors as `ConnectionError`, which the route handler turns into an SSE `error` event.

`manager.py` — add `async def stream(model_id, messages)` mirroring `generate`: prepend system prompt, look up provider, forward.

### B.2 File upload service

`backend/app/services/files.py` (new):

```python
ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
EXT_BY_MIME = {
    "image/png": "png", "image/jpeg": "jpg",
    "image/webp": "webp", "image/gif": "gif",
}
IMAGES_DIR = Path("images")
FILE_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")

class StoredFile(BaseModel):
    """JSON response is camelCase (`originalName`) to match the frontend
    `AttachmentRef` type — Pydantic field stays snake_case for Python idiom."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str
    url: str
    mime: str
    size: int
    original_name: str

def store_upload(upload: UploadFile) -> StoredFile:
    # 1. validate Content-Type ∈ ALLOWED_MIME → 415 otherwise
    # 2. stream-read into BytesIO, abort + 413 if exceeds MAX_UPLOAD_BYTES
    # 3. PIL.Image.open(buffer).verify() to validate magic bytes → 400 on fail
    # 4. file_id = uuid4().hex; ext = EXT_BY_MIME[upload.content_type]
    # 5. write images/<id>.tmp atomically, then os.rename to images/<id>.<ext>
    # 6. return StoredFile(id, url=f"/api/files/{file_id}", mime, size, original_name=upload.filename)

def open_image_bytes(file_id: str) -> tuple[bytes, str] | None:
    if not FILE_ID_PATTERN.match(file_id):
        return None
    for mime, ext in EXT_BY_MIME.items():
        path = IMAGES_DIR / f"{file_id}.{ext}"
        if path.exists():
            return path.read_bytes(), mime
    return None
```

`backend/app/routers/files.py` (new):

```python
router = APIRouter(prefix="/api", tags=["files"])

@router.post("/files", response_model=StoredFile)
async def upload_file(file: UploadFile = File(...)):
    return store_upload(file)

@router.get("/files/{file_id}")
async def get_file(file_id: str):
    if not FILE_ID_PATTERN.match(file_id):
        raise HTTPException(400, "Invalid file id")
    result = open_image_bytes(file_id)
    if result is None:
        raise HTTPException(404, "File not found")
    data, mime = result
    return Response(content=data, media_type=mime)
```

The `original_name` is returned to the client for display only; the on-disk path uses the UUID exclusively. No part of the user-supplied filename ever reaches the file system.

### B.3 Streaming chat endpoint

`backend/app/routers/chat.py` (modified):

```python
class Attachment(BaseModel):
    id: str

class ChatMessageWithAttachments(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    attachments: list[Attachment] = []

class ChatStreamRequest(BaseModel):
    messages: list[ChatMessageWithAttachments]
    model_id: str | None = None

@router.post("/chat/stream")
async def chat_stream(request: Request, body: ChatStreamRequest):
    manager = request.app.state.vlm_manager

    async def event_stream():
        try:
            openai_messages = build_openai_messages(body.messages)
            openai_messages = enforce_image_cap(openai_messages, max_images=4)
        except FileNotFoundError as exc:
            yield _sse_error("file_missing", str(exc))
            return

        try:
            async for delta in manager.stream(body.model_id, openai_messages):
                if await request.is_disconnected():
                    return
                yield _sse_delta(delta)
            yield _sse_done()
        except ConnectionError as exc:
            yield _sse_error("connection", str(exc))
        except BadRequestError as exc:
            yield _sse_error("bad_request", str(exc))
        except Exception:
            traceback.print_exc()
            yield _sse_error("internal", "Internal error")

    return StreamingResponse(event_stream(), media_type="text/event-stream")

def _sse_delta(delta: str) -> str:
    return f"data: {json.dumps({'delta': delta, 'done': False})}\n\n"

def _sse_done() -> str:
    return f"data: {json.dumps({'delta': '', 'done': True})}\n\n"

def _sse_error(kind: str, message: str) -> str:
    return f"data: {json.dumps({'error': kind, 'message': message})}\n\n"
```

The legacy `POST /api/chat` non-stream endpoint is kept (used by `scripts/smoke_qwen3_vl.py`) but rewritten to delegate to the same async path: it `await`s `manager.generate()` (the wrapper) and returns the gathered string.

#### SSE event format

| Kind | Payload |
|------|---------|
| Delta | `data: {"delta":"...","done":false}\n\n` |
| Done | `data: {"delta":"","done":true}\n\n` |
| Error | `data: {"error":"connection|file_missing|bad_request|internal","message":"..."}\n\n` |

Single-line JSON in `data:` frames. No event types, no comments — keeps the parser straightforward.

#### Disconnect handling

Between yields, `request.is_disconnected()` is polled. When the client aborts (user clicks Stop, navigates away, network drops):

1. The polling returns `True`, the generator returns.
2. FastAPI cleans up the `event_stream` async generator.
3. The async generator's reference to the upstream `result` iterator is dropped.
4. `AsyncOpenAI` closes its HTTP session; vLLM receives the TCP RST and stops generation.

This relies on vLLM ≥ 0.5 honoring upstream cancellation. Verified via manual smoke test in Phase 1 (R1 in Risks).

### B.4 Image resolution + cap policy

`backend/app/services/messages.py` (new):

```python
def build_openai_messages(msgs: list[ChatMessageWithAttachments]) -> list[dict]:
    out = []
    for m in msgs:
        if not m.attachments:
            out.append({"role": m.role, "content": m.text})
            continue
        parts = []
        for att in m.attachments:
            data = open_image_bytes(att.id)
            if data is None:
                raise FileNotFoundError(f"Attachment {att.id} not found")
            blob, mime = data
            data_uri = f"data:{mime};base64,{base64.b64encode(blob).decode()}"
            parts.append({"type": "image_url", "image_url": {"url": data_uri}})
        if m.text:
            parts.append({"type": "text", "text": m.text})
        out.append({"role": m.role, "content": parts})
    return out

PLACEHOLDER = "[ảnh trong lượt trước đã được lược bỏ do giới hạn 4 ảnh]"

def enforce_image_cap(messages: list[dict], max_images: int = 4) -> list[dict]:
    """Drop oldest image_url parts when total > max_images.
    Replace dropped parts with a text placeholder so message shape stays valid.
    If a message ends up with no content parts, replace its content with the
    placeholder string (avoid empty content arrays which vLLM rejects).
    """
    # Implementation walks forward, counts images; when count > max, replaces
    # earliest image_url parts with text placeholders until count == max.
    # Coalesces adjacent text placeholders within a message.
```

### B.5 Models endpoint

`models.yaml`:

```yaml
- id: "qwen3-vl-8b-vllm"
  ...
  capabilities:
    vision: true

- id: "gpt-5.4-mini"
  ...
  capabilities:
    vision: false
```

`manager.list_models()` reads the `capabilities` block (defaults to `{"vision": False}` if absent) and returns:

```json
{
  "models": [
    {"id": "qwen3-vl-8b-vllm", "name": "Qwen3-VL 8B (vLLM)",
     "capabilities": {"vision": true}}
  ]
}
```

### B.6 Error matrix

| Endpoint | Condition | Status | Body |
|----------|-----------|--------|------|
| `POST /api/files` | MIME not in whitelist | 415 | `{"detail":"Unsupported media type"}` |
| `POST /api/files` | size > 10MB | 413 | `{"detail":"File too large"}` |
| `POST /api/files` | bytes don't decode as image | 400 | `{"detail":"Not a valid image"}` |
| `POST /api/files` | empty file | 400 | `{"detail":"Not a valid image"}` |
| `GET /api/files/{id}` | id fails regex | 400 | `{"detail":"Invalid file id"}` |
| `GET /api/files/{id}` | file missing | 404 | `{"detail":"File not found"}` |
| `POST /api/chat/stream` | model_id unknown | 200 | SSE `error: bad_request` (manager falls through to default; if default also missing, error) |
| `POST /api/chat/stream` | attachment id missing | 200 | SSE `error: file_missing` (no provider call) |
| `POST /api/chat/stream` | vLLM unreachable pre-first-chunk | 200 | SSE `error: connection` (after retries) |
| `POST /api/chat/stream` | vLLM 500 mid-stream | 200 | partial deltas + SSE `error: internal` |
| `POST /api/chat/stream` | vLLM 400 (token cap, etc.) | 200 | SSE `error: bad_request` with vLLM message |

### B.7 Image-aware compressor (Phase 5)

Port of `open-webui/vast-templates/qwen3-vl-8b/functions/qwenvl_image_compress.py`. Lives at `backend/app/services/image_compressor/`. Replaces dumb `enforce_image_cap` truncation with caption-aware history compression: every image in history gets a Vietnamese caption (cached by SHA-256 of image bytes), and a router LLM decides whether the *current* user turn needs pixels or whether captions alone suffice. Result: conversations can have arbitrarily many images while sending ≤ 1 pixel-bearing turn to the model.

**Module layout:**

```
backend/app/services/image_compressor/
├── __init__.py        # re-exports ImageCompressorEngine
├── cache.py           # CaptionCache (aiosqlite, WAL, keyed by sha256)
├── prompts.py         # CAPTION_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT (VN)
├── types.py           # StatusEvent, CompressionResult dataclasses
├── messages.py        # iter_image_parts, find_latest_image_turn, text_of, has_images, rewrite_messages, hash_image_url
└── engine.py          # ImageCompressorEngine.compress() orchestrator
```

**Cache schema** (mirrors reference exactly):

```sql
CREATE TABLE IF NOT EXISTS captions (
    img_hash    TEXT PRIMARY KEY,
    caption     TEXT NOT NULL,
    model       TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    bytes_size  INTEGER,
    user_id     TEXT
);
CREATE INDEX IF NOT EXISTS idx_created ON captions(created_at);
```

`PRAGMA journal_mode = WAL`, `synchronous = NORMAL`, `busy_timeout = 5000`. Default path: `backend/data/img_captions.db` (created on first run; `data/` added to `.gitignore`).

**Cache API** (`cache.py`):

```python
class CaptionCache:
    async def init() -> None
    async def get(h: str) -> Optional[str]
    async def get_many(hashes: list[str]) -> dict[str, str]
    async def put(h, caption, model, bytes_size=None, user_id=None) -> None
    async def put_many(items: list[tuple[str, str, str, Optional[int], Optional[str]]]) -> None
```

`put_many` uses `INSERT OR IGNORE` so concurrent puts of the same hash are safe (last-writer wins is irrelevant — all writers wrote the same caption for the same hash).

**Helpers** (`messages.py`):

```python
def has_images(msg: dict) -> bool
def iter_image_parts(msgs: list[dict]) -> Iterator[tuple[int, int, str]]
def find_latest_image_turn(msgs: list[dict]) -> Optional[int]
def text_of(msg: dict) -> str
async def hash_image_url(url: str, fetch_base: str, fetch_timeout_s=10) -> tuple[str, bytes]
def rewrite_messages(msgs, keep_idx, captions_by_url) -> list[dict]
```

`hash_image_url` supports `data:`, absolute `http(s)://`, and relative paths (resolved against `fetch_base`, defaults to `http://127.0.0.1:8000` in dev). Returns `(sha256_hex, raw_bytes)`. `rewrite_messages` deep-copies, strips `image_url` parts at every turn except `keep_idx`, and appends `[Past image #N: <caption>]` text to the message's content (coalescing into the existing trailing text part if any).

**Engine** (`engine.py`):

```python
# Type aliases (defined in types.py):
# Scanned = tuple[int, int, str, str, bytes]  # (msg_idx, content_idx, url, sha256, raw_bytes)
# StatusEvent: dataclass(message: str, done: bool = False)
# CompressionResult: dataclass(messages: list[dict], thinking_md: str)
# CompressionEvent = StatusEvent | CompressionResult

class ImageCompressorEngine:
    def __init__(
        self, cache: CaptionCache, vlm_manager: VLMManager,
        caption_model_id: str, router_model_id: str,
        webui_internal_base: str = "http://127.0.0.1:8000",
        caption_max_tokens: int = 80, router_max_tokens: int = 60,
        caption_timeout_s: int = 30, router_timeout_s: int = 15,
        router_failopen_keep: bool = True,
    ): ...

    async def caption_one(self, data_url: str) -> str
    async def route(self, user_text: str, captions: list[str]) -> tuple[bool, str]
    async def ensure_captions(self, scanned: list[Scanned]) -> dict[str, str]

    async def compress(self, messages: list[dict]) -> AsyncIterator[CompressionEvent]:
        """Yield zero or more StatusEvent values, then exactly one terminal CompressionResult.

        Wrapped in an outer try/except: ANY exception inside the body is caught,
        logged via `log.exception`, and the generator yields a passthrough
        CompressionResult(messages=messages, thinking_md='') and returns.
        Callers don't see exceptions from this method.
        """
```

`caption_one` and `route` are implemented by calling `self.vlm_manager.generate(model_id, messages, max_tokens, temperature)` — the same path used by the existing legacy `/api/chat` endpoint. Reusing the manager (vs httpx in the reference) means provider-specific transforms (Qwen pixel-bound injection) apply automatically.

`compress()` is an async generator. Sequence:

1. Scan history with `iter_image_parts`. If 0 images → yield `CompressionResult(messages, thinking_md='')` and return (fast path, 0 LLM calls).
2. Hash each image URL. Skip URLs that fail to hash (log warning, continue). If 0 hashable → return passthrough.
3. Cache lookup with `get_many`. Compute `misses = [s for s in scanned if s.hash not in cache]`.
4. If `misses`: yield `StatusEvent(message=f"🖼️ Captioning {len(misses)} new image(s)...")`.
5. Run captioner via `asyncio.gather(caption_one(url) for url in misses)`. Each individual failure → that url omitted from `captions_by_url` (fail-open per image). Successes → `cache.put_many(...)`.
6. Decide `keep_idx`:
   - If latest user turn has images → `keep_idx = latest_idx` (decision_label `"kept new upload"`).
   - Else (latest turn text-only): yield `StatusEvent("🧭 Routing: do we need pixels?")`, run `route(user_text, captions_for_latest)`. Result `(decision, reason)`. `decision=True` → keep latest image turn. `decision=False` → strip all images. Yield `StatusEvent(decision_label)`.
   - Router exception → fall open to `router_failopen_keep` (default `True` = keep).
7. Build thinking-log markdown via `_build_thinking_log(...)` (see prompts.py for template — same `<details><summary>🧠 Image compressor reasoning</summary>` shape as reference).
8. `body_messages = rewrite_messages(messages, keep_idx, captions_by_url)`.
9. Yield `StatusEvent(message="✅ Compressor done", done=True)`.
10. Yield `CompressionResult(messages=body_messages, thinking_md=...)`.

**Models config:** `backend/app/models/vlm/models.yaml` adds top-level `compressor:` key with two sub-keys:

```yaml
compressor:
  caption_model_id: "qwen3-vl-8b-vllm"
  router_model_id: "qwen3-vl-8b-vllm"
```

If absent, compressor is disabled (engine never instantiated; `chat_stream` skips compressor step entirely). When present, both ids must resolve to existing model entries; `VLMManager` validates this at startup.

**Engine integration in `chat_stream` (`routers/chat.py`):**

```python
async def event_stream():
    try:
        openai_messages = build_openai_messages(body.messages)
    except FileNotFoundError as exc:
        yield _sse_error("file_missing", str(exc)); return

    engine = getattr(request.app.state, "compressor_engine", None)
    if engine is not None:
        try:
            thinking_md = ""
            async for event in engine.compress(openai_messages):
                if isinstance(event, StatusEvent):
                    yield _sse_status(event.message, event.done)
                else:  # CompressionResult
                    openai_messages = event.messages
                    thinking_md = event.thinking_md
                    break
            if thinking_md:
                yield _sse_delta(thinking_md + "\n\n")
        except Exception:
            log.exception("compressor crash; falling through to safety net")
            # openai_messages stays as built_openai_messages output
    openai_messages = enforce_image_cap(openai_messages, max_images=4)

    # ...rest of streaming as before
```

**Lifespan wiring (`app/main.py`):**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = VLMManager.from_yaml(...)
    app.state.vlm_manager = manager
    compressor_cfg = manager.compressor_config()  # reads "compressor" block from yaml
    if compressor_cfg:
        cache = CaptionCache(path="backend/data/img_captions.db")
        await cache.init()
        app.state.compressor_engine = ImageCompressorEngine(
            cache=cache, vlm_manager=manager, **compressor_cfg
        )
    else:
        app.state.compressor_engine = None
    yield
```

**SSE protocol extension:**

| Event shape | Persisted by frontend? |
|------------|------------------------|
| `{"delta":"...","done":false}` | ✅ (incl. compressor's thinking-log markdown which is part of first delta) |
| `{"type":"status","message":"...","done":bool}` (Phase 5 new) | ❌ ephemeral toast |
| `{"delta":"","done":true}` | terminal |
| `{"error":"...","message":"..."}` | ✅ → errorKind |

The frontend SSE parser added in Phase 1 already silently ignores unknown JSON shapes, so the new `{type:"status"}` event is backward-compatible with any Phase ≤ 4 frontend.

**Cap policy (Phase 5):**

- **Per-upload (frontend)**: `MAX_IMAGES = 4` per **user turn** (current message being composed). Relaxed from "per conversation total" — `lib/fileValidate.checkAttachmentCap` now counts only `attachments.length` of the in-flight composer, ignoring history.
- **Safety net (backend)**: `enforce_image_cap(max_images=4)` retained as last-resort defense if compressor fails (engine exception → passthrough → safety net dumb-strips with `[ảnh ... lược bỏ]`). Both caps match vLLM's `--limit-mm-per-prompt image=4`.

**Failure modes:**

| Failure | Handler | Result |
|---------|---------|--------|
| `caption_one` raises (timeout, vLLM 5xx) | logged, omitted from `captions_by_url` | that image kept as bytes (fail-open per image) |
| `route` raises or non-JSON | logged, fall back to `router_failopen_keep=True` | images preserved |
| `cache.put_many` IntegrityError on duplicate hash | `INSERT OR IGNORE` swallows it | first writer's row stays |
| `hash_image_url` fails (network 404 for relative path) | logged, scan entry skipped | that image untouched in body |
| Engine `compress()` itself raises | caught at `chat_stream` level, log.exception, fall through to `enforce_image_cap` | dumb safety-net active; user gets degraded experience |
| Router model_id and caption_model_id mismatch on startup | `VLMManager` raises `ValueError` at lifespan | server fails to start (clear error, fix yaml) |

## Frontend Design

### C.1 New dependencies

Runtime:
- `react-markdown` ^9
- `remark-gfm` ^4
- `rehype-highlight` ^7
- `highlight.js` ^11

Dev:
- `vitest` ^2
- `@testing-library/dom` ^10 (for utilities only — no component tests in v1)
- `jsdom` ^25
- `@playwright/test` ^1

No KaTeX. No `rehype-raw` (preserve react-markdown's default sanitization).

### C.2 Module layout

```
frontend/src/
├── pages/PlaygroundPage.tsx          (refactored — orchestrator only, ~150 LOC)
├── playground/
│   ├── components/
│   │   ├── MessageList.tsx
│   │   ├── MessageBubble.tsx          (markdown for assistant; plain for user)
│   │   ├── InlineEditor.tsx
│   │   ├── AttachmentPreview.tsx
│   │   ├── AttachmentRail.tsx
│   │   ├── DropOverlay.tsx
│   │   ├── ComposerBar.tsx
│   │   ├── StopButton.tsx
│   │   └── ModelDropdown.tsx
│   ├── hooks/
│   │   ├── useChatStream.ts           (SSE + AbortController)
│   │   ├── useFileUpload.ts
│   │   ├── useConversations.ts        (reducer + localStorage sync)
│   │   └── useModels.ts
│   ├── lib/                           (pure, easy to unit test)
│   │   ├── sseParser.ts
│   │   ├── messageReducer.ts
│   │   ├── fileValidate.ts
│   │   └── storage.ts
│   └── types.ts
```

`lib/*.ts` modules are pure (no React imports), making them straightforward Vitest targets without RTL.

### C.3 Types

```ts
type AttachmentRef = {
  id: string;
  url: string;
  mime: string;
  originalName: string;
  // `size` is in the upload response but not stored in conversation state
  // (no v1 feature consumes it).
};

type MessageStatus = "streaming" | "done" | "stopped" | "error";

type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  attachments?: AttachmentRef[];
  status?: MessageStatus;
  errorKind?: "connection" | "file_missing" | "bad_request" | "internal";
};

type Conversation = {
  id: string;
  title: string;
  modelId: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
};

type ConversationsState = {
  schemaVersion: 1;
  conversations: Record<string, Conversation>;
  activeId: string | null;
};
```

### C.4 SSE consumption

`lib/sseParser.ts` — pure parser maintaining a buffer between calls:

```ts
export type SseEvent =
  | { type: "delta"; delta: string }
  | { type: "done" }
  | { type: "error"; errorKind: string; message: string }
  | { type: "parse_error"; raw: string };

export function drainEvents(buffer: string): {
  events: SseEvent[];
  rest: string;
} {
  // 1. Split buffer on /\r?\n\r?\n/
  // 2. For each block except the last (incomplete), look for "data: " prefix
  // 3. JSON.parse the payload
  //    - {delta, done: false} → {type:"delta", delta}
  //    - {done: true}         → {type:"done"}
  //    - {error, message}     → {type:"error", errorKind: error, message}
  //    - parse fail           → {type:"parse_error", raw}
  // 4. Return events + the trailing incomplete fragment as rest
}
```

`hooks/useChatStream.ts`:

```ts
function useChatStream() {
  const abortRef = useRef<AbortController | null>(null);

  async function send({ messages, modelId, onDelta, onDone, onError }) {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    let resp: Response;
    try {
      resp = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages, model_id: modelId }),
        signal: ctrl.signal,
      });
    } catch (e) {
      if (e.name === "AbortError") return;
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
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const { events, rest } = drainEvents(buf);
        buf = rest;
        for (const ev of events) {
          if (ev.type === "delta") onDelta(ev.delta);
          else if (ev.type === "done") return onDone();
          else if (ev.type === "error") return onError(ev);
          // parse_error: log and continue
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") {
        onError({ errorKind: "stream", message: String(e) });
      }
    }
  }

  return { send, abort: () => abortRef.current?.abort() };
}
```

### C.5 File upload

`hooks/useFileUpload.ts`:

```ts
async function uploadFile(file: File): Promise<AttachmentRef> {
  validateFile(file);  // throws FriendlyError
  const fd = new FormData();
  fd.append("file", file);
  const resp = await fetch("/api/files", { method: "POST", body: fd });
  if (!resp.ok) throw await FriendlyError.fromResponse(resp);
  const data = await resp.json();
  if (!data.id || !data.url) throw new FriendlyError("invalid_response");
  return data;
}
```

`lib/fileValidate.ts`:

```ts
export const ALLOWED_MIME = ["image/png", "image/jpeg", "image/webp", "image/gif"];
export const MAX_BYTES = 10 * 1024 * 1024;

export function validateFile(f: File) {
  if (!ALLOWED_MIME.includes(f.type)) throw new FriendlyError("unsupported_mime");
  if (f.size === 0) throw new FriendlyError("empty_file");
  if (f.size > MAX_BYTES) throw new FriendlyError("too_large");
}

export function checkAttachmentCap(currentCount: number, historyImageCount: number) {
  return currentCount + historyImageCount < 4;
}
```

#### Composer integration

- `<input type="file" multiple accept="image/*">` for the picker.
- `onPaste` on textarea: scan `e.clipboardData.items`, dispatch image items through `uploadFile`.
- Drop zone: page-level wrapper handles `dragenter/leave/over/drop` and toggles `<DropOverlay>`. On `drop`, scan `e.dataTransfer.files`.
- Cap display: when count reaches 4, the attach button shows `4/4` and is disabled with tooltip "Tối đa 4 ảnh".

### C.6 Vision-aware UI

`useModels` caches `[{id, name, capabilities:{vision}}]` from `/api/models`. The composer reads the active model's `capabilities.vision`:

- `vision === false` → attach button hidden, drop overlay disabled, `onPaste` rejects image items with toast "Model hiện tại không hỗ trợ ảnh".
- Switching mid-conversation to a non-vision model when attachments exist (current input or history) → banner above composer: "Model mới không hỗ trợ ảnh; gửi sẽ thất bại". Send button disabled.

### C.7 Markdown rendering

```tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";

function AssistantContent({ text }: { text: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={{ a: SafeLink }}
    >
      {text}
    </ReactMarkdown>
  );
}
```

`SafeLink` opens external links in a new tab with `rel="noopener noreferrer"`. User messages render with plain `whitespace-pre-wrap` to avoid surprising rendering of user-typed characters.

Streaming partial markdown re-renders on every delta. An unclosed code fence renders inline-ish until the closing ``` arrives, at which point it snaps to a block. No debouncing in v1; revisit if Phase 2 manual perf check shows lag.

### C.8 Stop / Regenerate / Edit

State machine for an assistant message:

```
streaming ─stop───→ stopped
        ├─error──→ error
        └─chunks─→ done
```

- **Stop**: rendered only when the latest message has `status === "streaming"`. Click → `useChatStream.abort()` → reducer `MARK_STOPPED`. Render layer decorates with suffix `[bị dừng]` based on `status`; the stored `text` is not mutated.
- **Regenerate**: rendered on the latest assistant message when `status` is `done`, `stopped`, or `error`. Click → reducer `POP_LAST_ASSISTANT` → `useChatStream.send()` with the prior context.
- **Edit**: rendered on any user message (hover-revealed). Click → `<MessageBubble>` swaps to `<InlineEditor>`. Save → reducer `EDIT_USER_AND_TRUNCATE` (replaces text, drops everything after) → `send()`. Esc / Cancel → revert. Empty text → reject with toast. Attachments are preserved.

Edit and Regenerate are disabled while any message in the conversation has `status === "streaming"`.

A reducer guard ensures `APPEND_DELTA` is a no-op if the target message's `status !== "streaming"` — prevents late SSE chunks from continuing to mutate a message the user already stopped (R10).

### C.9 Persistence

`lib/messageReducer.ts` — pure reducer:

```ts
type Action =
  | { type: "NEW_CONVERSATION"; modelId: string }
  | { type: "DELETE_CONVERSATION"; id: string }
  | { type: "SELECT_CONVERSATION"; id: string }
  | { type: "ADD_USER_MESSAGE"; conversationId: string; message: Message }
  | { type: "ADD_ASSISTANT_PLACEHOLDER"; conversationId: string; messageId: string }
  | { type: "APPEND_DELTA"; conversationId: string; messageId: string; delta: string }
  | { type: "MARK_DONE"; conversationId: string; messageId: string }
  | { type: "MARK_STOPPED"; conversationId: string; messageId: string }
  | { type: "MARK_ERROR"; conversationId: string; messageId: string; errorKind: string }
  | { type: "POP_LAST_ASSISTANT"; conversationId: string }
  | { type: "EDIT_USER_AND_TRUNCATE"; conversationId: string; messageId: string; newText: string }
  | { type: "RENAME_TITLE"; conversationId: string; title: string }
  | { type: "HYDRATE"; state: ConversationsState };

export function conversationsReducer(
  state: ConversationsState,
  action: Action,
): ConversationsState { /* ... */ }
```

`lib/storage.ts`:

```ts
const KEY = "playground.conversations";

export function read(): ConversationsState | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed.schemaVersion !== 1) return null;
    return parsed;
  } catch (e) {
    console.warn("Storage parse failed; resetting", e);
    return null;
  }
}

export function write(state: ConversationsState): void {
  localStorage.setItem(KEY, JSON.stringify(state));
}
```

`useConversations` wires reducer + storage:

```ts
function useConversations() {
  const [state, dispatch] = useReducer(conversationsReducer, INIT, init => {
    const stored = read();
    return stored ?? init;
  });
  useEffect(() => {
    try { write(state); }
    catch (e) {
      if ((e as DOMException).name === "QuotaExceededError") {
        toast("Bộ nhớ đầy. Hãy xoá conversation cũ.");
      }
    }
  }, [state]);
  // Hydrate-time coercion: any message with status "streaming" → "stopped"
  // performed in the lazy initializer above (inside `init` callback)
  return { state, dispatch };
}
```

On hydrate, any message with `status === "streaming"` is coerced to `"stopped"`. Image rendering uses `<img src={attachment.url} onError={swapToPlaceholder}>` so server-deleted files degrade gracefully.

### C.10 Polish

- **Smart auto-scroll**: track whether the user is at the bottom (`scrollTop + clientHeight >= scrollHeight - 32`). Auto-scroll only when at-bottom. Resume auto-scroll when user scrolls back to bottom.
- **First-chunk indicator**: while `status === "streaming" && text === ""`, render a pulsing "..." dot.
- **Error bubble**: assistant message with `status: "error"` shows a red-tinted bubble with the error message and a "Thử lại" button (regenerate).
- **Title auto-generation**: from the first user message text, truncated to 40 chars. If the first message has no text (image-only), title is `"Conversation YYYY-MM-DD HH:MM"`.

### C.11 Status banner + thinking-log rendering (Phase 5)

**Status banner** (ephemeral live progress from compressor):

`StatusBanner.tsx` is a small pill rendered at the top of the chat scroll area (above `MessageList`). It accepts `{message: string, done: boolean} | null`. Renders nothing when `null`. When the user receives `{type:"status", done:false}` events, banner shows the latest message with a subtle spinner. When `done:true` arrives, spinner stops; banner auto-clears after 1500ms via `setTimeout` in a `useEffect`. New events arriving before timeout cancel it (via `useRef<number>` cleanup).

State lives in `PlaygroundPage` as `useState<StatusBannerState | null>`. `useChatStream` accepts a new optional `onStatus` callback wired to `setStatus`. The banner is **not persisted** in the conversation reducer — reload clears it (correct behavior; status is transient).

Visual shape (Tailwind):
```
[16px round pill, pale blue bg, soft border, 12px text]
🖼️ Captioning 2 new image(s)...    [ subtle spinner ]
```

**Thinking-log rendering** (persisted in `msg.text`):

The compressor prepends a Markdown `<details>` block to the first delta of the assistant message:

```html
<details>
<summary>🧠 Image compressor reasoning (3 ảnh, 1 caption mới, 🎯 keep images)</summary>

**Step 1 — Image scan**
- Tổng 3 ảnh; cache miss: 1, hit: 2

**Step 2 — Captions in use**
- `a1b2c3d4` → "Một con mèo đen..."
- `e5f6g7h8` → "Bãi biển hoàng hôn..."
- `i9j0k1l2` → "Phòng khách..."

**Step 3 — Router**
- User: "Còn cái này thì sao?"
- 🎯 Router: keep images
- Reason: *Câu hỏi tham chiếu trực tiếp ảnh ("cái này")*

**Step 4 — Rewrite**
- Token estimate saved: ~2400
</details>

```

`react-markdown` + `rehype-raw` (NEW dep, see below) renders `<details>` natively as a collapsed expandable block. Browser default styling is fine for v1; CSS tweaks deferred.

**New frontend dep:** `rehype-raw@7.x`. `react-markdown` v9+ does not allow raw HTML by default; `rehype-raw` plugin enables it. Added to `MessageBubble`'s plugin chain after `rehype-highlight`.

Security note: raw HTML in assistant content is supplied by **our own backend** (engine.py builds the `<details>` block). Model output is appended *after* the thinking block, so any model-injected HTML still passes through `rehype-raw` — but the existing Phase 2 safeguard (DOMPurify is NOT used, but XSS via raw `<script>` tags is mitigated because `rehype-raw` allows HTML elements but `react-markdown` strips event handlers and `<script>` is rendered as inert text by default per Browser HTML5 parser rules — verify in A5.9).

**SSE parser changes** (`lib/sseParser.ts`):

Add new event variant. Before:
```ts
type SSEEvent =
  | { type: "delta"; delta: string }
  | { type: "done" }
  | { type: "error"; errorKind: ErrorKind; message?: string }
  | { type: "parse_error" };
```
After:
```ts
type SSEEvent =
  | { type: "delta"; delta: string }
  | { type: "done" }
  | { type: "error"; errorKind: ErrorKind; message?: string }
  | { type: "status"; message: string; statusDone: boolean }   // NEW
  | { type: "parse_error" };
```

`drainEvents` matches `payload.type === "status"` *before* the existing `done`/`error` branches and yields the new typed event. Unknown `type` values still fall through to delta/done/error/ignored as today.

**chatStream changes** (`lib/chatStream.ts`):

Add `onStatus?: (msg: string, done: boolean) => void` to `ChatStreamOptions`. In the event loop:

```ts
case "status":
  options.onStatus?.(event.message, event.statusDone);
  break;
```

`onDelta`, `onDone`, `onError`, `abort()` semantics unchanged. Status events do **not** count as "first chunk" — they don't reset the in-flight retry semantics from Phase 1.

**PlaygroundPage wiring:**

```tsx
const [status, setStatus] = useState<{message: string; done: boolean} | null>(null);
const statusTimeoutRef = useRef<number | null>(null);

const onStatus = (message: string, done: boolean) => {
  if (statusTimeoutRef.current) {
    window.clearTimeout(statusTimeoutRef.current);
    statusTimeoutRef.current = null;
  }
  setStatus({ message, done });
  if (done) {
    statusTimeoutRef.current = window.setTimeout(() => setStatus(null), 1500);
  }
};

// ... in JSX, above <MessageList>:
<StatusBanner status={status} />
```

`onStatus` is passed into the existing `useChatStream` hook. On stream abort/error/conversation switch, `setStatus(null)` clears the banner immediately.

**Per-upload cap (`lib/fileValidate.ts`):**

```ts
// Phase 4 (current):
export function checkAttachmentCap(currentInComposer: number, historyImageCount: number): boolean {
  return currentInComposer + historyImageCount < MAX_IMAGES;
}

// Phase 5 (new):
export function checkAttachmentCap(currentInComposer: number): boolean {
  return currentInComposer < MAX_IMAGES;
}
```

`historyImageCount` parameter removed; callers updated to pass only the in-flight composer count. `MAX_IMAGES` stays at 4. The `historyImageCount` prop on `ComposerBar` is removed (was used only by this cap). Phase 2/4 tests that asserted "history+composer ≥ 4 → block" need updating; Phase 5 plan documents the migration.

## Phase Breakdown

Five phases. Each is independently shippable — no half-broken state at the boundary between phases. Tests are split into **Happy** (T-x.y), **Adversarial** (A-x.y), and **E2E** (E-x.y).

### Phase 1 — Backend foundation

**Deliverables:**
- Async migration: `VLMProvider.stream()` base + `OpenAICompatibleProvider` + `QwenVLLMProvider` + `VLMManager.stream`. `generate()` becomes a wrapper.
- `services/files.py` (`store_upload`, `open_image_bytes`).
- `routers/files.py` (`POST /api/files`, `GET /api/files/{id}`).
- `services/messages.py` (`build_openai_messages`, `enforce_image_cap`).
- `routers/chat.py`: new `POST /api/chat/stream` SSE; `/api/chat` rewritten on async path.
- `models.yaml`: `capabilities.vision`. `manager.list_models` exposes it.
- `requirements.txt`: add `python-multipart`, `pytest-asyncio`, `pytest-httpx` (or `respx`).

**Happy (pytest):**

| ID | Description |
|----|-------------|
| T1.1 | `OpenAICompatibleProvider.stream` yields N deltas when SDK mock returns an async iterator with 3 chunks. |
| T1.2 | `QwenVLLMProvider.stream` applies transforms: messages passed to SDK have `<image>` stripped and `min_pixels`/`max_pixels` injected on every `image_url`. |
| T1.3 | `QwenVLLMProvider.stream` retries once on `APIConnectionError` raised before first chunk, then succeeds. |
| T1.4 | `VLMManager.stream` prepends the system prompt as the first message. |
| T1.5 | `generate()` (wrapper) gathers `["hello", " world"]` into `"hello world"`. |
| T1.6 | `POST /api/files` with valid PNG → 200; body has `{id, url, mime: "image/png", size}`; file exists at `images/<id>.png`. |
| T1.7 | `GET /api/files/<id>` → 200; bytes match original; `Content-Type: image/png`. |
| T1.8 | `POST /api/chat/stream` with mocked provider yielding 2 chunks → response is `text/event-stream`; parse 2 delta events + 1 done. |
| T1.9 | `build_openai_messages`: message with 1 attachment → content array with 1 `image_url` (data URI) + 1 `text`. |
| T1.10 | `enforce_image_cap`: history has 5 images, max 4 → oldest replaced with placeholder text; newest 4 preserved. |
| T1.11 | `enforce_image_cap`: message with 1 image_url and no text → replaced with placeholder string content (not empty array). |
| T1.12 | `/api/models` returns each model with `capabilities.vision: bool`. |

**Adversarial (pytest):**

| ID | Description |
|----|-------------|
| A1.1 | Upload `.txt` with spoofed `Content-Type: image/png` → 400 (PIL `verify()` fails). |
| A1.2 | Upload zero-byte file → 400. |
| A1.3 | Upload 11MB file → 413. |
| A1.4 | Upload `image/svg+xml` (not in whitelist) → 415. |
| A1.5 | Upload with filename `../../etc/passwd` → file stored at `images/<uuid>.png`; response `original_name` may echo client value but no path component reaches disk. |
| A1.6 | `GET /api/files/foo/bar` (path traversal) → 400. |
| A1.7 | `GET /api/files/<32-hex>` non-existent → 404. |
| A1.8 | `GET /api/files/<id>` finds file under any whitelisted extension via the `EXT_BY_MIME` loop. |
| A1.9 | `POST /api/chat/stream` with unknown `model_id` and no default fallback → SSE `error: bad_request`. |
| A1.10 | Provider raises `ConnectionError` before first chunk → SSE `error: connection`; no `delta` events emitted. |
| A1.11 | Provider yields 3 chunks then raises → 3 `delta` events + `error: internal`. |
| A1.12 | Request with attachment id missing on disk → SSE `error: file_missing`; provider not called. |
| A1.13 | Test client disconnect: simulate `Request.is_disconnected() → True` after chunk 1 → loop exits; spy on chunk consumer confirms no further yields. |
| A1.14 | `enforce_image_cap` with 8 images in one message → reduced to 4; placeholder text segments coalesced. |
| A1.15 | `build_openai_messages` with file deleted between request start and resolution → `FileNotFoundError` → handler emits SSE `error: file_missing`. |
| A1.16 | Two concurrent uploads with mocked uuid collision → second returns 409 (theoretical; verify behavior is defined). |
| A1.17 | Valid MIME but corrupt JPEG bytes → 400 "Not a valid image" from PIL. |
| A1.18 | SDK chunk with `delta.content = None` (function-call chunk) → skipped, no empty delta yielded. |
| A1.19 | Multi-byte UTF-8 split across SDK chunks (`b"\xc3"`, `b"\xa9"`) → SSE wrapper emits valid UTF-8 (TextDecoder with `stream=True`). |
| A1.20 | Provider raises `BadRequestError` (vLLM 400 for token cap) → SSE `error: bad_request` carrying vLLM message. |

### Phase 2 — Frontend streaming + multimodal + markdown

**Deliverables:**
- Module layout from C.2.
- `PlaygroundPage.tsx` refactored to ~150 LOC orchestrator; logic in hooks.
- `react-markdown` + plugins integrated; `highlight.js` theme imported.
- `useChatStream` + `sseParser` integrated end-to-end.
- Multi-image upload (picker, drag/drop overlay, paste).
- `AttachmentRail` with thumbnails + remove button.
- `ModelDropdown` gates attach button on `capabilities.vision`.
- Client-side cap pre-check (`fileValidate.checkAttachmentCap`).

**Happy (Vitest, pure libs):**

| ID | Description |
|----|-------------|
| T2.1 | `drainEvents` parses 1 buffer with 3 SSE blocks → 3 typed events; rest = "". |
| T2.2 | `drainEvents` handles event split across two reads (buffer carries trailing partial). |
| T2.3 | `drainEvents` emits `{type:"done"}` for `{done:true}` payload. |
| T2.4 | `drainEvents` emits `{type:"error", errorKind, message}` for error payload. |
| T2.5 | `messageReducer.APPEND_DELTA` appends to the correct message in the correct conversation. |
| T2.6 | `messageReducer.ADD_ASSISTANT_PLACEHOLDER` creates message with `status: "streaming"`, empty text. |
| T2.7 | `messageReducer.MARK_DONE` flips status from `streaming` to `done`. |
| T2.8 | `fileValidate.validateFile` accepts PNG/JPEG/WebP/GIF; rejects SVG, PDF, exe. |
| T2.9 | `fileValidate.validateFile` rejects > 10MB and zero-byte. |
| T2.10 | `fileValidate.checkAttachmentCap` returns false at 4 attachments + history. |

**Happy (Vitest, `useChatStream` with mocked fetch):**

| ID | Description |
|----|-------------|
| T2.11 | `send()` POSTs JSON with `messages` and `model_id`. |
| T2.12 | `send()` parses stream, calls `onDelta` 3 times then `onDone`. |
| T2.13 | `send()` with `resp.ok === false` calls `onError({errorKind:"http"})`. |

**Happy (E2E Playwright, manual smoke):**

| ID | Description |
|----|-------------|
| E2.1 | Golden path: select Qwen3-VL → type "ảnh này gì" + upload 1 image → click Send → tokens stream visibly → final markdown rendered. |
| E2.2 | Multi-image: upload 3 images in a single turn → 3 thumbnails in rail → send → response. |

**Adversarial (Vitest + manual):**

| ID | Description |
|----|-------------|
| A2.1 | Buffer with `\r\n\r\n` (Windows newlines) parses correctly. |
| A2.2 | Malformed JSON in `data:` → `{type:"parse_error"}`, no crash. |
| A2.3 | Empty `data:` line → skipped, no event emitted. |
| A2.4 | `APPEND_DELTA` to non-existent messageId → no-op (no message created). |
| A2.5 | `useChatStream.send()` while a session is active → previous aborted before new starts. |
| A2.6 | `useChatStream.abort()` mid-stream → reader.cancel() runs; `onError` not called for AbortError. |
| A2.7 | Vietnamese text split across two SSE chunks → renders correctly (no mojibake). |
| A2.8 | Markdown input `<script>alert(1)</script>` → rendered as text, not executed. |
| A2.9 | Markdown with unclosed code fence (` ```python\nprin`) → renders without crash. |
| A2.10 | Drop 5 files at once → 5th rejected with toast; first 4 uploaded. |
| A2.11 | Drop `.svg` file → rejected with toast "Định dạng không hỗ trợ". |
| A2.12 | Paste binary clipboard image (e.g., from Excel screenshot) → uploaded. |
| A2.13 | Paste plain text URL → enters textarea normally, no upload triggered. |
| A2.14 | Switch to non-vision model with attachments present → banner shown, Send disabled. |
| A2.15 | Network drops during upload → `useFileUpload` rejects with friendly error; UI not stuck pending. |
| A2.16 | `/api/files` response missing `id` field → `useFileUpload` throws `invalid_response`. |
| A2.17 | SSE stalls 30s with no chunks → UI remains responsive; Stop button still clickable. |

### Phase 3 — UX controls (stop / regenerate / edit linear)

**Deliverables:**
- `StopButton` rendered conditionally; click triggers abort + `MARK_STOPPED`.
- Regenerate button on last assistant bubble for non-`streaming` statuses.
- Edit button on user bubbles (hover-reveal); `InlineEditor` swap; Save → truncate + resend.
- "Thử lại" on error bubbles (alias for regenerate).
- Reducer guard: `APPEND_DELTA` no-op when target status ≠ `"streaming"`.

**Happy (Vitest reducer):**

| ID | Description |
|----|-------------|
| T3.1 | `MARK_STOPPED` updates status; text untouched. |
| T3.2 | `POP_LAST_ASSISTANT` removes last message iff role === assistant; otherwise no-op. |
| T3.3 | `EDIT_USER_AND_TRUNCATE` replaces text and drops all messages after. |
| T3.4 | `EDIT_USER_AND_TRUNCATE` preserves `attachments` of edited message. |

**Happy (Playwright):**

| ID | Description |
|----|-------------|
| E3.1 | Send → after first chunk, click Stop → partial preserved → click Regenerate → new reply replaces. |
| E3.2 | Edit user msg #2 in 4-msg conv → Save → msgs #3, #4 vanish → new reply generated as #3. |
| E3.3 | Edit → change text → Esc → bubble reverts. |

**Adversarial:**

| ID | Description |
|----|-------------|
| A3.1 | Click Stop when `status === "done"` → button not rendered. |
| A3.2 | Click Regenerate during streaming → button disabled. |
| A3.3 | Stop then immediately Regenerate → 2 distinct fetch calls; no double-stream. |
| A3.4 | Edit message #1 in 5-msg conv → all 4 subsequent dropped (reducer test). |
| A3.5 | Edit while streaming → button disabled. |
| A3.6 | Save edit with empty text → rejected with toast "Tin nhắn rỗng". |
| A3.7 | Regenerate when network down → SSE error → bubble shows "Thử lại". |
| A3.8 | 5 rapid Regenerate clicks → previous aborted on each; only one final reply. |
| A3.9 | Edit user message with 4 attachments → attachments persist (reducer test). |
| A3.10 | Edit in convo A while another convo B has pending input → state isolation; B unaffected. |
| A3.11 | Stop with partial code fence (` ```python\nprin`) → renders as-is + decorated `[bị dừng]`. |

### Phase 4 — Persistence + polish

**Deliverables:**
- `lib/storage.ts` versioned read/write.
- `useConversations` reducer + storage sync; hydrate on mount.
- Sidebar conversation list (existing UI, new plumbing).
- `<img src={attachment.url}>` with `onError` placeholder.
- Smart auto-scroll, first-chunk indicator, error bubble with retry, model-switch banner, hydrate-time `streaming → stopped` coercion.

**Happy (Vitest):**

| ID | Description |
|----|-------------|
| T4.1 | `storage.write` + `storage.read` round-trip preserves shape. |
| T4.2 | `HYDRATE` action overwrites state with payload. |
| T4.3 | `RENAME_TITLE` updates only the targeted conversation. |
| T4.4 | `DELETE_CONVERSATION` removes entry; if active, `activeId` falls back to most-recent or null. |

**Happy (Playwright):**

| ID | Description |
|----|-------------|
| E4.1 | Send 2 messages → reload → sidebar shows conversation → click → 2 messages + image previews render. |
| E4.2 | New chat → 2 conversations in sidebar. |
| E4.3 | Delete conversation → removed from sidebar and localStorage. |

**Adversarial:**

| ID | Description |
|----|-------------|
| A4.1 | localStorage manually edited to "not-json" → `read()` returns null → app starts with default state; warn logged; no white screen. |
| A4.2 | localStorage with `schemaVersion: 99` → ignored, reset to default. |
| A4.3 | localStorage with message `status: "streaming"` (reload during stream) → coerced to `stopped` on hydrate. |
| A4.4 | 51 conversations cause `QuotaExceededError` on write → toast prompts user to delete; in-memory state stays. |
| A4.5 | Conversation references attachment id with file deleted server-side → `<img>` 404 → swaps to placeholder. |
| A4.6 | Conversation with attachment `mime: "image/svg+xml"` (manual edit) → `<img>` either renders or fails benignly; no crash. |
| A4.7 | Two tabs: tab A sends, tab B reload → tab B sees new state (no realtime sync; reload is the contract). |
| A4.8 | Switch models between conversations → dropdown reflects active conversation's `modelId`. |
| A4.9 | Switch conversations during streaming → previous stream aborted; previous state shows `stopped` partial. |
| A4.10 | Auto-scroll: user scrolls up mid-stream → view stays put → tokens still append → user scrolls to bottom → auto-scroll resumes. |
| A4.11 | First-render perf: load 100-message conversation from storage → < 500ms (smoke; no hard assert). |
| A4.12 | Reload with conversation in error state → bubble error visible; "Thử lại" works. |
| A4.13 | First message has no text (image-only) → title falls back to `"Conversation YYYY-MM-DD HH:MM"`. |
| A4.14 | Save edit on msg #1 of 5-msg convo → localStorage write count is 1, not 5 (debounce/batched effect). |

### Phase 5 — Image-aware history compression + thinking log

**Deliverables:**
- `backend/app/services/image_compressor/` package: `cache.py` (aiosqlite WAL), `messages.py` (helpers), `prompts.py` (VN system prompts), `types.py` (StatusEvent/CompressionResult), `engine.py` (orchestrator).
- `requirements.txt`: add `aiosqlite`.
- `models.yaml`: add top-level `compressor:` block with `caption_model_id` + `router_model_id`.
- `app/main.py` lifespan: instantiate cache + engine once if `compressor` block present; store on `app.state.compressor_engine`.
- `routers/chat.py`: thread engine into `chat_stream`; emit new `{type:"status",...}` SSE events; prepend `<details>` thinking-log block to first model delta.
- `services/messages.py`: `enforce_image_cap` retained as safety net; doc-comment updated to reflect Phase 5 role.
- Frontend: `sseParser` recognizes `{type:"status"}`; `chatStream` `onStatus` callback; new `StatusBanner` component; `PlaygroundPage` wires status state; `fileValidate.checkAttachmentCap` simplified to per-upload only; `MessageBubble` markdown plugin chain gains `rehype-raw`.

**Happy (pytest):**

| ID | Description |
|----|-------------|
| T5.B1 | `CaptionCache.put` then `get` round-trips. `get_many` over 3 hashes (2 hit, 1 miss) returns dict of 2 hits. `put_many` with duplicate hash: second put leaves first's caption intact (`INSERT OR IGNORE`). |
| T5.B2 | `iter_image_parts`: empty msgs → empty iter; str-content msg → empty; mixed parts (1 text + 2 image) → yields (idx, j, url) for each image. |
| T5.B3 | `rewrite_messages`: `keep_idx=2` of 4-msg history with 1 image each → msgs 0,1,3 get `[Past image #1: cap]` text inserts; msg 2 untouched. Coalesces with trailing text part if present. |
| T5.B4 | `hash_image_url`: `data:image/png;base64,iVBOR...` decoded → returns sha256 of decoded bytes. Bad scheme `ftp://...` → raises `ValueError`. Empty data URL → raises. |
| T5.B5 | `engine.caption_one` with mocked AsyncOpenAI returning `"Một con mèo."` → returns `"Một con mèo."` trimmed. |
| T5.B6 | `engine.route` with mocked AsyncOpenAI returning `'{"need_images":true,"reason":"câu hỏi đề cập ảnh"}'` → returns `(True, "câu hỏi đề cập ảnh")`. Returning malformed JSON → returns `(failopen_keep, "router failure: JSONDecodeError")`. |
| T5.B7 | `engine.compress` integration: 3-image history, all cache misses → yields ≥1 StatusEvent then exactly 1 CompressionResult; rewritten messages have `[Past image #N:]` inserts. |
| T5.B8 | `chat_stream` integration: 2 images in history, mocked compressor + provider → SSE stream contains `{type:"status"}` event then `{delta: "<details>..."}` then model deltas then `{done:true}`. |
| T5.B9 | `chat_stream` no-image fast path: 0 images → 0 status events, 0 thinking-log delta; provider called immediately with original messages. |

**Adversarial (pytest):**

| ID | Description |
|----|-------------|
| A5.B1 | `caption_one` raises `httpx.TimeoutException` → `ensure_captions` omits that url; other captions complete; cache stores only successes. |
| A5.B2 | `route` raises `httpx.HTTPError` → returns `(failopen_keep=True, "router failure: ...")`; images preserved. |
| A5.B3 | Two concurrent `cache.put_many` with same hash → both complete without IntegrityError; cache has exactly 1 row. |
| A5.B4 | `hash_image_url` for `/api/files/<id>` (relative) → fetches via `webui_internal_base`, returns `(sha256, bytes)`. 404 → raises. |
| A5.B5 | Engine `compress()` itself raises (e.g., cache db locked indefinitely) → `chat_stream` catches and falls through; SSE stream still yields model deltas via dumb safety-net. |
| A5.B6 | History has the same image bytes uploaded twice with different `id`s → `hash_image_url` returns same hash → cache hit on second; only 1 caption call total. |

**Happy (Vitest):**

| ID | Description |
|----|-------------|
| T5.1 | `sseParser` parses `{type:"status","message":"X","done":false}` → yields `{type:"status",message:"X",statusDone:false}`. |
| T5.2 | `sseParser` parses `{type:"status","message":"Done","done":true}` → yields `{statusDone:true}`. |
| T5.3 | `sseParser` ignores `{type:"future_unknown_shape"}` → no event yielded; following delta still parsed (forward-compat preserved). |
| T5.4 | `chatStream.send` invokes `onStatus("X", false)` then `onStatus("Y", true)` when status events arrive in stream. |
| T5.5 | `<StatusBanner status={{message:"X",done:false}} />` renders text "X" + spinner role. With `done:true`, after 1500ms `setStatus(null)` is called via callback prop (test with `vi.useFakeTimers`). |

**Happy (Playwright):**

| ID | Description |
|----|-------------|
| E5.1 | Send msg with 1 image → mock backend emits `{type:"status",message:"🖼️ Captioning..."}` → banner appears with text → mock emits `done:true` → banner clears within 2s. |
| E5.2 | Mock backend prepends `<details><summary>🧠 Image compressor reasoning (1 ảnh, 1 caption mới, kept new upload)</summary>...</details>` to first delta + appends "Hello world." → assistant bubble renders summary text + answer; clicking summary expands details. |

**Adversarial:**

| ID | Description |
|----|-------------|
| A5.1 | (covered by T5.B5) Caption call fails for one image → that image kept (fail-open) — verify in chat_stream integration test. |
| A5.2 | (covered by T5.B6) Router fails → `failopen_keep=True` → images preserved. |
| A5.3 | (covered by T5.B3) Cache write contention: 2 concurrent `put_many` with same hash → INSERT OR IGNORE leaves one row. |
| A5.4 | Conversation with cached images → click Regenerate → Network tab shows 0 caption calls (cache hit 100%). Manual Playwright assertion via `page.on("request", ...)`. |
| A5.5 | Restart backend → cache file `backend/data/img_captions.db` survives → next request: cache hit. Manual smoke. |
| A5.6 | User edits user-msg #1 (4 images) → truncate → next stream: same hashes → still uses cached captions. Manual smoke. |
| A5.7 | SSE proxy buffering check: 3 status events 200ms apart → arrive on FE 200ms apart (not all at once). Manual smoke (DevTools EventStream tab). |
| A5.8 | Thinking log `<details>` rendered collapsed by default (browser default). Click summary → expands. Verified in E5.2. |
| A5.9 | XSS check: assistant content with raw `<script>alert(1)</script>` → does NOT execute (rehype-raw + html5-parser rules render as text or strip). Vitest with RTL. |
| A5.10 | Per-upload cap: history has 4 images, composer empty → user attaches 4 more in current turn → all 4 attach (cap is per-turn now, not per-conversation). Manual smoke. |
| A5.11 | Compressor disabled (no `compressor` block in `models.yaml`) → `chat_stream` skips compressor entirely → 0 status events, 0 thinking log; behavior identical to Phase 4. Pytest. |

### Cross-phase notes

- **Test infra setup**: Phase 1 adds `pytest-asyncio` and `pytest-httpx` (or `respx`). Phase 2 sets up Vitest + jsdom. Phase 3 onward configures `@playwright/test` (Chromium project only). Phase 5 adds `aiosqlite` to backend deps.
- **CI**: not configured in v1. Local pre-commit run is the contract.
- **Smoke script**: `scripts/smoke_qwen3_vl.py` continues to work via `manager.generate()`. Run as acceptance for any phase that touches providers (1, 3 if reducer affects flow, 4 only if attachment resolution changes, 5 changes request-shape preprocessing but `manager.generate` itself is unchanged).

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | vLLM cancellation may not propagate when client disconnects. | Phase 1 manual test: trigger client abort post-first-chunk; inspect vLLM logs for "cancelled". If not, call `result.aclose()` explicitly. |
| R2 | Mocking `AsyncOpenAI` async iterators is fiddly. | Phase 1 introduces `make_async_stream_mock(deltas)` shared helper. |
| R3 | Retry-on-stream is only valid pre-first-chunk. Mid-stream errors propagate. | Documented; A1.11 covers post-first-chunk error path. |
| R4 | `request.is_disconnected()` polling adds overhead per yield. | Acceptable for v1; switch to `asyncio.shield` + task cancel if measured lag. |
| R5 | vLLM 32k context may overflow for long multimodal histories. | No truncation in v1; surface vLLM 400 as `error: bad_request`; user starts a new chat. |
| R6 | PIL magic-byte verify on 10MB images may be slow. | Use `Image.open(buf).verify()` (header-only) instead of full decode. |
| R7 | localStorage 5–10MB cap fills with many conversations. | Toast prompt to delete oldest; no auto-eviction. |
| R8 | Windows non-atomic rename across volumes. | tmp file always co-located with destination in `images/`; rename is atomic in same folder. |
| R9 | React-markdown re-parses on every delta — potential lag on long messages. | Measure in Phase 2 manual; debounce 32ms or only re-render on newline if needed. |
| R10 | Race between abort and in-flight SSE chunks. | Reducer guard: `APPEND_DELTA` no-op when `status !== "streaming"`. |
| R11 | Edit/regenerate during in-flight stream creates inconsistent state. | `useChatStream.abort()` runs synchronously before reducer mutation. |
| R12 | CORS / dev proxy for `<img src="/api/files/...">`. | Same-origin via Vite proxy; verified in Phase 1 manual smoke. |
| R13 | Backwards compat for `/api/chat` non-stream. | Route kept; rewritten to delegate to async `manager.generate()` wrapper. |
| R14 | `delta.content = None` chunks (e.g., function-call placeholders). | Skipped (A1.18). |
| R15 | HEIC images from iPhone fail browser/PIL decode. | Rejected at `fileValidate` (not in whitelist); user converts manually. |
| R16 | Solo-dev time estimate: Phase 1 ~3–4 days, P2 ~2–3, P3 ~1–2, P4 ~2, P5 ~2–3. ~10–14 days total. | Phase 1+2 alone is shippable as a v0.5 if needed sooner. |
| R17 | Caption call latency on cold cache (~3-5s/img × up to 4 imgs in parallel = 12-20s) blocks first model token. | `asyncio.gather` runs caption calls in parallel; status banner gives user real-time feedback; cache hits make subsequent turns instant. Document expected cold-start latency in README. |
| R18 | aiosqlite WAL is single-writer; multi-uvicorn-worker deploy could see brief write blocking. | v1 documents single-worker deploy. If horizontal scale needed → switch to Postgres or per-process cache (loses dedup). |
| R19 | Caption hallucination — captioner LLM may invent details not present in image. | System prompt explicitly constrains ("KHÔNG suy diễn cảm xúc, KHÔNG bịa"). Caption sloppy → answer sloppy; same risk as the reference filter. Caption is cheap enough that user can manually re-caption by deleting cache entry. |
| R20 | Router LLM returns non-JSON despite `response_format={type:"json_object"}` (vLLM bug). | `try/except json.JSONDecodeError → fail-open keep_images`. Test A5.B2 covers. |
| R21 | SSE buffering between Vite proxy and uvicorn could batch status events. | `StreamingResponse` + `text/event-stream` is chunked by uvicorn; Vite passes through. Manual smoke A5.7 verifies real-time delivery in dev. |
| R22 | `rehype-raw` allowing raw HTML opens XSS surface if model output ever contains `<script>`. | Test A5.9 verifies inert rendering. Backend never echoes user-supplied HTML; only the engine's own `<details>` block is HTML. If future Phase 6 adds user-authored markdown that may contain HTML, revisit. |
| R23 | Conversations from Phase 4 (pre-compressor) reload with >4 images → compressor first run hammers caption LLM. | Acceptable: one-time cost. Cache fills; subsequent runs hit. User can opt out via `compressor:` block removal. |

## Open Questions (defaults committed)

| # | Question | Default |
|---|----------|---------|
| Q1 | Stop generation: mutate text or decorate? | Decorate-only (`status` change; render layer adds `[bị dừng]`). |
| Q2 | Edit with empty text: reject or allow? | Reject with toast. |
| Q3 | Reload during `status: "streaming"`: coerce to what? | `stopped` + decoration. |
| Q4 | localStorage quota: auto-evict LRU or prompt? | Prompt user via toast. |
| Q5 | File ID URL form: `/api/files/<id>` or `/api/files/<id>.<ext>`? | No extension; server detects via the `EXT_BY_MIME` loop. |
| Q6 | Vision-incompatible model with attachments: block client-side or let vLLM 400? | Block client-side; Send disabled. |
| Q7 | Auto-title: first user msg or LLM summary? | First message text truncated 40 chars; fallback to date string for image-only. |
| Q8 | SSE format: JSON in `data:` or typed events? | JSON in `data:`. |
| Q9 | Warn user when context approaches 28k chars? | No warning; surface 400 instead. |
| Q10 | Orphan upload cleanup script? | None in v1; documented manual cleanup. |
| Q11 | Compressor toggle UI (enable/disable, force-keep-all-images)? | Hardcoded to always-on. UI toggles deferred to potential Phase 6. |
| Q12 | Caption cache eviction (TTL, max-size)? | None in v1. SQLite grows unbounded. Manual `DELETE FROM captions WHERE created_at < ?` if needed. |
| Q13 | Smart cache-aware safety net (lookup caption before falling back to dumb placeholder)? | No. Safety net stays dumb. If compressor crashes regularly, fix compressor; do not duplicate logic. |
| Q14 | Caption / router separate model from main model? | No. Default `qwen3-vl-8b-vllm` for both, configurable via `compressor.caption_model_id` + `compressor.router_model_id` in `models.yaml`. Future phases could add a smaller text-only router. |

## Acceptance Criteria

A phase is complete when:

1. All Happy tests for that phase pass.
2. All Adversarial tests for that phase pass.
3. `scripts/smoke_qwen3_vl.py` continues to pass (Phase 1+ touches providers).
4. Manual browser smoke of the golden path produces no console errors.
5. Code self-review via the `simplify` skill before commit.
