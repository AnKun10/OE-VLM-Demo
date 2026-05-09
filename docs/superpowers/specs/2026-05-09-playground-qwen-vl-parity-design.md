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
           ├─ enforce_image_cap: total images ≤ 4, drop oldest with placeholder
           ├─ VLMManager.stream(model_id, messages)
           │  └─ QwenVLLMProvider.stream
           │     ├─ apply transforms (strip_image_tokens, inject_pixel_bounds)
           │     └─ AsyncOpenAI client.chat.completions.create(stream=True)
           │        → async iterator of ChatCompletionChunk
           ├─ wrap chunks → SSE: data: {"delta":"...","done":false}\n\n
           └─ on client disconnect → break loop → cleanup async generator
PlaygroundPage
  └─ ReadableStream reader → sseParser → reducer APPEND_DELTA
  └─ on done/error → reducer MARK_DONE / MARK_ERROR
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

## Phase Breakdown

Four phases. Each is independently shippable — no half-broken state at the boundary between phases. Tests are split into **Happy** (T-x.y), **Adversarial** (A-x.y), and **E2E** (E-x.y).

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

### Cross-phase notes

- **Test infra setup**: Phase 1 adds `pytest-asyncio` and `pytest-httpx` (or `respx`). Phase 2 sets up Vitest + jsdom. Phase 3 onward configures `@playwright/test` (Chromium project only).
- **CI**: not configured in v1. Local pre-commit run is the contract.
- **Smoke script**: `scripts/smoke_qwen3_vl.py` continues to work via `manager.generate()`. Run as acceptance for any phase that touches providers (1, 3 if reducer affects flow, 4 only if attachment resolution changes).

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
| R16 | Solo-dev time estimate: Phase 1 ~3–4 days, P2 ~2–3, P3 ~1–2, P4 ~2. ~8–11 days total. | Phase 1+2 alone is shippable as a v0.5 if needed sooner. |

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

## Acceptance Criteria

A phase is complete when:

1. All Happy tests for that phase pass.
2. All Adversarial tests for that phase pass.
3. `scripts/smoke_qwen3_vl.py` continues to pass (Phase 1+ touches providers).
4. Manual browser smoke of the golden path produces no console errors.
5. Code self-review via the `simplify` skill before commit.
