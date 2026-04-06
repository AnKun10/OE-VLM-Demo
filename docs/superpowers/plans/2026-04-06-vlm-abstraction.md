# VLM Abstraction Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate VLM logic into a provider-based abstraction with YAML config, OpenAI SDK client, multi-model runtime selection, and a frontend model dropdown.

**Architecture:** Replace the in-process vLLM integration with an HTTP client pattern using the OpenAI SDK. A `VLMManager` loads model definitions from `models.yaml`, instantiates provider classes per model, and routes generation requests. The frontend gets a new `GET /api/models` endpoint and sends `model_id` with each chat request.

**Tech Stack:** Python (FastAPI, openai SDK, pyyaml), TypeScript/React (ShadCN Select), existing Vite+TailwindCSS frontend.

**Note:** No test infrastructure is configured in this project. Steps focus on manual verification via the running dev server.

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `backend/app/models/vlm/__init__.py` | Exports `VLMManager` |
| Create | `backend/app/models/vlm/providers/__init__.py` | Empty package init |
| Create | `backend/app/models/vlm/providers/base.py` | Abstract `VLMProvider` base class |
| Create | `backend/app/models/vlm/providers/openai_compatible.py` | OpenAI SDK client provider |
| Create | `backend/app/models/vlm/manager.py` | Loads YAML, routes requests to providers |
| Create | `backend/app/models/vlm/models.yaml` | Model registry with inline-commented attributes |
| Create | `backend/app/models/memory/__init__.py` | Placeholder |
| Create | `backend/app/models/summary/__init__.py` | Placeholder |
| Modify | `backend/app/main.py` | Swap lifespan to use `VLMManager` |
| Modify | `backend/app/routers/chat.py` | Use manager, add `model_id`, new `/api/models` endpoint |
| Modify | `backend/app/config.py` | Remove VLM-specific settings |
| Modify | `backend/requirements.txt` | Add `openai`, `pyyaml`; remove `vllm`, `bitsandbytes` |
| Modify | `backend/.env.example` | Update env var examples |
| Delete | `backend/app/services/vlm_service.py` | Replaced by `models/vlm/` |
| Modify | `frontend/src/pages/PlaygroundPage.tsx` | Add model selector dropdown, send `model_id` |

---

### Task 1: Create placeholder folders (memory, summary)

**Files:**
- Create: `backend/app/models/memory/__init__.py`
- Create: `backend/app/models/summary/__init__.py`

- [ ] **Step 1: Create memory placeholder**

```python
# backend/app/models/memory/__init__.py
```

Empty file. Just creates the package.

- [ ] **Step 2: Create summary placeholder**

```python
# backend/app/models/summary/__init__.py
```

Empty file. Just creates the package.

- [ ] **Step 3: Commit**

```bash
git add backend/app/models/memory/__init__.py backend/app/models/summary/__init__.py
git commit -m "feat: add placeholder memory and summary model folders"
```

---

### Task 2: Create VLM provider base class

**Files:**
- Create: `backend/app/models/vlm/__init__.py`
- Create: `backend/app/models/vlm/providers/__init__.py`
- Create: `backend/app/models/vlm/providers/base.py`

- [ ] **Step 1: Create the vlm package init**

```python
# backend/app/models/vlm/__init__.py
from .manager import VLMManager

__all__ = ["VLMManager"]
```

This will fail to import until `manager.py` exists (Task 4). That's fine — we create it now so the package structure is complete.

- [ ] **Step 2: Create the providers package init**

```python
# backend/app/models/vlm/providers/__init__.py
```

Empty file.

- [ ] **Step 3: Create the abstract base class**

```python
# backend/app/models/vlm/providers/base.py
from __future__ import annotations

from abc import ABC, abstractmethod


class VLMProvider(ABC):
    """Base class for all VLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Send messages and return the model's text response."""
        ...
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/models/vlm/
git commit -m "feat: add VLMProvider abstract base class and vlm package structure"
```

---

### Task 3: Create OpenAI-compatible provider

**Files:**
- Create: `backend/app/models/vlm/providers/openai_compatible.py`

- [ ] **Step 1: Implement the provider**

```python
# backend/app/models/vlm/providers/openai_compatible.py
from __future__ import annotations

from openai import OpenAI

from .base import VLMProvider


class OpenAICompatibleProvider(VLMProvider):
    """Provider for any OpenAI-compatible API (vLLM, OpenAI, etc.)."""

    def __init__(self, base_url: str, api_key: str, model_id: str) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id

    def generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
```

- [ ] **Step 2: Verify import works**

Run from `backend/` directory:

```bash
python -c "from app.models.vlm.providers.openai_compatible import OpenAICompatibleProvider; print('OK')"
```

This will fail because `vlm/__init__.py` imports `manager.py` which doesn't exist yet. That's expected — it will resolve in Task 4. The provider file itself is correct.

- [ ] **Step 3: Commit**

```bash
git add backend/app/models/vlm/providers/openai_compatible.py
git commit -m "feat: add OpenAI-compatible VLM provider"
```

---

### Task 4: Create models.yaml and VLMManager

**Files:**
- Create: `backend/app/models/vlm/models.yaml`
- Create: `backend/app/models/vlm/manager.py`

- [ ] **Step 1: Create the model registry YAML**

```yaml
# backend/app/models/vlm/models.yaml
models:
  - id: "qwen-vl-local"              # Unique identifier — sent by frontend to select this model
    name: "Qwen 2.5 VL 3B (Local)"   # Display name shown in the frontend dropdown
    provider: "openai_compatible"     # Provider class to use (maps to providers/ module)
    base_url: "http://localhost:8001/v1"  # API endpoint for this provider
    model_id: "Qwen/Qwen2.5-VL-3B-Instruct"  # Model name sent in API requests
    api_key_env: null                 # .env variable holding the API key (null = no auth, uses "none")
    system_prompt: >                  # System message prepended to every conversation
      Ban la tro ly mua sam cua RunShop, cua hang giay chay bo.
      Tra loi bang tieng Viet, ngan gon va huu ich.
    max_tokens: 256                   # Maximum tokens for model generation
    temperature: 0                    # Sampling temperature (0 = deterministic)

  - id: "gpt-4o"                      # Unique identifier — sent by frontend to select this model
    name: "GPT-4o (OpenAI)"           # Display name shown in the frontend dropdown
    provider: "openai_compatible"     # Provider class to use (maps to providers/ module)
    base_url: "https://api.openai.com/v1"  # API endpoint for this provider
    model_id: "gpt-4o"               # Model name sent in API requests
    api_key_env: "OPENAI_API_KEY"     # .env variable holding the API key
    system_prompt: >                  # System message prepended to every conversation
      Ban la tro ly mua sam cua RunShop, cua hang giay chay bo.
      Tra loi bang tieng Viet, ngan gon va huu ich.
    max_tokens: 256                   # Maximum tokens for model generation
    temperature: 0                    # Sampling temperature (0 = deterministic)
```

- [ ] **Step 2: Implement the VLMManager**

```python
# backend/app/models/vlm/manager.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .providers.base import VLMProvider
from .providers.openai_compatible import OpenAICompatibleProvider

# Maps provider name in YAML -> provider class
PROVIDER_MAP: dict[str, type[VLMProvider]] = {
    "openai_compatible": OpenAICompatibleProvider,
}


class VLMManager:
    """Loads model configs from YAML and routes generation requests."""

    def __init__(self) -> None:
        self.models: dict[str, dict[str, Any]] = {}
        self.providers: dict[str, VLMProvider] = {}
        self.default_model: str = ""

    def load(self) -> None:
        yaml_path = Path(__file__).parent / "models.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for entry in config["models"]:
            model_id = entry["id"]
            provider_name = entry["provider"]

            if provider_name not in PROVIDER_MAP:
                print(f"[VLMManager] Unknown provider '{provider_name}' for model '{model_id}', skipping.")
                continue

            # Resolve API key from environment
            api_key_env = entry.get("api_key_env")
            if api_key_env:
                api_key = os.environ.get(api_key_env, "")
                if not api_key:
                    print(f"[VLMManager] Warning: env var '{api_key_env}' not set for model '{model_id}'.")
                    api_key = "none"
            else:
                api_key = "none"

            provider_cls = PROVIDER_MAP[provider_name]
            provider = provider_cls(
                base_url=entry["base_url"],
                api_key=api_key,
                model_id=entry["model_id"],
            )

            self.models[model_id] = entry
            self.providers[model_id] = provider

            if not self.default_model:
                self.default_model = model_id

        print(f"[VLMManager] Loaded {len(self.providers)} model(s): {list(self.providers.keys())}")

    def list_models(self) -> list[dict[str, str]]:
        return [
            {"id": model_id, "name": cfg["name"]}
            for model_id, cfg in self.models.items()
        ]

    def generate(self, model_id: str | None, messages: list[dict]) -> str:
        resolved_id = model_id if model_id and model_id in self.providers else self.default_model

        if not resolved_id or resolved_id not in self.providers:
            raise RuntimeError("No VLM models are configured.")

        config = self.models[resolved_id]
        provider = self.providers[resolved_id]

        # Prepend system prompt
        system_prompt = config.get("system_prompt", "").strip()
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        return provider.generate(
            messages=messages,
            max_tokens=config.get("max_tokens", 256),
            temperature=config.get("temperature", 0),
        )
```

- [ ] **Step 3: Verify the full import chain works**

```bash
cd backend && python -c "from app.models.vlm import VLMManager; m = VLMManager(); print('OK')"
```

Expected: prints `OK` (no YAML loaded yet, just instantiation).

- [ ] **Step 4: Commit**

```bash
git add backend/app/models/vlm/models.yaml backend/app/models/vlm/manager.py
git commit -m "feat: add VLMManager and models.yaml model registry"
```

---

### Task 5: Wire up backend (main.py, chat.py, config.py, requirements)

**Files:**
- Modify: `backend/app/main.py`
- Modify: `backend/app/routers/chat.py`
- Modify: `backend/app/config.py`
- Modify: `backend/requirements.txt`
- Modify: `backend/.env.example`
- Delete: `backend/app/services/vlm_service.py`

- [ ] **Step 1: Update requirements.txt**

Replace the full file with:

```
fastapi
uvicorn[standard]
pydantic
pydantic-settings
pillow
openai
pyyaml
```

Removed `vllm` and `bitsandbytes`. Added `openai` and `pyyaml`.

- [ ] **Step 2: Update config.py — remove VLM settings**

Replace the full file with:

```python
# backend/app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    class Config:
        env_file = ".env"


settings = Settings()
```

The VLM model name and device are now in `models.yaml`.

- [ ] **Step 3: Update .env.example**

Replace the full file with:

```
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB=oe_vlm_shop
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=products
EMBEDDING_MODEL=all-MiniLM-L6-v2
OPENAI_API_KEY=your-openai-api-key-here
```

- [ ] **Step 4: Update main.py — use VLMManager in lifespan**

Replace the full file with:

```python
# backend/app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.models.vlm import VLMManager
from app.routers import chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = VLMManager()
    manager.load()
    app.state.vlm_manager = manager
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

app.mount("/images", StaticFiles(directory="images"), name="images")


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Update chat.py — use manager, add model_id, add /api/models**

Replace the full file with:

```python
# backend/app/routers/chat.py
import base64
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    image_urls: list[str] = []
    model_id: str | None = None


class ChatResponse(BaseModel):
    reply: str


def resolve_image_url(url: str) -> str | None:
    """Resolve an image source to a data URI suitable for OpenAI-style image_url content.

    Supports base64 data URIs (passed through) and local /images/ paths
    (converted to base64 data URIs).
    """
    if url.startswith("data:"):
        return url
    if url.startswith("/images/"):
        path = Path("images") / url.removeprefix("/images/")
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            suffix = path.suffix.lstrip(".").lower()
            mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "jpeg")
            return f"data:image/{mime};base64,{data}"
    return None


@router.get("/models")
async def list_models(request: Request):
    manager = request.app.state.vlm_manager
    return {"models": manager.list_models()}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    manager = request.app.state.vlm_manager
    message = body.message.strip()

    # Resolve image data URI (use first valid one)
    image_data_url: str | None = None
    for url in body.image_urls:
        image_data_url = resolve_image_url(url)
        if image_data_url is not None:
            break

    # Build OpenAI-style messages (without system prompt — manager handles that)
    messages: list[dict] = []

    # Add conversation history (last 4 messages)
    for msg in body.history[-4:]:
        messages.append({"role": msg.role, "content": msg.content})

    # Build current user message content
    if image_data_url is not None:
        user_content: list[dict] = [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": message},
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": message})

    try:
        reply = manager.generate(body.model_id, messages)
    except Exception as exc:
        print(f"VLM generation error: {exc}")
        return ChatResponse(reply="Fail to response!")

    return ChatResponse(reply=reply)
```

Key changes:
- Router prefix changed from `/api/chat` to `/api` (so it can serve both `/api/chat` and `/api/models`).
- `chat()` endpoint now takes `Request` to access `app.state.vlm_manager`.
- System prompt removed — manager injects it.
- `model_id` from request body is passed to `manager.generate()`.
- New `GET /api/models` endpoint.

- [ ] **Step 6: Delete vlm_service.py**

```bash
rm backend/app/services/vlm_service.py
```

- [ ] **Step 7: Verify the backend starts**

```bash
cd backend && python -c "from app.main import app; print('App created OK')"
```

Expected: prints `App created OK` (lifespan doesn't run on import, only on server start).

- [ ] **Step 8: Commit**

```bash
git add backend/app/main.py backend/app/routers/chat.py backend/app/config.py backend/requirements.txt backend/.env.example
git rm backend/app/services/vlm_service.py
git commit -m "feat: wire VLMManager into FastAPI, replace vlm_service with provider pattern"
```

---

### Task 6: Frontend — add model selector dropdown

**Files:**
- Modify: `frontend/src/pages/PlaygroundPage.tsx`

- [ ] **Step 1: Add model state and fetch logic**

At the top of the `PlaygroundPage` component (after the existing state declarations around line 83), add:

```tsx
const [models, setModels] = useState<{ id: string; name: string }[]>([]);
const [selectedModel, setSelectedModel] = useState<string>("");

useEffect(() => {
  fetch("/api/models")
    .then((res) => res.json())
    .then((data) => {
      setModels(data.models);
      if (data.models.length > 0) setSelectedModel(data.models[0].id);
    })
    .catch(() => {});
}, []);
```

- [ ] **Step 2: Update chatAPI to accept model_id**

Change the `chatAPI` function (around line 36) from:

```tsx
async function chatAPI(
  message: string,
  history: { role: string; content: string }[],
  imageUrls: string[] = [],
): Promise<{ reply: string }> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, image_urls: imageUrls }),
  });
  if (!res.ok) throw new Error("Request failed");
  return res.json();
}
```

To:

```tsx
async function chatAPI(
  message: string,
  history: { role: string; content: string }[],
  imageUrls: string[] = [],
  modelId: string = "",
): Promise<{ reply: string }> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      history,
      image_urls: imageUrls,
      model_id: modelId || undefined,
    }),
  });
  if (!res.ok) throw new Error("Request failed");
  return res.json();
}
```

- [ ] **Step 3: Pass selectedModel to chatAPI call**

In the `handleSend` function (around line 175), change:

```tsx
const data = await chatAPI(apiContent, history, imageUrls);
```

To:

```tsx
const data = await chatAPI(apiContent, history, imageUrls, selectedModel);
```

- [ ] **Step 4: Add model selector dropdown next to mic button**

In the action row of the input area (around line 552), locate the `<div className="flex items-center gap-1">` that contains the image upload and mic buttons. Add the model selector dropdown after the mic button, inside that same flex container.

Change this block (the `<div className="flex items-center gap-1">` starting around line 553):

```tsx
<div className="flex items-center gap-1">
  <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />
  <button
    onClick={() => fileRef.current?.click()}
    className="p-2 rounded-lg transition-colors"
    style={{ color: TEXT_MUTED }}
    onMouseEnter={(e) => {
      e.currentTarget.style.background = `${BORDER}`;
      e.currentTarget.style.color = TEXT_SECONDARY;
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.background = "transparent";
      e.currentTarget.style.color = TEXT_MUTED;
    }}
  >
    <ImagePlus size={18} />
  </button>
  <button
    className="p-2 rounded-lg transition-colors"
    style={{ color: TEXT_MUTED }}
    onMouseEnter={(e) => {
      e.currentTarget.style.background = `${BORDER}`;
      e.currentTarget.style.color = TEXT_SECONDARY;
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.background = "transparent";
      e.currentTarget.style.color = TEXT_MUTED;
    }}
  >
    <Mic size={18} />
  </button>
</div>
```

To:

```tsx
<div className="flex items-center gap-1">
  <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />
  <button
    onClick={() => fileRef.current?.click()}
    className="p-2 rounded-lg transition-colors"
    style={{ color: TEXT_MUTED }}
    onMouseEnter={(e) => {
      e.currentTarget.style.background = `${BORDER}`;
      e.currentTarget.style.color = TEXT_SECONDARY;
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.background = "transparent";
      e.currentTarget.style.color = TEXT_MUTED;
    }}
  >
    <ImagePlus size={18} />
  </button>
  <button
    className="p-2 rounded-lg transition-colors"
    style={{ color: TEXT_MUTED }}
    onMouseEnter={(e) => {
      e.currentTarget.style.background = `${BORDER}`;
      e.currentTarget.style.color = TEXT_SECONDARY;
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.background = "transparent";
      e.currentTarget.style.color = TEXT_MUTED;
    }}
  >
    <Mic size={18} />
  </button>
  {models.length > 0 && (
    <select
      value={selectedModel}
      onChange={(e) => setSelectedModel(e.target.value)}
      className="text-xs rounded-lg px-2 py-1.5 outline-none cursor-pointer transition-colors"
      style={{
        color: TEXT_SECONDARY,
        background: "transparent",
        border: `1px solid ${BORDER}`,
        maxWidth: 160,
      }}
    >
      {models.map((m) => (
        <option key={m.id} value={m.id}>
          {m.name}
        </option>
      ))}
    </select>
  )}
</div>
```

Uses a native `<select>` element for simplicity — it matches the minimalist input area style and avoids pulling in the heavier ShadCN Select (which uses a portal/popover that could conflict with the input area layout).

- [ ] **Step 5: Update footer text**

Change line 606:

```tsx
<p className="text-center mt-2.5 text-[11px]" style={{ color: TEXT_MUTED }}>
  AI Playground sử dụng LLaVA-1.5-7b-hf. Kết quả có thể không chính xác.
</p>
```

To:

```tsx
<p className="text-center mt-2.5 text-[11px]" style={{ color: TEXT_MUTED }}>
  AI Playground sử dụng các mô hình AI. Kết quả có thể không chính xác.
</p>
```

- [ ] **Step 6: Update sidebar footer text**

Change line 304:

```tsx
Powered by LLaVA via vLLM
```

To:

```tsx
Powered by OE-VLM
```

- [ ] **Step 7: Verify frontend compiles**

```bash
cd frontend && npm run build
```

Expected: Build succeeds with no TypeScript errors.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/pages/PlaygroundPage.tsx
git commit -m "feat: add model selector dropdown to playground input area"
```

---

### Task 7: Manual integration verification

- [ ] **Step 1: Install new backend dependencies**

```bash
cd backend && pip install openai pyyaml
```

- [ ] **Step 2: Start the backend**

```bash
cd backend && uvicorn app.main:app --reload --port 8000
```

Check console output for: `[VLMManager] Loaded N model(s): [...]`

- [ ] **Step 3: Test /api/models endpoint**

```bash
curl http://localhost:8000/api/models
```

Expected: `{"models":[{"id":"qwen-vl-local","name":"Qwen 2.5 VL 3B (Local)"},{"id":"gpt-4o","name":"GPT-4o (OpenAI)"}]}`

- [ ] **Step 4: Test /api/chat with model_id**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Xin chào", "model_id": "qwen-vl-local"}'
```

Expected: Either a response from the model (if vLLM is running on port 8001) or an error message like "Fail to response!" (if no VLM server is running — this is fine, it means the routing works).

- [ ] **Step 5: Start frontend and verify dropdown**

```bash
cd frontend && npm run dev
```

Open `http://localhost:5173` and navigate to AI Playground. Verify:
- Model dropdown appears next to the mic button
- Dropdown is populated with models from the backend
- Sending a message includes the selected model
