# VLM Abstraction Layer Design

## Problem

The current VLM integration is tightly coupled to vLLM's Python API (`LLM.chat()`), with the model loaded in-process. The system prompt is hardcoded in the chat router. This makes it impossible to support multiple models (local or cloud) or switch between them at runtime.

## Goal

Restructure the VLM layer so that:

- The backend is a **client** that calls external VLM servers via the OpenAI SDK
- Multiple models can be configured simultaneously (local vLLM, OpenAI, future Anthropic, etc.)
- The user selects which model to use from a frontend dropdown
- Each model has its own system prompt template
- Adding a new provider type (e.g., Anthropic) requires only one new file

## Folder Structure

```
backend/app/models/
├── __init__.py
├── vlm/
│   ├── __init__.py              # Exports VLMManager
│   ├── manager.py               # Loads YAML, routes generate() to correct provider
│   ├── models.yaml              # Model registry
│   └── providers/
│       ├── __init__.py
│       ├── base.py              # Abstract VLMProvider base class
│       └── openai_compatible.py # OpenAI SDK client (vLLM, OpenAI, etc.)
├── memory/
│   └── __init__.py              # Placeholder for future memory features
└── summary/
    └── __init__.py              # Placeholder for future summarization features
```

## Model Registry (`models.yaml`)

```yaml
models:
  - id: "qwen-vl-local"              # Unique identifier, sent by frontend
    name: "Qwen 2.5 VL 3B (Local)"   # Display name for frontend dropdown
    provider: "openai_compatible"     # Which provider class to use
    base_url: "http://localhost:8001/v1"  # Provider API endpoint
    model_id: "Qwen/Qwen2.5-VL-3B-Instruct"  # Model name sent to the API
    api_key_env: null                 # Env variable for API key (null = no auth)
    system_prompt: >                  # System prompt prepended to every request
      Ban la tro ly mua sam cua RunShop, cua hang giay chay bo.
      Tra loi bang tieng Viet, ngan gon va huu ich.
    max_tokens: 256                   # Max tokens for generation
    temperature: 0                    # Sampling temperature

  - id: "gpt-4o"
    name: "GPT-4o (OpenAI)"
    provider: "openai_compatible"
    base_url: "https://api.openai.com/v1"
    model_id: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"     # Reads API key from this .env variable
    system_prompt: >
      Ban la tro ly mua sam cua RunShop, cua hang giay chay bo.
      Tra loi bang tieng Viet, ngan gon va huu ich.
    max_tokens: 256
    temperature: 0
```

### Adding a new model

To add a model that uses an existing provider (e.g., another OpenAI-compatible endpoint), just add an entry to `models.yaml`. No code changes needed.

To add a new provider type (e.g., Anthropic), create `providers/anthropic.py` implementing `VLMProvider`, and register the provider name in the manager's provider map.

## Provider Abstraction

### Base class (`providers/base.py`)

```python
from abc import ABC, abstractmethod

class VLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        """Send messages and return the model's text response."""
        ...
```

Minimal interface. `messages` follows OpenAI-style format (`[{role, content}]`).

### OpenAI-compatible provider (`providers/openai_compatible.py`)

```python
from openai import OpenAI
from .base import VLMProvider

class OpenAICompatibleProvider(VLMProvider):
    def __init__(self, base_url: str, api_key: str, model_id: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id

    def generate(self, messages, max_tokens, temperature) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
```

Covers vLLM served as OpenAI-compatible, OpenAI API, and any `/v1/chat/completions` endpoint. For local vLLM with no auth, `api_key` is set to `"none"`.

## VLM Manager (`manager.py`)

```python
class VLMManager:
    def __init__(self):
        self.models: dict          # id -> model config from YAML
        self.providers: dict       # id -> instantiated VLMProvider
        self.default_model: str    # first model id in YAML

    def load(self):
        # 1. Read models.yaml
        # 2. For each model, resolve api_key from os.environ using api_key_env
        #    (null -> "none")
        # 3. Instantiate the correct provider class based on "provider" field
        # 4. Store in self.providers keyed by model id

    def list_models(self) -> list[dict]:
        # Returns [{id, name}, ...] for the frontend dropdown

    def generate(self, model_id: str, messages: list[dict]) -> str:
        # 1. Look up provider by model_id (fall back to default if not found)
        # 2. Look up model config for max_tokens, temperature, system_prompt
        # 3. Prepend system_prompt as first message
        # 4. Call provider.generate()
        # 5. Return text response
```

- System prompt injection happens here, not in the router.
- Instantiated once at app startup via FastAPI lifespan.
- Provider map: `{"openai_compatible": OpenAICompatibleProvider}`. Extend this dict when adding new provider types.

## API Changes

### Modified: `POST /api/chat`

Request body adds `model_id`:

```python
class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    image_urls: list[str] = []
    model_id: str | None = None  # None -> use default model
```

Response unchanged: `{"reply": "..."}`.

### New: `GET /api/models`

Returns available models for the frontend dropdown:

```json
{
  "models": [
    {"id": "qwen-vl-local", "name": "Qwen 2.5 VL 3B (Local)"},
    {"id": "gpt-4o", "name": "GPT-4o (OpenAI)"}
  ]
}
```

## Changes to Existing Files

| File | Change |
|------|--------|
| `services/vlm_service.py` | **Deleted** — replaced by `models/vlm/` |
| `main.py` | Lifespan creates `VLMManager`, calls `manager.load()`, stores on `app.state`. Removes `vlm_service` imports. |
| `config.py` | Removes `vlm_model_name` and `vlm_device` (moved to YAML). File kept for future non-VLM settings. |
| `routers/chat.py` | Uses `manager.generate(model_id, messages)` from `app.state`. Removes hardcoded system prompt. Keeps image resolution logic. |
| `requirements.txt` | Adds `openai`, `pyyaml`. Removes `vllm`, `bitsandbytes`. |
| `.env.example` | Adds `OPENAI_API_KEY` example. Removes VLM model/device vars. |

## Frontend Changes

- **Model selector dropdown** placed next to the mic button in the chat input area (bottom-left of the input bar).
- On mount, fetches `GET /api/models` to populate the dropdown options.
- Selected `model_id` is sent with each `POST /api/chat` request.
- Defaults to the first model in the list.

## Placeholder Folders

- `backend/app/models/memory/` — empty `__init__.py`. Reserved for future conversation memory/RAG features.
- `backend/app/models/summary/` — empty `__init__.py`. Reserved for future conversation summarization features.
