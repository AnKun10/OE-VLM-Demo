# Design: Qwen3-VL-8B backend integration

Date: 2026-04-17
Status: Approved (brainstorming complete — pending implementation plan)
Scope: Add Qwen3-VL-8B as a selectable model via a Qwen-aware provider talking to a remote `vllm serve` over an SSH-tunneled HTTP endpoint. Replace the LLaVA entry in `models.yaml`. Record deferred work in `qwen_development.md`.

Related reading: `development_roadmap.md` (Phase 1 — Qwen via vLLM).

## Goals

- Add a new model entry `qwen3-vl-8b-vllm` that selects the new provider from the existing frontend dropdown.
- Introduce a decomposed provider package that other model families can use as a template.
- Handle three Qwen-specific backend concerns: image-token leakage, `min_pixels` / `max_pixels` injection, and connection retry.
- Leave everything else (routing, manager, chat router, lifespan, frontend) untouched.

## Non-goals

- Running vLLM in-process or using `transformers` directly.
- Streaming, multi-turn polish, video input, per-request system prompt override — recorded in `qwen_development.md`, not built now.
- Automated tests for the end-to-end `/api/chat` path (covered by a manual smoke script).

## Architecture

```
Frontend (playground, dropdown picks "qwen3-vl-8b-vllm")
    │
    ▼
POST /api/chat                           (unchanged)
    │
    ▼
chat.py router                           (unchanged — already builds OpenAI-style image_url parts)
    │
    ▼
VLMManager.generate()                    (unchanged plumbing; +1 line in PROVIDER_MAP,
    │                                     +read optional pixel-bound overrides from YAML)
    ▼
QwenVLLMProvider                         ◄── NEW
    ├─ strip_image_tokens(messages)
    ├─ inject_pixel_bounds(messages, min_px, max_px)
    ├─ OpenAI SDK call with timeout + 1 retry
    └─ return content
    │
    ▼
vLLM HTTP server                         (http://localhost:8003/v1
                                          via SSH tunnel to vast.ai GPU host)
```

All Qwen-awareness is contained in the new provider package. Swapping to another model is a YAML edit.

## Component decomposition

New package `backend/app/models/vlm/providers/qwen_vllm/`:

```
providers/qwen_vllm/
├─ __init__.py       # exports QwenVLLMProvider
├─ config.py         # ALL constants — regexes, default pixel bounds, timeout, retry count
├─ transforms.py     # pure functions — strip_image_tokens(), inject_pixel_bounds()
└─ provider.py       # QwenVLLMProvider class — SDK call + retry, composes transforms
```

### Template contract for future model packages

Every model-specific provider package MUST follow this 4-file shape:

| File | Responsibility | Rule |
|---|---|---|
| `config.py` | All constants and defaults for that model. | **Only** file to touch when tuning hyperparams. No imports from `transforms` or `provider`. |
| `transforms.py` | Pure functions over OpenAI-format messages. | No SDK, no I/O, no logging. Must be unit-testable without network. |
| `provider.py` | `VLMProvider` subclass. Wires `config` + `transforms` around the OpenAI SDK call with retry. | Only file that imports `openai`. |
| `__init__.py` | Single export: the provider class. | One line. |

### `config.py` contents for Qwen3-VL-8B

```python
# Regex patterns stripped from user text (quirk i — image-token leakage)
IMAGE_TOKEN_PATTERNS: tuple[str, ...] = (
    r"<image>",
    r"<\|image_pad\|>",
    r"<\|vision_start\|>",
    r"<\|vision_end\|>",
)

# Default pixel bounds for the Qwen processor (quirk ii).
# Per roadmap: min = 256*28*28 = 200 704, working max = 1 605 632.
DEFAULT_MIN_PIXELS: int = 200_704
DEFAULT_MAX_PIXELS: int = 1_605_632

# HTTP policy (quirk iv)
REQUEST_TIMEOUT_S: float = 120.0
MAX_RETRIES: int = 1
RETRY_BACKOFF_S: float = 1.0
```

### `transforms.py` responsibilities

Two pure functions:

- `strip_image_tokens(messages: list[dict]) -> list[dict]` — walks messages, removes each pattern from string user content, leaves `image_url` parts and non-user turns alone. Returns a new list (does not mutate input).
- `inject_pixel_bounds(messages: list[dict], min_px: int, max_px: int) -> list[dict]` — walks messages, attaches `"min_pixels"` and `"max_pixels"` to each `image_url` dict. No-op for text-only messages. Returns a new list.

### `provider.py` responsibilities

`QwenVLLMProvider(VLMProvider)` with constructor signature:

```python
def __init__(
    self,
    base_url: str,
    api_key: str,
    model_id: str,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> None
```

- Builds an `OpenAI` client with `timeout=REQUEST_TIMEOUT_S`.
- Resolves `min_pixels` / `max_pixels` from constructor args (YAML override) falling back to `config.DEFAULT_*`.
- `generate(messages, max_tokens, temperature)`:
  1. `messages = strip_image_tokens(messages)`
  2. `messages = inject_pixel_bounds(messages, min_px, max_px)`
  3. Call `chat.completions.create(...)` wrapped in a 1-retry loop on `APIConnectionError` with `RETRY_BACKOFF_S` sleep between attempts.
  4. On exhaustion, raise `ConnectionError` with a message identifying the model and base URL (matches the contract of the existing provider, so `chat.py`'s handler still shows "Lỗi kết nối: …").
  5. Pass through `BadRequestError` unchanged.

## Manager and YAML changes

### `backend/app/models/vlm/manager.py`

Two edits:

1. Register the provider:
   ```python
   from .providers.qwen_vllm import QwenVLLMProvider

   PROVIDER_MAP: dict[str, type[VLMProvider]] = {
       "openai_compatible": OpenAICompatibleProvider,
       "qwen_vllm": QwenVLLMProvider,
   }
   ```
2. Pass optional `min_pixels` / `max_pixels` from each YAML entry into the provider constructor. Generic providers (e.g., `openai_compatible`) must keep ignoring fields they don't know about. Cleanest approach: pick known kwargs by provider class (small dispatch) or pass a `config: dict` blob through a new base-class hook. Implementation plan to decide; design accepts either as long as the YAML stays declarative.

### `backend/app/models/vlm/models.yaml`

- Remove the `llava-1.5-7b-local` entry.
- Add:
  ```yaml
  - id: "qwen3-vl-8b-vllm"
    name: "Qwen3-VL 8B (vLLM)"
    provider: "qwen_vllm"
    base_url: "http://localhost:8003/v1"
    model_id: "Qwen/Qwen3-VL-8B-Instruct"
    api_key_env: null
    system_prompt: >
      You are a helpful AI assistant!
    max_tokens: 512
    temperature: 0
    # Optional Qwen-specific overrides — fall back to config.py defaults if omitted.
    min_pixels: 200704
    max_pixels: 1605632
  ```

The existing `qwen-vl-local` (Qwen 2.5-VL 3B on port 8001) and `gpt-5.4-mini` entries stay.

## Data flow

1. Frontend sends `{message, history, image_urls, model_id: "qwen3-vl-8b-vllm"}`.
2. `chat.py` resolves images to data URIs and builds OpenAI-style `image_url` content parts.
3. `VLMManager.generate()` prepends the per-model system prompt and routes to `QwenVLLMProvider`.
4. `QwenVLLMProvider` applies `strip_image_tokens` → `inject_pixel_bounds` → SDK call (with retry) → returns text.
5. Manager returns the text; `chat.py` wraps it in `ChatResponse(reply=...)`.

## Error handling contract

| Failure | Raised | Surface to user |
|---|---|---|
| vLLM server down / unreachable (after 1 retry) | `ConnectionError` | `chat.py` already catches → "Lỗi kết nối: …" |
| vLLM returns 4xx | `BadRequestError` (passes through) | `chat.py` catches bare `Exception` → "Xin lỗi, không thể xử lý yêu cầu." |
| Sanitization produces empty user text | Empty string preserved; vLLM decides | Not the provider's concern. |

No new try/except scaffolding in `chat.py` or `manager.py`.

## Testing

### Unit tests — `backend/tests/vlm/providers/qwen_vllm/`

- `test_transforms.py`
  - `strip_image_tokens` removes each configured pattern from string user content.
  - Leaves `image_url` parts untouched.
  - Leaves assistant turns untouched.
  - Idempotent on already-clean text.
- `test_provider.py`
  - Monkey-patches `OpenAI.chat.completions.create` to raise `APIConnectionError` once then succeed → asserts one retry happened and result returned.
  - Raises `APIConnectionError` twice → asserts `ConnectionError` surfaces with model/base-url in the message.

Add `pytest` to the backend dependencies (exact file — `requirements.txt` vs a new `requirements-dev.txt` — see Open Questions). Run with `pytest backend/tests`.

### Smoke script — `backend/scripts/smoke_qwen3_vl.py`

- POSTs a known local image path + a question to `/api/chat` with `model_id="qwen3-vl-8b-vllm"`.
- Asserts reply is non-empty and does not start with the known error prefixes ("Lỗi kết nối", "Xin lỗi").
- Exits 0 on success, 1 on failure. Meant to be run manually after `vllm serve` is up.

### Manual checklist (lives in `qwen_development.md`)

- Text-only chat via `/playground` returns a reply.
- Text + image chat via `/playground` returns a reply that references the image.
- Kill the vLLM server mid-request → frontend shows "Lỗi kết nối: …", not a 500.

## Files changed

| Path | Change |
|---|---|
| `backend/app/models/vlm/providers/qwen_vllm/__init__.py` | NEW |
| `backend/app/models/vlm/providers/qwen_vllm/config.py` | NEW |
| `backend/app/models/vlm/providers/qwen_vllm/transforms.py` | NEW |
| `backend/app/models/vlm/providers/qwen_vllm/provider.py` | NEW |
| `backend/app/models/vlm/manager.py` | `PROVIDER_MAP` entry + YAML override plumbing |
| `backend/app/models/vlm/models.yaml` | Remove LLaVA entry, add Qwen3-VL-8B entry |
| `backend/requirements.txt` (or new `backend/requirements-dev.txt`) | Add `pytest` |
| `backend/tests/vlm/providers/qwen_vllm/test_transforms.py` | NEW |
| `backend/tests/vlm/providers/qwen_vllm/test_provider.py` | NEW |
| `backend/scripts/smoke_qwen3_vl.py` | NEW |
| `qwen_development.md` | NEW — deferred work, vast.ai operational notes |

## `qwen_development.md` — contents outline

A companion doc at repo root. At minimum:

- **Operational notes**
  - vast.ai start command: `vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8003 --gpu-memory-utilization 0.85 --max-model-len 32768 --limit-mm-per-prompt image=4`
  - SSH tunnel: extend `vast_ai.txt`'s existing pattern with `-L 8003:localhost:8003`.
  - Port 8003 chosen so `8001` (Qwen 2.5 3B) and `8002` (freed by LLaVA removal) stay available for other variants.
- **Deferred quirks** (numbering matches `development_roadmap.md` Phase 1 gotchas)
  - (iii) Stop-token sanitization — add when/if the YAML exposes `stop`.
  - (v) Video input with per-request `max_num_frames` cap — add when the frontend uploads video.
  - (vi) Per-request system prompt override — add when the playground UI exposes it.
- **Future phases**
  - Streaming (Qwen via vLLM supports `stream=true` out of the box once the backend streams).
  - Multi-turn history trimming to a token budget.
  - Per-request metrics on `/metrics`.
  - Bearer-token auth on the vLLM endpoint.
- **Manual verification checklist** (copied from the Testing section).
- **Rationale pointers** — link back to the relevant `development_roadmap.md` gotchas for each handled quirk.

## Open questions (deferred, not blocking)

- Whether YAML overrides for `min_pixels` / `max_pixels` should be validated at load time (e.g., reject `min > max`). The implementation plan can decide; the design accepts a bare pass-through with no validation as a starting point.
- Whether the unit-test directory `backend/tests/` should be added to `.gitignore` exclusions and whether `pytest` belongs in a `requirements-dev.txt` rather than `requirements.txt`. The implementation plan can decide.
