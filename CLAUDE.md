# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OE-VLM Shop** is a Vietnamese-language e-commerce chatbot demo with a multi-model VLM (Vision Language Model) backend. Users can chat with different AI models (local or cloud) via text and image, selected from a frontend dropdown.

## Development Commands

### Backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir images
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev      # Vite dev server at http://localhost:5173
npm run build    # tsc + vite build
```

No test or lint commands are configured.

### Running Qwen3-VL-8B (vLLM)

The `qwen3-vl-8b-vllm` model runs on a remote GPU host (vast.ai) and is reached via an SSH tunnel. Ports: `8001` = Qwen 2.5-VL 3B, `8003` = Qwen3-VL 8B. See `qwen_development.md` for full ops notes.

1. **On the GPU host** — start the vLLM server:
   ```bash
   vllm serve Qwen/Qwen3-VL-8B-Instruct \
       --port 8003 \
       --gpu-memory-utilization 0.85 \
       --max-model-len 32768 \
       --limit-mm-per-prompt image=4
   ```

2. **On the local dev machine** — open the SSH tunnel (forwards frontend, backend, and vLLM ports):
   ```bash
   ssh -L 5173:localhost:5173 \
       -L 8000:localhost:8000 \
       -L 8003:localhost:8003 \
       -p <SSH_PORT> root@<VAST_IP>
   ```

3. **Start backend + frontend** using the commands above (`uvicorn ... --port 8000` and `npm run dev`). Pick "Qwen3-VL 8B (vLLM)" in the playground dropdown.

4. **Smoke test** (optional):
   ```bash
   cd backend
   python scripts/smoke_qwen3_vl.py path/to/image.jpg "What is in this image?"
   ```

**AgilePruner mode (optional):** to enable visual-token pre-pruning, install the fork at `https://github.com/AnVu10/vllm` (branch `agilepruner-qwen3vl`) on the GPU host and add these flags:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8003 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --limit-mm-per-prompt image=4 \
    --agilepruner-enable \
    --agilepruner-ratio 0.5 \
    --agilepruner-tau-max 0.25 \
    --agilepruner-erank-avg 95.0
```

Smoke test (local):
```bash
cd backend
python scripts/smoke_agilepruner.py path/to/image.jpg "Describe this image"
```

**Disclaimer:** `--agilepruner-erank-avg 95.0` is the LLaVA training-set value from the paper (Appendix D); it is NOT calibrated for Qwen3-VL. The pruning still works correctly but the adaptive threshold scale is not tuned for this model family. See the fork's `AGILEPRUNER.md` for details.

## Architecture

### Stack
- **Backend**: FastAPI + Uvicorn (Python 3.11+)
- **VLM**: Provider-based abstraction using OpenAI SDK as client
- **Frontend**: React 18 + Vite + TypeScript, TailwindCSS + ShadCN UI

### VLM Provider System

The core abstraction lives in `backend/app/models/vlm/`. This is a strategy pattern:

1. **`models.yaml`** — Declarative model registry. Each entry defines an id, provider type, API endpoint, model name, API key env var, system prompt, and generation params. To add a new model using an existing provider, just add a YAML entry.

2. **`providers/base.py`** — Abstract `VLMProvider` with a single `generate(messages, max_tokens, temperature) -> str` method.

3. **`providers/openai_compatible.py`** — Covers any OpenAI-compatible API (vLLM, OpenAI, etc.) via the `openai` Python SDK. To add a non-OpenAI provider (e.g., Anthropic), create a new file in `providers/`, implement `VLMProvider`, and register it in `manager.py`'s `PROVIDER_MAP`.

4. **`providers/qwen_vllm/`** — Specialized provider for Qwen-family VL models on vLLM. Adds input transforms (`strip_image_tokens`, `inject_pixel_bounds`), per-model `min_pixels`/`max_pixels` from YAML (defaults in `config.py`), a request timeout, and a one-shot retry on `APIConnectionError`. Use the `qwen_vllm` provider type in `models.yaml` for Qwen3-VL.

5. **`manager.py`** — `VLMManager` loads YAML at startup, resolves API keys from environment variables, instantiates providers (via `PROVIDER_MAP` and each provider's `extra_kwargs_from_entry` classmethod hook for provider-specific YAML fields), and routes `generate()` calls. It also prepends the per-model system prompt. Stored on `app.state.vlm_manager` via FastAPI lifespan.

### Request Flow

```
Frontend (PlaygroundPage) → POST /api/chat {message, history, image_urls, model_id}
  → chat.py: resolves images to data URIs, builds OpenAI-style messages
  → VLMManager.generate(): prepends system prompt, routes to correct provider
  → Provider: calls external API via OpenAI SDK → returns text
  → Response: {reply: "..."}
```

### Routes
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat with VLM (text + optional image + model selection) |
| `/api/models` | GET | List available models for frontend dropdown |
| `/health` | GET | Health check |

### Frontend Pages
- **ShopLayout** (`/`, `/products`, `/products/:id`) — E-commerce storefront with navbar, footer, and floating chatbot widget
- **PlaygroundPage** (`/playground`) — Full-page chat interface with model selector dropdown, image upload, conversation history

### API Proxy
Vite dev server proxies `/api/*` to `http://localhost:8000`. Frontend always uses relative `/api` paths.

### Environment Config
Backend reads from `backend/.env` (see `backend/.env.example`). API keys for cloud models (e.g., `OPENAI_API_KEY`) are referenced by name in `models.yaml` via `api_key_env` and resolved from environment at startup.

### Placeholder Modules
`backend/app/models/memory/` and `backend/app/models/summary/` are empty placeholders for future conversation memory and summarization features.
