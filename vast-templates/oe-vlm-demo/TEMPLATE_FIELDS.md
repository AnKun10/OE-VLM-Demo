# Vast Template Fields — OE-VLM-Demo (Qwen3-VL 8B + FastAPI + Vite)

This file is the canonical source for the values you paste into the Vast.ai
Console template editor. Keep it open side-by-side with the Vast Console.

> Onstart source: `vast-templates/oe-vlm-demo/onstart.sh`
> Reference template: `open-webui/vast-templates/qwen3-vl-8b/` (the open-webui
> setup we adapted from). Read its `TEMPLATE_FIELDS.md` for the rationale
> behind shared decisions (image choice, `--enforce-eager`, port mapping
> philosophy).

## 1. Identification

| Field | Value |
|---|---|
| Template Name | `OE-VLM-Demo - Qwen3-VL 8B (FastAPI + Vite)` |
| Template Description | `Qwen3-VL 8B served via vLLM on :8003 + OE-VLM-Demo FastAPI backend on :8000 + Vite dev server on :5173. All three reached via SSH tunnel. Repo auto-cloned from OE_REPO_URL on first boot.` |

## 2. Docker Repository And Environment

| Field | Value |
|---|---|
| Image Path:Tag | `vastai/vllm:v0.20.0-cuda-13.0` |
| Version Tag | `v0.20.0-cuda-13.0` (pick from dropdown) |

Same image as the open-webui template. vLLM 0.20 ships pre-installed; no
install delay.

## 3. Docker Options (port mapping)

```
-p 1111:1111 -p 7860:7860 -p 8080:8080 -p 8265:8265
```

Same as open-webui template. **Don't expose 8000 / 5173 / 8003** — reach them
via SSH tunnel only.

| Port | Why mapped | Why NOT mapped |
|---|---|---|
| 1111 | Instance Portal | — |
| 7860 | image's vLLM web UI (unused but pre-bound) | — |
| 8080 | image's Jupyter | — |
| 8265 | image's Ray dashboard | — |
| 8003 | — | vLLM API has no auth; SSH-tunnel only |
| 8000 | — | FastAPI backend; SSH-tunnel only |
| 5173 | — | Vite dev server; SSH-tunnel only |

## 4. Environment Variables

Paste each row as a Key/Value pair. Order does not matter.

| Key | Value |
|---|---|
| `OPEN_BUTTON_PORT` | `1111` |
| `OPEN_BUTTON_TOKEN` | `1` |
| `JUPYTER_DIR` | `/` |
| `DATA_DIRECTORY` | `/workspace/` |
| `PORTAL_CONFIG` | `localhost:1111:11111:/:Instance Portal\|localhost:7860:17860:/:vLLM UI\|localhost:8080:18080:/:Jupyter\|localhost:8265:18265:/:Ray Dashboard` |
| `VLLM_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` |
| `VLLM_PORT` | `8003` |
| `VLLM_ARGS` | `--max-model-len 32768 --gpu-memory-utilization 0.85 --trust-remote-code --dtype float16 --served-model-name qwen3-vl-8b --enforce-eager --download-dir /workspace/.hf_cache --limit-mm-per-prompt {"image":4}` |
| `AUTO_PARALLEL` | `false` |
| `HF_HOME` | `/workspace/.hf_cache` |
| `OE_REPO_URL` | `https://github.com/<your-user>/OE-VLM-Demo.git` |
| `OE_BRANCH` | `dev/AnKun10` |
| `OE_BACKEND_PORT` | `8000` |
| `OE_FRONTEND_PORT` | `5173` |
| `OE_ENABLE_FRONTEND` | `true` |
| `OE_ENABLE_BACKEND` | `true` |

### Critical notes (verbatim from the open-webui template — same image, same constraints)

- **`--enforce-eager` IS required for Qwen3-VL multimodal on vLLM 0.20.**
  Without it, vLLM's CUDA graph captures a fixed-size deepstack buffer (~82
  tokens). Real images often produce more (we observed 88 for a 64×64 PNG)
  → engine crashes with `Requested more deepstack tokens than available in
  buffer`. Trade-off: ~10–20% throughput penalty vs. CUDA-graph mode, but
  this is the only known-working setup for Qwen3-VL on this vLLM version.
- **Don't add `RAY_ADDRESS` / `RAY_ARGS`** — those are for multi-GPU tensor
  parallel, which we explicitly disable via `AUTO_PARALLEL=false`.
- **Don't add `--mm-encoder-attn-backend TORCH_SDPA`** — vLLM 0.20 doesn't
  recognize this flag and the engine fails to initialise.

### OE-VLM-Demo–specific notes

- **`VLLM_PORT=8003`** matches `backend/app/models/vlm/models.yaml` which
  hardcodes `base_url: "http://localhost:8003/v1"` for `qwen3-vl-8b-vllm`.
  If you change the port here, also update `models.yaml`.
- **`OE_REPO_URL`** must be reachable from the pod with no auth, OR include
  a PAT (e.g., `https://<user>:<token>@github.com/...`). If the repo is
  private and you don't want to embed credentials, leave this blank and
  rsync the working tree from your laptop after the pod boots — see the
  README's "Manual repo bring-up" section.
- **`OE_BRANCH`** picks which branch the onstart checks out. Defaults to
  `dev/AnKun10` until Phase 1 merges to `main`.
- **`OE_ENABLE_FRONTEND=false`** to skip Vite (e.g., when you only want the
  backend for `curl` testing or for the smoke script). The `npm install`
  step is still skipped.
- **`OE_ENABLE_BACKEND=false`** to run vLLM only (debugging upstream).

### Optional: add Open WebUI alongside (if you want both)

If you also want Open WebUI on the same pod (handy for cross-checking),
add these and the onstart will skip them otherwise:

| Key | Value |
|---|---|
| `OPENWEBUI_ENABLE` | `false` (keep false unless you really need both) |

The OE-VLM-Demo template ignores Open WebUI vars by default. Switching this
to `true` would require re-introducing the open-webui-specific blocks from
the reference template; we don't ship them here to keep the script tight.

## 5. On-start Script

Paste the entire contents of `vast-templates/oe-vlm-demo/onstart.sh` into
the "On-start Script" textarea. Keep edits in the repo file so changes are
version-controlled.

## 6. Common pitfalls

- **`PORTAL_CONFIG` literal `|` (pipe) characters**: paste raw `|`, not
  `\|`. The markdown table escapes pipes only for rendering.
- **Quotes around values**: do NOT add surrounding `"..."` to env-var
  values. The Vast UI handles quoting itself; extra quotes become part of
  the value.
- **`VLLM_ARGS`**: a single long string with space-separated flags, all on
  one line. The `{"image":4}` JSON inside it has no whitespace — Vast's
  shell-style parsing chokes on spaces inside JSON values.
- **GPU offer**: when renting from this template, pick a GPU with **≥ 24 GB
  VRAM** (RTX 3090, 4090, A5000, A6000). Smaller GPUs cannot fit
  Qwen3-VL-8B at fp16 + 32k context.
- **Persistent storage**: pick an offer with a "Volume" / persistent
  `/workspace`. Otherwise the model weights, repo clone, and Python venv
  all redownload on every restart (~20 min penalty).

## 7. Verification after save

When you click "Save" in Vast Console:
- Template appears in your Templates list with the new name.
- "Use" button on the template card lets you rent an instance from it.
- The original open-webui template is unaffected.
