# OE-VLM-Demo on Vast.ai — Template

A Vast.ai template that brings up the full OE-VLM-Demo stack on a single GPU
pod: vLLM serving Qwen3-VL-8B on :8003, FastAPI backend on :8000, Vite dev
server on :5173. All three are reached over SSH tunnel.

## Files

| File | Purpose |
|---|---|
| `onstart.sh` | Pasted into the Vast template's "On-start Script" field |
| `TEMPLATE_FIELDS.md` | Copy-paste source for every Vast Console field |
| `README.md` | This file — deploy procedure, ops, troubleshooting |

The template is adapted from `open-webui/vast-templates/qwen3-vl-8b/`
(included in the repo as a vendored reference). Differences:

- vLLM listens on `:8003` (not `:8000`), matching `models.yaml`.
- No Open WebUI; instead our FastAPI + Vite stack.
- Adds Node.js install (NodeSource 20.x) for the frontend.
- Repo is cloned from `OE_REPO_URL` (or rsynced manually).

## Deploy procedure

### 1. Configure the Vast template (one-time)

1. Open Vast Console → **Templates → New Template** (or clone an existing one).
2. Open `TEMPLATE_FIELDS.md` from this repo and paste each field exactly as
   shown. Pay attention to `VLLM_PORT=8003`, `OE_REPO_URL`, and `OE_BRANCH`.
3. Open `onstart.sh` from this repo and paste its entire contents into the
   "On-start Script" textarea.
4. **Save**.

### 2. Rent a test instance

1. Templates → click "Use" on the new template.
2. Pick a GPU offer with **≥ 24 GB VRAM** (RTX 3090, 4090, A5000, A6000).
3. Confirm storage is "Volume" / persistent so `/workspace` survives
   restarts.
4. Rent.
5. SSH with port forwards as soon as the pod is `running`:
   ```bash
   ssh -L 5173:127.0.0.1:5173 \
       -L 8000:127.0.0.1:8000 \
       -L 8003:127.0.0.1:8003 \
       -p <SSH_PORT> root@<VAST_IP>
   ```

### 3. Watch the boot

```bash
tail -f /workspace/logs/onstart.log
```

Cold boot: 15–25 min (vLLM weights ~16 GB + Python deps + npm install + Node.js install).
Warm boot (after restart): 2–4 min (everything cached on `/workspace`).

When the script exits cleanly you'll see:

```
=== Bootstrap done. ===
SSH tunnel from your laptop:
  ssh -L 5173:127.0.0.1:5173 ...
```

Three tmux sessions should be running:

```bash
tmux ls
# backend:  1 windows ...
# frontend: 1 windows ...
# vllm:     1 windows ...
```

### 4. Verify

In your laptop browser open `http://localhost:5173/playground`. You should see
the Vietnamese chat interface, the model dropdown listing
`Qwen3-VL 8B (vLLM)`, and be able to send a text+image chat.

If something looks off:

```bash
# Models endpoint (should include capabilities.vision)
curl -s http://localhost:8000/api/models

# Direct vLLM probe
curl -s http://localhost:8003/v1/models

# Smoke script (uses /api/chat non-stream path)
ssh -p <SSH_PORT> root@<VAST_IP>
cd /workspace/oe-vlm-demo/backend
. /workspace/.venvs/oe-backend/bin/activate
python scripts/smoke_qwen3_vl.py /workspace/oe-vlm-demo/backend/images/<some.jpg> "What is in this image?"
```

### 5. Manual repo bring-up (if you can't / don't want to clone from GitHub)

If `OE_REPO_URL` is empty, the onstart short-circuits backend + frontend.
You then bring the repo over yourself:

```bash
# From your laptop (in the project root):
rsync -avz --exclude '.git' --exclude 'node_modules' --exclude 'backend/.venv' \
   --exclude 'backend/images/*' --exclude '.superpowers' \
   ./ root@<VAST_IP>:/workspace/oe-vlm-demo/
```

Then SSH in and re-run the onstart manually:

```bash
ssh -p <SSH_PORT> root@<VAST_IP>
OE_ENABLE_BACKEND=true OE_ENABLE_FRONTEND=true \
  bash /var/lib/vast/onstart.sh
```

(Path may differ; the on-start script is the file you pasted in step 1.)

## Operations

### Restart only the backend

```bash
ssh -p <SSH_PORT> root@<VAST_IP>
tmux kill-session -t backend
bash /var/lib/vast/onstart.sh
```

The script's idempotent guards skip vLLM relaunch (already serving), skip
the venv install (already present), and just re-launch the `backend` tmux
session.

### Restart only the frontend

```bash
tmux kill-session -t frontend
bash /var/lib/vast/onstart.sh
```

### Restart only vLLM

```bash
tmux kill-session -t vllm
bash /var/lib/vast/onstart.sh
```

### Pull latest code without restart

```bash
cd /workspace/oe-vlm-demo
git pull
# Backend reload is automatic via uvicorn --reload? No — we run without --reload.
# Restart explicitly:
tmux kill-session -t backend
bash /var/lib/vast/onstart.sh
# Vite picks up frontend changes via HMR — no restart needed for src/ edits.
```

### Run only vLLM (stop FastAPI / Vite for debugging)

Set `OE_ENABLE_BACKEND=false` and `OE_ENABLE_FRONTEND=false` in the template
env, restart the instance.

## Troubleshooting

| Symptom | Check | Likely cause |
|---|---|---|
| `vllm.log` shows `unrecognized argument: --mm-encoder-attn-backend` | n/a | vLLM 0.20 dropped the flag — confirm it isn't in `VLLM_ARGS` |
| `vllm.log` shows `Requested more deepstack tokens than available in buffer` then engine dies | n/a | `--enforce-eager` missing. The onstart hardcodes it; verify it wasn't edited out |
| Backend `/health` returns 502 / connection refused | `tmux ls` and `tail -f /workspace/logs/backend.log` | Backend not yet up; uvicorn may be importing model packages. Wait 30s |
| `/api/models` returns `{"models": []}` | `tail /workspace/logs/backend.log` | YAML failed to load. Check `backend/app/models/vlm/models.yaml` exists in the cloned repo |
| Backend responds but `/api/chat` returns `Lỗi kết nối` | `curl http://127.0.0.1:8003/v1/models` from the pod | vLLM not reachable. Check `tmux ls` for `vllm` session and `tail /workspace/logs/vllm.log` |
| Frontend loads but the model dropdown is empty | Browser devtools → Network → `/api/models` response | Vite proxy routes `/api/*` → `localhost:8000`. If the response is empty, see the row above |
| `npm install` fails with `EACCES` | n/a | Probably running as non-root. Re-run as root or `sudo`; the onstart already runs as root |
| `npm install` is slow (~5 min cold) | n/a | Expected on first boot. Cached at `/workspace/.npm-cache` after; subsequent boots are fast |
| Vite dev server fails to bind `:5173` | `ss -tlnp \| grep 5173` | Old session leftover. `tmux kill-session -t frontend` and re-run onstart |
| `git clone` fails: `Repository not found` | n/a | Repo is private and `OE_REPO_URL` doesn't have a PAT. Either embed `https://<user>:<token>@github.com/...` or use the manual rsync path |
| `git pull` fails on restart with merge conflicts | `cd /workspace/oe-vlm-demo && git status` | You edited files on the pod. Either commit, stash, or `git reset --hard origin/<branch>` if the edits are disposable |
| Pod restarts and the venv is missing | `ls /workspace/.venvs/oe-backend/bin/uvicorn` | Storage was ephemeral, not persistent. Use a different offer with a Volume |
| `images/` 404s on `/api/files/<id>` | `ls /workspace/oe-vlm-demo/backend/images/` | Files were uploaded but pod restart didn't preserve them. They live in the repo dir on `/workspace`, so a persistent volume keeps them. If repo dir is fresh, that's expected |

## Rollback

The template change is independent of the open-webui template (different
file, different name). To roll back: simply delete this template from the
Vast Console — your existing templates are untouched.

If you've rented a pod from this template and want to abandon it: stop +
destroy the instance from the Vast Console.

## What this template does NOT include

- **Open WebUI** — see `open-webui/vast-templates/qwen3-vl-8b/` if you want
  that frontend instead.
- **Production frontend build** — Vite dev server is fine for a demo. For
  production: `npm run build` and serve the `dist/` via a static-file
  server.
- **Auth, rate limiting, TLS** — none of these are configured. The pod is
  reached over SSH tunnel only, which is the security model.
- **Public model registration** — vLLM's API has `--api-key` for token
  auth, but we don't set it. Fine on SSH-tunnel-only access; do NOT expose
  port 8003 publicly.
