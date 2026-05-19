# OE-VLM-Demo + AgilePruner on Vast.ai — Template

A Vast.ai template that brings up the full OE-VLM-Demo stack with the
**AgilePruner visual-token pre-pruning** patch applied to vLLM. Same pod layout
as the stock `oe-vlm-demo` template (Qwen3-VL-8B on :8003, FastAPI :8000,
Vite :5173, SSH-tunneled), plus a clone of the AgilePruner fork and a
file-overlay step that patches the pre-installed vLLM in place — no C++
rebuild because the fork is pinned to the same v0.20.0 as the Docker image.

For the stock (no-AgilePruner) deployment, see the sibling template
`vast-templates/oe-vlm-demo/`.

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
- **AgilePruner patch:** step [3/6] clones the fork and overlays its Python
  files onto the pre-installed vLLM; step [4/6] passes four `--agilepruner-*`
  flags to vllm serve.

## AgilePruner-specific behavior

The boot adds a new step [3/6] between repo clone and vLLM launch:

1. Clone `${AP_FORK_URL}` branch `${AP_FORK_BRANCH}` to `/workspace/vllm-fork`.
2. Locate the pre-installed vLLM site-packages via `python3 -c "import vllm, os; ..."`.
3. Backup `qwen3_vl.py` and `arg_utils.py` as `*.orig` (one-time, idempotent).
4. Copy the fork's `qwen3_vl.py`, `agilepruner.py` (new file), and `arg_utils.py` into the site-packages.
5. Touch a sentinel `.agilepruner_patched` to mark the install.

The vLLM serve command then appends four flags:
`--agilepruner-enable --agilepruner-ratio $AP_AGILEPRUNER_RATIO --agilepruner-tau-max $AP_AGILEPRUNER_TAU_MAX --agilepruner-erank-avg $AP_AGILEPRUNER_ERANK_AVG`.

To turn the patch off without changing template: set `AP_AGILEPRUNER_ENABLE=false` and restart the pod (or kill+relaunch the `vllm` tmux session after editing env). The stock vLLM bin still boots unmodified because the backup `.orig` files in the vLLM site-packages dir are NOT auto-restored. To restore stock vLLM in place (without a pod restart), see the "Want to revert to stock vLLM" row in the Troubleshooting table below.

If `git clone` of the fork fails (repo private, network issue), the script logs a warning and continues with stock vLLM rather than aborting.

## Deploy procedure

### 1. Configure the Vast template (one-time)

1. Open Vast Console → **Templates → New Template** (or clone an existing one).
2. Open `TEMPLATE_FIELDS.md` from this repo and paste each field exactly as
   shown. Pay attention to `VLLM_PORT=8003`, `OE_REPO_URL`, `OE_BRANCH`, and
   the six `AP_*` env vars.
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

Cold boot: 15–25 min (vLLM weights ~16 GB + AgilePruner fork clone + Python deps + npm install + Node.js install).
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

**AgilePruner-specific verify:** confirm the patch is active by checking the vLLM log for a DEBUG line per image:

```bash
ssh -p <SSH_PORT> root@<VAST_IP>
grep '\[AgilePruner\]' /workspace/logs/vllm.log | head -5
```

Expected output (one line per image processed):

```
[AgilePruner] image_idx=0 N=1024 K=512 erank=87.4 tau_base=...
```

If you see no lines after sending a chat request with an image, either the flag is off (`grep agilepruner /workspace/logs/vllm.log` should show the four flags in the launch command) or the patch didn't apply (`ls /workspace/.agilepruner_patched` should exist — actually that sentinel lives in the vLLM site-packages dir, not /workspace).

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
  bash /root/onstart.sh
```

(Path may differ; the on-start script is the file you pasted in step 1.)

## Operations

### Restart only the backend

```bash
ssh -p <SSH_PORT> root@<VAST_IP>
tmux kill-session -t backend
bash /root/onstart.sh
```

The script's idempotent guards skip vLLM relaunch (already serving), skip
the venv install (already present), and just re-launch the `backend` tmux
session.

### Restart only the frontend

```bash
tmux kill-session -t frontend
bash /root/onstart.sh
```

### Restart only vLLM

```bash
tmux kill-session -t vllm
bash /root/onstart.sh
```

Note: if the AgilePruner patch is already applied (sentinel exists), the patch
step is idempotent — it re-overlays the files (updating from the fork's latest)
but does NOT re-backup originals.

### Pull latest code without restart

```bash
cd /workspace/oe-vlm-demo
git pull
# Backend reload is automatic via uvicorn --reload? No — we run without --reload.
# Restart explicitly:
tmux kill-session -t backend
bash /root/onstart.sh
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
| `vllm.log` shows `unrecognized argument: --agilepruner-enable` | `cat /workspace/logs/onstart.log \| grep '\[3/6\]'` | Patch didn't apply. Either fork clone failed or vllm-site-packages overlay was skipped. Check earlier onstart log for warnings |
| Patch active but `[AgilePruner]` log lines never appear | Send a chat request with an image, then re-check the log | Patch's CLI flags accepted but selection branch unreached. Likely indicates `--agilepruner-enable` was stripped from VLLM_ARGS — check the tmux command |
| Want to revert to stock vLLM without re-renting | SSH, `cd $(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")` | Restore: `mv model_executor/models/qwen3_vl.py.orig model_executor/models/qwen3_vl.py && mv engine/arg_utils.py.orig engine/arg_utils.py && rm model_executor/models/agilepruner.py .agilepruner_patched`. Then `tmux kill-session -t vllm && bash /root/onstart.sh` with `AP_AGILEPRUNER_ENABLE=false` |

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
- **Calibration of `erank_avg` for Qwen3-VL** — the template ships with the
  paper's LLaVA value (95.0). Calibrating on a Qwen3-VL distribution is a
  documented follow-up (see the fork's `AGILEPRUNER.md`).
