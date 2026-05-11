#!/bin/bash
# OE-VLM-Demo bootstrap on top of vastai/vllm:v0.20.0-cuda-13.0.
# Brings up: (1) vLLM Qwen3-VL on :8003, (2) FastAPI backend on :8000,
# (3) Vite dev server on :5173. All three are accessed via SSH tunnel.
# Idempotent: safe to re-run after the pod restarts.
set -e

mkdir -p /workspace/logs /workspace/.venvs /workspace/.hf_cache /workspace/.npm-cache

exec > >(tee -a /workspace/logs/onstart.log) 2>&1
echo "=== OE-VLM-Demo bootstrap on vastai/vllm: $(date) ==="

# -----------------------------------------------------------------------------
# Configuration (env vars with sensible defaults)
# -----------------------------------------------------------------------------
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8003}"
OE_REPO_URL="${OE_REPO_URL:-}"
OE_BRANCH="${OE_BRANCH:-dev/AnKun10}"
OE_BACKEND_PORT="${OE_BACKEND_PORT:-8000}"
OE_FRONTEND_PORT="${OE_FRONTEND_PORT:-5173}"
OE_ENABLE_BACKEND="${OE_ENABLE_BACKEND:-true}"
OE_ENABLE_FRONTEND="${OE_ENABLE_FRONTEND:-true}"

REPO_DIR=/workspace/oe-vlm-demo
BACKEND_VENV=/workspace/.venvs/oe-backend

export HF_HOME="${HF_HOME:-/workspace/.hf_cache}"

# -----------------------------------------------------------------------------
# [1/5] System packages: tmux + git + python3-venv + curl already there;
# add Node.js (NodeSource 20.x) for the frontend, only if missing.
# -----------------------------------------------------------------------------
if [ "$OE_ENABLE_FRONTEND" = "true" ] && ! command -v node >/dev/null 2>&1; then
  echo "[1/5] Installing Node.js 20.x via NodeSource..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
else
  echo "[1/5] Node.js: $(command -v node >/dev/null && node --version || echo skipped)"
fi

apt-get install -y python3-venv tmux git >/dev/null 2>&1 || true

# -----------------------------------------------------------------------------
# [2/5] Repo: clone if OE_REPO_URL is set and dir doesn't exist.
# If OE_REPO_URL is unset, we expect the user to rsync the tree manually
# (see README "Manual repo bring-up").
# -----------------------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
  echo "[2/5] Repo already at $REPO_DIR. Pulling latest from $OE_BRANCH..."
  git -C "$REPO_DIR" fetch origin "$OE_BRANCH" || true
  git -C "$REPO_DIR" checkout "$OE_BRANCH" || true
  git -C "$REPO_DIR" pull --ff-only origin "$OE_BRANCH" || true
elif [ -n "$OE_REPO_URL" ]; then
  echo "[2/5] Cloning $OE_REPO_URL (branch $OE_BRANCH) into $REPO_DIR..."
  git clone --depth 1 --branch "$OE_BRANCH" "$OE_REPO_URL" "$REPO_DIR"
else
  echo "[2/5] OE_REPO_URL is empty and $REPO_DIR has no .git/."
  echo "       Skipping clone — rsync your working tree to $REPO_DIR before"
  echo "       this script will be useful. See README 'Manual repo bring-up'."
  echo "       Continuing with vLLM bring-up only."
  OE_ENABLE_BACKEND=false
  OE_ENABLE_FRONTEND=false
fi

# -----------------------------------------------------------------------------
# [3/5] vLLM: launch on $VLLM_PORT under tmux if not already serving.
# Mirrors the open-webui template's pattern (which detects on :8000).
# -----------------------------------------------------------------------------
if ! curl -sf -m 2 "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
  if ! tmux has-session -t vllm 2>/dev/null; then
    echo "[3/5] vLLM not serving on :${VLLM_PORT} — launching it under tmux..."
    # NOTE: --enforce-eager is required for Qwen3-VL multimodal on vLLM 0.20.
    # See open-webui/vast-templates/qwen3-vl-8b/TEMPLATE_FIELDS.md for the
    # full rationale.
    tmux new -d -s vllm "\
      export HF_HOME=${HF_HOME}; \
      vllm serve ${VLLM_MODEL} \
        --host 127.0.0.1 --port ${VLLM_PORT} \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code \
        --dtype float16 \
        --served-model-name qwen3-vl-8b \
        --enforce-eager \
        --download-dir ${HF_HOME} \
        --limit-mm-per-prompt '{\"image\":4}' \
      2>&1 | tee /workspace/logs/vllm.log"
  else
    echo "[3/5] vLLM tmux session already exists, leaving it alone."
  fi
else
  echo "[3/5] vLLM already serving on :${VLLM_PORT}, skip launch."
fi

echo "[3/5] Waiting for vLLM /health on :${VLLM_PORT}..."
until curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null; do sleep 5; done
echo "[3/5] vLLM is ready."

# -----------------------------------------------------------------------------
# [4/5] Backend: persistent venv on /workspace, install deps, run uvicorn.
# -----------------------------------------------------------------------------
if [ "$OE_ENABLE_BACKEND" = "true" ]; then
  if [ ! -x "$BACKEND_VENV/bin/uvicorn" ]; then
    echo "[4/5] Creating backend venv (one-time, ~2-3 min)..."
    /usr/bin/python3 -m venv "$BACKEND_VENV"
    "$BACKEND_VENV/bin/pip" install --upgrade pip
  fi
  echo "[4/5] Installing backend requirements..."
  "$BACKEND_VENV/bin/pip" install --quiet -r "$REPO_DIR/backend/requirements.txt"

  # Ensure images dir exists so the static mount in main.py succeeds.
  mkdir -p "$REPO_DIR/backend/images"

  tmux kill-session -t backend 2>/dev/null || true
  tmux new -d -s backend "\
    cd $REPO_DIR/backend && \
    $BACKEND_VENV/bin/uvicorn app.main:app \
      --host 127.0.0.1 --port ${OE_BACKEND_PORT} \
    2>&1 | tee /workspace/logs/backend.log"
  echo "[4/5] Backend launched on :${OE_BACKEND_PORT} (tmux session 'backend')."

  # Brief sanity check (don't block).
  sleep 3
  curl -sf "http://127.0.0.1:${OE_BACKEND_PORT}/health" >/dev/null 2>&1 \
    && echo "[4/5] Backend /health OK." \
    || echo "[4/5] Backend not responding yet (may still be starting). See backend.log."
else
  echo "[4/5] OE_ENABLE_BACKEND=false → skip."
fi

# -----------------------------------------------------------------------------
# [5/5] Frontend: npm install (cached on /workspace) + vite dev.
# -----------------------------------------------------------------------------
if [ "$OE_ENABLE_FRONTEND" = "true" ]; then
  echo "[5/5] Installing frontend deps (cached at /workspace/.npm-cache)..."
  ( cd "$REPO_DIR/frontend" && npm config set cache /workspace/.npm-cache && npm install --silent )

  tmux kill-session -t frontend 2>/dev/null || true
  tmux new -d -s frontend "\
    cd $REPO_DIR/frontend && \
    npm run dev -- --host 127.0.0.1 --port ${OE_FRONTEND_PORT} \
    2>&1 | tee /workspace/logs/frontend.log"
  echo "[5/5] Frontend launched on :${OE_FRONTEND_PORT} (tmux session 'frontend')."
else
  echo "[5/5] OE_ENABLE_FRONTEND=false → skip."
fi

cat <<EOF

=== Bootstrap done. ===
SSH tunnel from your laptop:
  ssh -L ${OE_FRONTEND_PORT}:127.0.0.1:${OE_FRONTEND_PORT} \\
      -L ${OE_BACKEND_PORT}:127.0.0.1:${OE_BACKEND_PORT} \\
      -L ${VLLM_PORT}:127.0.0.1:${VLLM_PORT} \\
      -p <SSH_PORT> root@<VAST_IP>

Then open http://localhost:${OE_FRONTEND_PORT}/playground in your browser.

Logs:
  /workspace/logs/onstart.log    — this script
  /workspace/logs/vllm.log       — Qwen3-VL inference
  /workspace/logs/backend.log    — FastAPI
  /workspace/logs/frontend.log   — Vite

Tmux sessions: tmux ls
  vllm     — Qwen3-VL on :${VLLM_PORT}
  backend  — uvicorn on :${OE_BACKEND_PORT}
  frontend — vite on :${OE_FRONTEND_PORT}
EOF
