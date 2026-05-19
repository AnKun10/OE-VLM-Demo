#!/bin/bash
# OE-VLM-Demo bootstrap on top of vastai/vllm:v0.20.0-cuda-13.0 with
# AgilePruner visual-token pre-pruning patch applied to vLLM.
# Brings up: (1) patched vLLM Qwen3-VL on :8003, (2) FastAPI backend on :8000,
# (3) Vite dev server on :5173. All three are accessed via SSH tunnel.
# Idempotent: safe to re-run after the pod restarts.
# The patch overlays Python files from the AgilePruner fork onto the
# pre-installed vLLM (no C++ rebuild required since v0.20.0 base matches).
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

# AgilePruner-specific configuration
AP_AGILEPRUNER_ENABLE="${AP_AGILEPRUNER_ENABLE:-true}"
AP_AGILEPRUNER_RATIO="${AP_AGILEPRUNER_RATIO:-0.5}"
AP_AGILEPRUNER_TAU_MAX="${AP_AGILEPRUNER_TAU_MAX:-0.25}"
AP_AGILEPRUNER_ERANK_AVG="${AP_AGILEPRUNER_ERANK_AVG:-95.0}"

REPO_DIR=/workspace/oe-vlm-demo
BACKEND_VENV=/workspace/.venvs/oe-backend

export HF_HOME="${HF_HOME:-/workspace/.hf_cache}"

# -----------------------------------------------------------------------------
# [1/6] System packages: tmux + git + python3-venv + curl already there;
# add Node.js (NodeSource 20.x) for the frontend, only if missing.
# -----------------------------------------------------------------------------
if [ "$OE_ENABLE_FRONTEND" = "true" ] && ! command -v node >/dev/null 2>&1; then
  echo "[1/6] Installing Node.js 20.x via NodeSource..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
else
  echo "[1/6] Node.js: $(command -v node >/dev/null && node --version || echo skipped)"
fi

apt-get install -y python3-venv tmux git >/dev/null 2>&1 || true

# -----------------------------------------------------------------------------
# [2/6] Repo: clone if OE_REPO_URL is set and dir doesn't exist.
# If OE_REPO_URL is unset, we expect the user to rsync the tree manually
# (see README "Manual repo bring-up").
# -----------------------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
  echo "[2/6] Repo already at $REPO_DIR. Pulling latest from $OE_BRANCH..."
  git -C "$REPO_DIR" fetch origin "$OE_BRANCH" || true
  git -C "$REPO_DIR" checkout "$OE_BRANCH" || true
  git -C "$REPO_DIR" pull --ff-only origin "$OE_BRANCH" || true
elif [ -n "$OE_REPO_URL" ]; then
  echo "[2/6] Cloning $OE_REPO_URL (branch $OE_BRANCH) into $REPO_DIR..."
  git clone --depth 1 --branch "$OE_BRANCH" "$OE_REPO_URL" "$REPO_DIR"
else
  echo "[2/6] OE_REPO_URL is empty and $REPO_DIR has no .git/."
  echo "       Skipping clone — rsync your working tree to $REPO_DIR before"
  echo "       this script will be useful. See README 'Manual repo bring-up'."
  echo "       Continuing with vLLM bring-up only."
  OE_ENABLE_BACKEND=false
  OE_ENABLE_FRONTEND=false
fi

# -----------------------------------------------------------------------------
# [3/6] AgilePruner patch: overlay the in-tree vllm-patches/ onto the
# pre-installed vLLM. The patch files live in OE-VLM-Demo at vllm-patches/
# (already cloned in step [2/6]). Pin: vLLM v0.20.0 — matches the Docker
# image. No git clone of a vLLM fork required.
# -----------------------------------------------------------------------------
if [ "$AP_AGILEPRUNER_ENABLE" = "true" ]; then
  AP_PATCHES_DIR="$REPO_DIR/vllm-patches"
  if [ ! -d "$AP_PATCHES_DIR" ]; then
    echo "[3/6] WARNING: $AP_PATCHES_DIR not found. AgilePruner disabled."
    AP_AGILEPRUNER_ENABLE=false
  else
    # Resolve the Python interpreter that has vllm installed. The vast image
    # often has vllm in a different venv (e.g. /venv/main/bin/python) than
    # the system 'python3' shim, so a bare `python3 -c "import vllm"` fails
    # even though the `vllm` CLI works. The CLI's shebang points at the right
    # interpreter; fall back to python3 only if the shebang parse fails.
    VLLM_PY=$(head -1 "$(which vllm 2>/dev/null)" 2>/dev/null | sed 's|^#!||' | awk '{print $1}' || true)
    if [ -z "$VLLM_PY" ] || [ ! -x "$VLLM_PY" ]; then
      VLLM_PY=python3
    fi
    echo "[3/6] Using Python interpreter: $VLLM_PY"
    # Verify pin matches the Docker image's vLLM version.
    PIN_EXPECTED=$(cat "$AP_PATCHES_DIR/PIN.txt" 2>/dev/null | tr -d '[:space:]' || echo "unknown")
    VLLM_VERSION=$("$VLLM_PY" -c "import vllm; print('v' + vllm.__version__)" 2>/dev/null || echo "unknown")
    if [ "$PIN_EXPECTED" != "$VLLM_VERSION" ]; then
      echo "[3/6] WARNING: pin mismatch (patches expect $PIN_EXPECTED, image has $VLLM_VERSION)."
      echo "       Overlay will proceed but may break vLLM. Set AP_AGILEPRUNER_ENABLE=false to skip."
    fi
    VLLM_SITE=$("$VLLM_PY" -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || true)
    if [ -z "$VLLM_SITE" ] || [ ! -d "$VLLM_SITE" ]; then
      echo "[3/6] WARNING: could not locate pre-installed vLLM. AgilePruner disabled."
      AP_AGILEPRUNER_ENABLE=false
    else
      echo "[3/6] Overlaying vllm-patches/ into $VLLM_SITE ..."
      SENTINEL="$VLLM_SITE/.agilepruner_patched"
      if [ ! -f "$SENTINEL" ]; then
        cp -n "$VLLM_SITE/model_executor/models/qwen3_vl.py" \
              "$VLLM_SITE/model_executor/models/qwen3_vl.py.orig" 2>/dev/null || true
        cp -n "$VLLM_SITE/engine/arg_utils.py" \
              "$VLLM_SITE/engine/arg_utils.py.orig" 2>/dev/null || true
      fi
      cp "$AP_PATCHES_DIR/vllm/model_executor/models/qwen3_vl.py" \
         "$VLLM_SITE/model_executor/models/qwen3_vl.py"
      cp "$AP_PATCHES_DIR/vllm/model_executor/models/agilepruner.py" \
         "$VLLM_SITE/model_executor/models/agilepruner.py"
      cp "$AP_PATCHES_DIR/vllm/engine/arg_utils.py" \
         "$VLLM_SITE/engine/arg_utils.py"
      touch "$SENTINEL"
      echo "[3/6] AgilePruner patch applied (pin $PIN_EXPECTED). Originals saved as *.orig (one-time)."
    fi
  fi
else
  echo "[3/6] AP_AGILEPRUNER_ENABLE=false — using stock vLLM."
fi

# -----------------------------------------------------------------------------
# [4/6] vLLM: launch on $VLLM_PORT under tmux if not already serving.
# Mirrors the open-webui template's pattern (which detects on :8000).
# -----------------------------------------------------------------------------
if ! curl -sf -m 2 "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
  if ! tmux has-session -t vllm 2>/dev/null; then
    echo "[4/6] vLLM not serving on :${VLLM_PORT} — launching it under tmux..."
    AP_ARGS=""
    if [ "$AP_AGILEPRUNER_ENABLE" = "true" ]; then
      AP_ARGS="--agilepruner-enable --agilepruner-ratio ${AP_AGILEPRUNER_RATIO} --agilepruner-tau-max ${AP_AGILEPRUNER_TAU_MAX} --agilepruner-erank-avg ${AP_AGILEPRUNER_ERANK_AVG}"
      echo "[4/6] AgilePruner flags: $AP_ARGS"
    fi
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
        ${AP_ARGS} \
      2>&1 | tee /workspace/logs/vllm.log"
  else
    echo "[4/6] vLLM tmux session already exists, leaving it alone."
  fi
else
  echo "[4/6] vLLM already serving on :${VLLM_PORT}, skip launch."
fi

echo "[4/6] Waiting for vLLM /health on :${VLLM_PORT}..."
until curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null; do sleep 5; done
echo "[4/6] vLLM is ready."

# -----------------------------------------------------------------------------
# [5/6] Backend: persistent venv on /workspace, install deps, run uvicorn.
# -----------------------------------------------------------------------------
if [ "$OE_ENABLE_BACKEND" = "true" ]; then
  if [ ! -x "$BACKEND_VENV/bin/uvicorn" ]; then
    echo "[5/6] Creating backend venv (one-time, ~2-3 min)..."
    /usr/bin/python3 -m venv "$BACKEND_VENV"
    "$BACKEND_VENV/bin/pip" install --upgrade pip
  fi
  echo "[5/6] Installing backend requirements..."
  "$BACKEND_VENV/bin/pip" install --quiet -r "$REPO_DIR/backend/requirements.txt"

  # Ensure images dir exists so the static mount in main.py succeeds.
  mkdir -p "$REPO_DIR/backend/images"

  tmux kill-session -t backend 2>/dev/null || true
  tmux new -d -s backend "\
    cd $REPO_DIR/backend && \
    $BACKEND_VENV/bin/uvicorn app.main:app \
      --host 127.0.0.1 --port ${OE_BACKEND_PORT} \
    2>&1 | tee /workspace/logs/backend.log"
  echo "[5/6] Backend launched on :${OE_BACKEND_PORT} (tmux session 'backend')."

  # Brief sanity check (don't block).
  sleep 3
  curl -sf "http://127.0.0.1:${OE_BACKEND_PORT}/health" >/dev/null 2>&1 \
    && echo "[5/6] Backend /health OK." \
    || echo "[5/6] Backend not responding yet (may still be starting). See backend.log."
else
  echo "[5/6] OE_ENABLE_BACKEND=false → skip."
fi

# -----------------------------------------------------------------------------
# [6/6] Frontend: npm install (cached on /workspace) + vite dev.
# -----------------------------------------------------------------------------
if [ "$OE_ENABLE_FRONTEND" = "true" ]; then
  echo "[6/6] Installing frontend deps (cached at /workspace/.npm-cache)..."
  ( cd "$REPO_DIR/frontend" && npm config set cache /workspace/.npm-cache && npm install --silent )

  tmux kill-session -t frontend 2>/dev/null || true
  tmux new -d -s frontend "\
    cd $REPO_DIR/frontend && \
    npm run dev -- --host 127.0.0.1 --port ${OE_FRONTEND_PORT} \
    2>&1 | tee /workspace/logs/frontend.log"
  echo "[6/6] Frontend launched on :${OE_FRONTEND_PORT} (tmux session 'frontend')."
else
  echo "[6/6] OE_ENABLE_FRONTEND=false → skip."
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
  /workspace/logs/vllm.log       — Qwen3-VL inference (AgilePruner patch active if AP_AGILEPRUNER_ENABLE=true)
  /workspace/logs/backend.log    — FastAPI
  /workspace/logs/frontend.log   — Vite

Tmux sessions: tmux ls
  vllm     — Qwen3-VL on :${VLLM_PORT}
  backend  — uvicorn on :${OE_BACKEND_PORT}
  frontend — vite on :${OE_FRONTEND_PORT}
EOF
