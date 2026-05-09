# Development Roadmap: Qwen & FLARE (Fusion) Integration

Context: our VLM backend (`backend/app/models/vlm/`) already uses a provider
abstraction with an OpenAI-compatible adapter. This doc plans how to wire up
**Qwen2.5-VL** and **FLARE (Fusion)** as selectable models, and captures the
traps we already hit while implementing these in `lmms-eval` so we don't pay
for them twice.

Source of truth for the lessons below:
- `lmms-eval/lmms_eval/models/simple/fusion.py`
- `lmms-eval/lmms_eval/models/simple/qwen2_5_vl.py`
- `lmms-eval/lmms_eval/models/chat/qwen2_5_vl.py`
- `lmms-eval/lmms_eval/models/chat/vllm.py` (in-process vLLM wrapper)

---

## Phase 1 — Qwen2.5-VL via vLLM OpenAI-compatible server (easy, do first)

Serve Qwen outside the backend, add a YAML entry, reuse the existing
`openai_compatible.py` provider. Zero backend code changes.

### Tasks
- [ ] Stand up vLLM server on GPU host:
  ```bash
  vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
      --port 8001 \
      --gpu-memory-utilization 0.85 \
      --max-model-len 32768 \
      --limit-mm-per-prompt image=4
  ```
- [ ] Add entry in `backend/app/models/vlm/models.yaml`:
  - `provider: openai_compatible`
  - `base_url: http://<host>:8001/v1`
  - `model: Qwen/Qwen2.5-VL-7B-Instruct`
  - `api_key_env: VLLM_API_KEY` (set to any string; vLLM ignores it)
  - `system_prompt: "You are a helpful assistant."`
- [ ] Verify end-to-end: text-only chat, then text+image, via `/playground`.
- [ ] Add a smoke-test script that POSTs a known image+question pair and
      checks the response is non-empty before deploy.

### Gotchas we already hit (Qwen)

1. **`\n\n` as a stop token truncates output.** In `qwen2_5_vl.py:214` we
   explicitly filter it out. If you ever expose `stop` via the API, strip
   `\n\n` server-side or output will be cut mid-paragraph.
2. **`<image>` placeholder leakage.** If upstream text contains a literal
   `<image>` token, Qwen's processor will break. Strip it in the adapter
   (`qwen2_5_vl.py:220-221`) before sending.
3. **`max_pixels` / `min_pixels` are mandatory knobs, not cosmetic.**
   Working range: `min_pixels = 256*28*28 = 200704`,
   `max_pixels = 1605632`. Too small → blurry; too large → OOM.
   Must be passed to the processor AND into each image content part.
4. **`qwen-vl-utils` is a required pip dep** (not transitive). Without it,
   `process_vision_info` import fails silently. Pin in `requirements.txt`.
5. **`padding_side` flips with batch size.** `left` when batch>1, `right`
   when batch==1 (`qwen2_5_vl.py:302`). If using transformers directly,
   reproduce this or generation produces garbage on batched requests.
6. **Video frame sampling is manual.** `process_vision_info` returns all
   frames; you must `np.linspace`-downsample to `max_num_frames`. Default
   32 is fine for chat but the last frame must be appended explicitly or
   ending shots get cut (`qwen2_5_vl.py:298-301`).
7. **`attn_implementation` allowlist.** Only `None`, `flash_attention_2`,
   `sdpa`, `eager` are valid. `flash_attention_2` needs CUDA 11.8+ and a
   matching wheel; prefer `sdpa` unless you've confirmed the wheel works.
8. **vLLM chat_template for VL models.** vLLM 0.6.x sometimes needs
   `--chat-template examples/tool_chat_template_qwen2vl.jinja` on older
   checkpoints. If multi-image responses look like single-image responses,
   this is why. Check vLLM release notes for your version.
9. **`gpu_memory_utilization` default (0.9) OOMs mid-request** when other
   processes share the GPU. Pin to 0.80–0.85 and leave headroom for the
   KV cache + image preprocessing buffers.
10. **vLLM's `.chat()` is in-process, not HTTP.** The lmms-eval
    `chat/vllm.py` wrapper uses `self.client.chat()` on an in-process
    `LLM()` instance — don't confuse it with our deployment model. For
    OE-VLM-Demo we use the `vllm serve` HTTP server.

---

## Phase 2 — FLARE (Fusion) sidecar service (hard)

FLARE is **not vLLM-compatible** — custom model class, custom image token
handling, custom `instructs` second input. Wrap it in a minimal FastAPI
sidecar exposing `/v1/chat/completions`, reuse our OpenAI provider.

### Architecture
```
OE-VLM-Demo backend  ──(OpenAI SDK)──>  flare-sidecar :8002
                                         │
                                         ├─ loads fusion model once at startup
                                         ├─ /v1/chat/completions endpoint
                                         └─ translates OpenAI messages → FUSION prompt
```

### Tasks
- [ ] Set up FLARE repo: `git clone https://github.com/starriver030515/FLARE`
      and `pip install -e ./FLARE` in the sidecar venv.
- [ ] New service `services/flare-sidecar/` with:
  - `main.py` — FastAPI app with `/v1/chat/completions` (non-streaming
    first; streaming is Phase 3).
  - `engine.py` — wraps `fusion.model.builder.load_pretrained_model` and
    the prompt-construction logic from `fusion.py:443-548`.
  - `Dockerfile` — CUDA base image, installs FLARE + torch 2.1+.
- [ ] Add `models.yaml` entry pointing at sidecar's `/v1` base URL.
- [ ] Load-test with 10 concurrent requests; FLARE is single-threaded per
      GPU, so add a request queue + semaphore on the sidecar.

### Gotchas we already hit (Fusion) — these cost us 10+ commits

1. **`inv_freq` device mismatch on Phi3-based FUSION checkpoints.** This
   was the #1 time sink. `from_pretrained(device_map=...)` leaves
   non-persistent buffers on CPU; transformers <4.48 doesn't add
   `.to(x.device)` in the rotary forward, so the first matmul crashes.
   See `fusion.py:47-107,176-239` for the full fix. Three things are
   required **together**:
   - Strip accelerate hooks (`_hf_hook`, `hf_device_map`) before `.to()`.
   - Bypass `PreTrainedModel.to()` with `torch.nn.Module.to(model, device)`
     — the PreTrained version silently skips non-persistent buffers.
   - Wrap each `RotaryEmbedding.forward` at **instance level** to force
     buffers onto the target device on every call (dynamic rope may
     recreate `inv_freq` on CPU mid-run).
   - Also apply a class-level monkey patch as belt-and-braces.
2. **`conv_templates` entries contain tuples, not lists.** `deepcopy` then
   `conv.messages = [list(m) for m in conv.messages]` — otherwise
   `append_message` errors on the second call because tuples are
   immutable. See `fusion.py:396-397,505-506`. Without this, **the first
   request works, the second one crashes** — don't fall for it in dev.
3. **Conversation templates are globally mutable.** If you forget the
   `deepcopy`, the previous request's messages leak into the next one.
   Symptom: responses look like the model is talking to itself.
4. **`device_map="cuda"` (not `"auto"`) to match FLARE builder defaults.**
   FLARE's `load_pretrained_model` has its own device handling; `"auto"`
   collides with it. Pin to an explicit device string.
5. **`IMAGE_PLACEHOLDER` handling is non-trivial.** The prompt may contain
   `<image-placeholder>`, `<image>`, or nothing — each path needs
   different substitution with `DEFAULT_IM_START/END_TOKEN`. See
   `fusion.py:383-393`. Getting this wrong produces coherent-but-blind
   responses (model ignores the image).
6. **`instructs` is a separate tokenized input.** FUSION takes both
   `input_ids` (the full prompt with image tokens) AND `instructs` (the
   question, stripped of image/special tokens). You cannot skip
   `instructs` — `fusion_arch.py` uses `instructs.ne(0)` as a mask.
   Pad value MUST be 0 (not eos_id) for the mask to work.
7. **Dtype must be `float16` for images.** `process_images` output must
   be cast to `torch.float16` (`fusion.py:375,481`). `bfloat16` silently
   produces wrong outputs on the custom vision encoder.
8. **`mm_use_im_start_end` is a config flag that changes prompt format.**
   Check `self.model.config.mm_use_im_start_end` — different checkpoints
   use different conventions. See `fusion.py:385-393`.
9. **Auto-detect conv template from the checkpoint name** (see
   `_detect_conv_template` in `fusion.py:109-127`). Wrong template →
   garbage output. Common rules:
   - `phi` in name → `phi_3`
   - `qwen` in name → `qwen_2`
   - `llama-2` → `llava_llama_2`
   - otherwise → `llava_v0`
10. **Multi-turn is not supported** in the lmms-eval wrapper
    (`fusion.py:564` raises `NotImplementedError`). If we want chat
    history in the demo, the sidecar must build the multi-turn prompt
    itself by concatenating into the same conv template before
    `get_prompt()`. Test this carefully — the `None` sentinel on the
    assistant role is how FUSION knows where to generate.
11. **Single-GPU, single-process.** FLARE has no tensor parallelism. Size
    your GPU accordingly (8B model needs ~20GB fp16 + context). Queue
    requests in the sidecar; don't try to parallelize.
12. **Image size list must match image tensor count.** `image_sizes` is
    passed alongside `images=` to `generate()` and must be a list of
    `(w, h)` tuples with the same length. Mismatch = silent wrong
    crop/resize.

---

## Phase 3 — Streaming, multi-turn polish, observability

Only after Phase 1 & 2 are stable.

- [ ] Streaming for Qwen (vLLM supports `stream=true` out of the box).
- [ ] Streaming for FLARE — needs a custom `TextIteratorStreamer` in the
      sidecar; requires refactoring the blocking `.generate()` into a
      background thread.
- [ ] Multi-turn history trimming (token budget per model).
- [ ] Per-request metrics (TTFT, tokens/sec) exposed on `/metrics` for
      the backend to log.
- [ ] Authentication on the sidecars (simple bearer token; currently
      both listen on LAN).

---

## Open questions to resolve before starting

- Where does the GPU host live? Vast.ai box (per `backend/vast_ai.txt`),
  or an on-prem machine?
- Concurrency target: 1 user at a time, or N concurrent? This drives
  queue design in the FLARE sidecar.
- Image size caps — do we resize client-side before upload, or let the
  backend handle it? Affects `max_pixels` tuning and upload latency.
- Do we need conversation persistence? The placeholder modules in
  `backend/app/models/memory/` and `summary/` suggest yes, but it's not
  on any current list.

---

## Quick reference: file pointers in lmms-eval we learned from

| Concern | File:line |
|---|---|
| FLARE load + device fixes | `lmms_eval/models/simple/fusion.py:168-239` |
| FLARE prompt construction | `lmms_eval/models/simple/fusion.py:381-422` |
| FLARE conv deepcopy fix | `lmms_eval/models/simple/fusion.py:396-397` |
| Qwen load | `lmms_eval/models/simple/qwen2_5_vl.py:81-92` |
| Qwen image/video message build | `lmms_eval/models/simple/qwen2_5_vl.py:223-310` |
| Qwen multi-turn | `lmms_eval/models/simple/qwen2_5_vl.py:371-581` |
| vLLM chat (in-process) | `lmms_eval/models/chat/vllm.py:86-121` |
| Base generate interface | `lmms_eval/api/model.py:84,103` |
