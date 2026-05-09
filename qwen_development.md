# Qwen3-VL-8B — Operational Notes & Deferred Work

Companion to `qwen3_vl_design.md`. Covers vast.ai operations, deferred quirks, future phases, and manual verification.

## Operational Notes

### vLLM start command (on vast.ai GPU host)

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8003 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 4}'
```

### SSH tunnel (from local dev machine)

Extend the pattern in `vast_ai.txt`:

```bash
ssh -L 5173:localhost:5173 \
    -L 8000:localhost:8000 \
    -L 8003:localhost:8003 \
    -p <SSH_PORT> root@<VAST_IP>
```

### Port allocation

- `8003` — Qwen3-VL 8B

## Deferred Quirks

Numbering matches the brainstorming-scope question. Each item maps back to a gotcha in `development_roadmap.md` Phase 1.

### (iii) Stop-token sanitization

Qwen uses `\n\n` as a stop token, which can truncate mid-paragraph. Add a stop-sanitizer to `transforms.py` if/when `models.yaml` starts passing a `stop` knob through to the provider. Not needed today.

### (v) Video input with `max_num_frames` cap

vLLM can accept video; the Qwen processor needs explicit frame sampling. Add when the frontend begins uploading video (`image_urls` generalizes to `media_urls`). Wire a YAML-driven `max_num_frames` into `config.py` and a new transform.

### (vi) Per-request system prompt override

Today `manager.py` prepends the YAML `system_prompt` on every request. If the playground UI adds a system-prompt field, route it through the `/api/chat` body and let it override the YAML value per request.

## Future Phases (from `development_roadmap.md` Phase 3)

- Streaming (`stream=true`) — vLLM supports it; the backend router + frontend need to handle Server-Sent Events.
- Multi-turn history trimming to a token budget per model.
- Per-request metrics on `/metrics` (TTFT, tokens/sec).
- Bearer-token auth on the vLLM endpoint.

## Manual Verification Checklist

After deploy:

- [ ] `/api/models` returns the `qwen3-vl-8b-vllm` entry.
- [ ] Text-only chat via `/playground` returns a non-empty reply.
- [ ] Text + image chat via `/playground` returns a reply that references the image.
- [ ] Kill the vLLM server mid-request → frontend shows "Lỗi kết nối: …" (not a 500).
- [ ] `python backend/scripts/smoke_qwen3_vl.py <image> "<question>"` exits 0.

## Rationale Pointers

| Quirk | Handled in | Source |
|---|---|---|
| Image-token leakage (i) | `transforms.strip_image_tokens` | `development_roadmap.md` Phase 1 gotcha 2 |
| `min_pixels` / `max_pixels` (ii) | `transforms.inject_pixel_bounds`, `config.DEFAULT_*` | `development_roadmap.md` Phase 1 gotcha 3 |
| Connection retry / timeout (iv) | `provider.QwenVLLMProvider`, `config.REQUEST_TIMEOUT_S` | `development_roadmap.md` Phase 1 gotcha 9 (partial) |
