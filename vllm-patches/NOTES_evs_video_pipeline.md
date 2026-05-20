# vLLM v0.20.0 EVS-for-Video Pipeline — Anatomy

All file paths relative to `vllm/` in the upstream clone unless fully qualified.

---

## 1. Pruning origin

- **Function:** `compute_retention_mask`
- **File:** `multimodal/evs.py` lines 38–92
- **Signature:**
  ```python
  def compute_retention_mask(
      video_embeds: torch.Tensor,       # (T*H/ms*W/ms, hidden_size) post-merge
      video_size_thw: tuple[int,int,int],
      spatial_merge_size: int,
      q: float,                          # pruning rate in [0,1)
  ) -> torch.Tensor
  ```
- **Returns:** `torch.Tensor` of dtype `bool`, shape `(T * H//ms * W//ms,)` — a flat retention
  mask over all post-merge tokens across all frames. `True` = keep.
- **Algorithm:** computes cosine dissimilarity between adjacent frames, always forces the
  first frame's tokens to dissimilarity=255 (always retained), then top-k selects
  `max(tokens_per_frame, int(total * (1-q)))` tokens globally.
- **Called from:** `model_executor/models/qwen3_vl.py` lines 2207–2215 inside
  `_postprocess_video_embeds_evs`, inside the per-video loop.

---

## 2. Extra-channel embedding format

- **File:** `model_executor/models/qwen3_vl.py`
- **Where produced for VIDEO:** `_create_final_video_embeddings` (line 2246) + `_get_expanded_positions` (line 2335).
- **Where produced for IMAGE:** `_postprocess_image_embeds_evs` (line 2135).

### Video path — `_create_final_video_embeddings` output

The function returns a tensor of shape `(full_frame_seq_len, D + 5)` where
`full_frame_seq_len` includes both retained video tokens AND the interleaved indicator
tokens (`<vision_start>`, timestamp text, `<vision_end>`), and `D` = visual hidden dim.

The extra 5 channels are computed in `_get_expanded_positions` (line 2356):

```python
expanded_positions = torch.zeros(seq_len, 5, ...)
# Channel layout [dim -5 .. -1]:
# [:, 0]  = t_index  — temporal MRoPE position (from unpruned grid)
# [:, 1]  = h_index  — height  MRoPE position  (from unpruned grid)
# [:, 2]  = w_index  — width   MRoPE position  (from unpruned grid)
# [:, 3]  = is_vision_start — 1 for <vision_start> tokens, 0 elsewhere
# [:, 4]  = is_video — 1 for actual video embedding positions, 0 for indicator tokens
```

For video tokens the t/h/w channels carry the **original unpruned-grid positions** of the
retained tokens (selected via `retention_mask`); for indicator tokens they carry the
corresponding original positions from the unpruned sequence (line 2396–2401).

### Image path — `_postprocess_image_embeds_evs`

Image embeddings (not pruned) also get 5 extra channels (line 2159–2173) for uniformity
with the video path so `recompute_mrope_positions` can process them identically:

```python
positions = compute_mrope_for_media(size, merge_size)   # (N, 4): [t, h, w, llm_grid_w]
positions = torch.cat([positions,
    torch.zeros_like(positions[:, 0:1])],               # 5th dummy channel = 0
    dim=1)
emb = torch.cat([emb, positions], dim=1)                # (N, D+5)
```

Channel 4 (is_video) is always 0 for images — i.e. images use the 4-channel semantics
inside a 5-channel wrapper: `[t, h, w, max_width, dummy=0]`.

### Consumers of the extra channels

1. `model_executor/models/qwen3_vl.py` line 2648 — `_recompute_mrope_positions` reads
   `mm[:, -5:]` to extract positions, then strips them: returns `mm[:, :-5]`.
2. `multimodal/evs.py` lines 248–259 — `recompute_mrope_positions` dispatches on
   `mm_pos.shape[0] == 5` vs `4` to choose the 5-channel (Qwen3-VL) vs 4-channel
   (Qwen2.5-VL) processing path.

---

## 3. `_iter_mm_grid_hw` for video (and image)

**File:** `model_executor/models/qwen3_vl.py` lines 2424–2496.

### Image branch (lines 2453–2458)

```python
if mm_feature.modality == "image":
    t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
    llm_grid_h = h // spatial_merge_size
    llm_grid_w = w // spatial_merge_size
    yield offset, llm_grid_h, llm_grid_w, llm_grid_h * llm_grid_w
```

For images, `actual_num_tokens` is always hardcoded to `llm_grid_h * llm_grid_w` — it reads
from the grid, not from the placeholder. This is why our AgilePruner patch breaks: the
placeholder has fewer tokens after pruning, but `_iter_mm_grid_hw` still yields the full
grid product.

### Video branch (lines 2459–2494)

For each frame, the code:
1. Scans `input_tokens.index(vision_start_token_id, offset)` to locate the frame boundary.
2. Tries `input_tokens.index(video_token_id, offset, vision_end_offset)`.
3. If found: `actual_num_tokens = vision_end_offset - video_offset` (contiguous block formula
   since `get_video_repl` packs all retained tokens consecutively before `<vision_end>`).
4. If not found (0-token frame after EVS): `actual_num_tokens = 0`, frame is skipped in
   position assignment (line 2531).

**Image equivalent:** For AgilePruner images, we would need to similarly scan the actual
placeholder token count from `mm_position.length` rather than computing `h*w`. The simplest
approach: yield `mm_position.length` as `actual_num_tokens` for images when pruning is
enabled.

---

## 4. Position handling for sparse pruned tokens

**File:** `model_executor/models/qwen3_vl.py` lines 2530–2583.

### Path A — `actual_num_tokens == 0` (video-only: entire frame pruned)
Lines 2531–2532: skipped entirely via `continue`. No positions emitted for this frame.

### Path B — Lumped placeholder (video-only)
Lines 2544–2565: when `actual_num_tokens > expected_tokens_per_frame`. This happens when
EVS retains tokens from multiple logical frames in the first frame's placeholder. Iterates
over `num_logical_frames` complete grids plus a `remainder` partial grid. The remainder
takes the first `remainder` positions from a row-major flat grid — **not** the actual
spatial positions of retained tokens. This is acknowledged as "should never be the case"
in the comment.

### Path C — Normal pruned frame
Lines 2566–2570: `actual_num_tokens <= expected_tokens_per_frame`. Falls through to:
```python
grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
llm_pos_ids_list.append(grid_indices + text_len + st_idx)
```
This assigns the FULL dense grid of positions (all `H*W` positions) regardless of how
many tokens actually remain. The result is that position count `!=` token count when
`actual_num_tokens < expected_tokens_per_frame`.

**Critical insight:** Path C does NOT correctly handle sparse pruned images. The correct
positions for retained tokens come from the extra channels (`mm_pos[0:3, :]` in
`recompute_mrope_positions` in `evs.py`). The `_get_mrope_input_positions` /
`_iter_mm_grid_hw` path computes a placeholder that `recompute_mrope_positions` then
OVERWRITES using the embedded positions. The two-phase design is:
1. Phase 1 (`get_mrope_input_positions`): assigns dense grid positions as a placeholder.
2. Phase 2 (`recompute_mrope_positions`): overwrites them with the true sparse positions
   read from the extra channels in the embeddings.

---

## 5. `recompute_mrope_positions`

### Instance wrapper
**File:** `model_executor/models/qwen3_vl.py` lines 2585–2620.  
Signature: `(self, input_ids, multimodal_embeddings, mrope_positions, num_computed_tokens)`  
Simply delegates to the static method after injecting `image_token_id`, `video_token_id`,
`vision_start_token_id` from `self.config`.

### Static implementation (strips extra channels)
**File:** `model_executor/models/qwen3_vl.py` lines 2622–2668.

```python
for mm in multimodal_embeddings:
    if mm.shape[0] > 0:
        mm_embeddings_out.append(mm[:, :-5])          # clean embedding (D dims)
        mm_embeddings_pos.append(mm[:, -5:].permute(1, 0).long())  # (5, N)
```

Returns `(mm_embeddings_out, positions, mrope_positions_delta)` where `mm_embeddings_out`
has channels `D` only — no extra channels — safe for LLM scatter.

### Core logic
**File:** `multimodal/evs.py` lines 154–356.

For each media item's `mm_pos` (shape `(5, N)` for Qwen3-VL):
- Locates the media's start in the global sequence via `vision_start_token_id` scanning.
- If `mm_pos.shape[0] == 5` and `mm_pos[4, :]` (is_video) has any True:
  uses `mm_pos[3, :]` (is_vision_start) to find `num_timestamp_tokens` and adjusts
  `global_mm_start` to account for preceding timestamp text tokens.
- Writes `positions[:, local_start:local_end] = mm_pos[0:3] + base` — overwrites the
  dense-grid placeholder with exact (t, h, w) positions from the embedded channels.
- Cascades text positions after the media span.

For images (`mm_pos[4, :]` all zeros since dummy), `has_video_tokens = False` so the
timestamp-adjustment branch is skipped and positions are written with original logic
(lines 332–338: `base = positions[-1, global_mm_start] + 1`).

### Caller chain
```
gpu_model_runner._gather_mm_embeddings (line 2993-3005)
  -> model.recompute_mrope_positions (line 2996)
     -> _recompute_mrope_positions (qwen3_vl.py:2622)
        -> evs.recompute_mrope_positions (evs.py:154)
```

Called once per request per prefill step, AFTER encoder output is retrieved from cache and
BEFORE the cleaned embeddings are returned to be scattered into `inputs_embeds`.

---

## 6. gpu_model_runner integration

**File:** `v1/worker/gpu_model_runner.py`

### Flow

```
execute_model
  _update_states (line 3833) — sets up CachedRequestState
    _init_mrope_positions (line 1492–1509)
      model.get_mrope_input_positions(prompt_token_ids, mm_features)
      -> assigns dense-grid placeholder positions to req_state.mrope_positions

  _execute_mm_encoder (line 3238)
    model.embed_multimodal(**mm_kwargs)   # returns (D+5) embeddings
    encoder_cache[mm_hash] = output

  _gather_mm_embeddings (line 3239)
    for each mm_feature in req_state.mm_features:
      encoder_output = encoder_cache[mm_hash]   # (N, D+5) with extra channels
      mm_embeds_item = encoder_output[start:end]

    if is_multimodal_pruning_enabled and uses_mrope:
      mm_embeds_req, new_mrope_positions, new_delta =
          model.recompute_mrope_positions(...)   # strips extra channels, fixes positions
      req_state.mrope_positions.copy_(new_mrope_positions)

    mm_embeds.extend(mm_embeds_req)   # now (N, D) clean

  model.embed_input_ids(input_ids, multimodal_embeddings=mm_embeds)
    _merge_multimodal_embeddings  # scatters clean (N, D) into inputs_embeds
```

### "Crash site" context (line 3833)
The reported crash at line ~3833 is `_update_states`. At that point,
`_init_mrope_positions` calls `get_mrope_input_positions` which calls `_iter_mm_grid_hw`.
For images, `_iter_mm_grid_hw` yields `actual_num_tokens = llm_grid_h * llm_grid_w` (from
the grid, not the placeholder). If the placeholder was shortened by AgilePruner to `K <
full grid`, the subsequent position assignment in `_get_mrope_input_positions` builds a
position tensor with `H*W` elements, but the actual placeholder span only covers `K`
positions. The mismatch then propagates and crashes when `embed_input_ids` tries to scatter
the `K` embeddings into `H*W` placeholder positions, or when `recompute_mrope_positions`
tries to match positions to embeddings.

### `is_multimodal_pruning_enabled` flag
Set at model load time (line 4891–4894):
```python
self.is_multimodal_pruning_enabled = (
    supports_multimodal_pruning(self.get_model())
    and mm_config is not None
    and mm_config.is_multimodal_pruning_enabled()
)
```
The model must declare `supports_multimodal_pruning: ClassVar[Literal[True]] = True`
(interface in `interfaces.py` line 417) AND the vLLM config must have multimodal pruning
enabled. Only then is `recompute_mrope_positions` called.

---

## 7. Image pipeline end-to-end

### Current vanilla path (no pruning)
```
_process_image_input  -> (N, D) split per image
_postprocess_image_embeds_evs  -> if pruning_enabled: appends 5 channels -> (N, D+5)
                                   else: returns as-is -> (N, D)
embed_multimodal returns (N, D+5) or (N, D)
encoder_cache[hash] = output
_gather_mm_embeddings retrieves slices
  -> if pruning_enabled: recompute_mrope_positions strips :-5 -> (N, D)
  -> else: passes through (N, D) unchanged
embed_input_ids receives (N, D) in both cases -> scatter into inputs_embeds OK
```

### What AgilePruner-for-images needs to add
After pruning, the returned embedding has shape `(K, D)` where `K < N`. The extra-channel
approach requires:
1. Attach positions for retained tokens before caching: output `(K, D+5)`.
2. Ensure `_iter_mm_grid_hw` yields `actual_num_tokens = K` for this image (not `H*W`).
3. `recompute_mrope_positions` then reads `mm_pos[0:3, :]` from the extra channels to
   write the correct sparse positions into `mrope_positions`.
4. `_init_mrope_positions` (phase 1, dense-grid placeholder) runs BEFORE pruning but is
   **overwritten** by phase 2 — so the placeholder positions assigned at phase 1 can be
   wrong as long as they span the correct sequence range.

The core problem with the current AgilePruner patch is that it changes the placeholder
**length** (K vs N) without going through the EVS plumbing. The EVS system works only
because the placeholder length stays fixed (via `get_video_repl` which emits exactly K
`<video_token>` ids for the retained count) while the positions are corrected in phase 2.
The image path currently hardcodes `actual_num_tokens = llm_grid_h * llm_grid_w` and
never calls `recompute_mrope_positions` for images in the non-pruning path.

---

## 8. Architectural insight

The minimum-surface-area patch to enable AgilePruner-for-images via the EVS pattern:

1. **`_postprocess_image_embeds_evs`** (qwen3_vl.py): run `compute_retention_mask` on the
   image embedding (treating it as a 1-frame video), prune `emb = emb[mask]`, then attach
   the original positions of retained tokens (from `compute_mrope_for_media`) as extra 5
   channels — channel 4 (is_video) should be set to 1 so `recompute_mrope_positions` uses
   the video-embedding branch. Return `(K, D+5)`.

2. **`_iter_mm_grid_hw` image branch** (qwen3_vl.py line 2458): when pruning is enabled,
   yield `mm_feature.mm_position.length` as `actual_num_tokens` instead of
   `llm_grid_h * llm_grid_w`. This requires the placeholder to be shortened to K at
   tokenizer/processor time (or at the point `mm_position.length` is set).

3. **Processor / `mm_position.length` fix**: The placeholder range must be updated to K
   before `_init_mrope_positions` runs. This is the hardest part — vLLM sets
   `mm_position.length` from the tokenized prompt (counting image tokens), so the
   tokenizer/processing chain must emit K `<image_token>` ids instead of H*W. The video
   path solves this via `get_video_repl` which is called inside `_create_final_video_embeddings`
   and its output is used as the replacement token sequence. Images need an equivalent.

4. **`supports_multimodal_pruning` ClassVar**: declare it on the patched model class so
   `gpu_model_runner` calls `recompute_mrope_positions` for the image case too.

The EVS video pattern is reusable almost verbatim for images; the only structural
difference is that images have no timestamps and no multi-frame interleaving, making the
pattern actually simpler.

---

## 9. Open issues / questions

1. **How does `mm_position.length` get set for images?** It's set during prompt processing
   before the model sees anything. For video, the EVS-aware `get_video_repl` emits K tokens
   at encode time. For images, the analogous hook is not yet clear — needs investigation of
   `Qwen3VLMultiModalProcessor` and how it writes `PlaceholderRange` for images.
   File to check: `model_executor/models/qwen3_vl.py` around line 800–1100 (processor
   class) and `multimodal/processing.py`.

2. **Does `_init_mrope_positions` run before or after encoder execution?** Based on the
   flow above, it runs inside `_update_states` at line 3833 which precedes `_execute_mm_encoder`
   (line 3238 is in `execute_model`). This means phase-1 positions are computed from the
   **original unshortened** `mm_position.length`. If we shorten `mm_position.length` at
   pruning time (after encoder), phase 1 will have used the wrong length. This timing
   constraint may require phase-1 to use the original grid regardless and rely entirely on
   phase-2 `recompute_mrope_positions` for correctness — exactly how EVS video works.

3. **`_gather_mm_embeddings` uses `pos_info.length` to slice encoder output** (lines 2942–
   2955). If `mm_position.length` is the original N but the cached encoder output has only
   K rows (after pruning), the slice `encoder_output[start_idx:end_idx]` will read beyond
   the tensor. The encoder output shape must match `mm_position.length`, or
   `mm_position.length` must be updated to K after encoding. Video EVS avoids this because
   the encoder output (from `embed_multimodal`) already includes the indicator tokens at
   the correct total length, making encoder output length == placeholder length. The image
   EVS wrapper must do the same.

4. **`_postprocess_image_embeds_evs` currently always runs** (not gated on pruning for the
   extra-channel attachment, only `self.is_multimodal_pruning_enabled`). The 5-channel
   format is therefore only appended when the flag is set. Confirm that `embed_input_ids`
   also only calls `recompute_mrope_positions` when the flag is set — yes, confirmed at
   `gpu_model_runner.py` line 2993. No issue here but worth noting for the patch.

5. **`recompute_mrope_positions` image branch behavior** (evs.py line 248): when
   `mm_pos.shape[0] == 5` but `mm_pos[4, :] == 0` (all zeros, i.e. images in current
   upstream), `has_video_tokens = False` and positions are written via the non-adjusted
   path (lines 328–338). For a pruned image we want `mm_pos[4, :]` to be 1 for retained
   tokens so the function actually uses the embedded positions. Verify that the non-video
   branch still writes `mm_pos[0:3] + base` correctly for images — it does (line 338 is
   unconditional), but the `offset` after the media is computed differently (line 343–346):
   for 5-channel it uses `mm_pos[0:3, :].max() + base + 1` which is correct for sparse
   positions. So setting `mm_pos[4, :] = 1` for image tokens is needed.
