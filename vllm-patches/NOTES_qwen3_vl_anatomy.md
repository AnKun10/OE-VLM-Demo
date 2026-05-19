# qwen3_vl.py — vLLM v0.20.0 anatomy

Pin: v0.20.0
Upstream commit: 88d34c6409e9fb3c7b8ca0c04756f061d2099eb1
File line count: 2880

## Classes

| Class | Line | Role |
|---|---|---|
| `Qwen3_VisionPatchEmbed` | 346 | 2D patch embedding for raw pixel values → patch tokens |
| `Qwen3_VisionMLP` | 375 | MLP block inside each vision transformer layer |
| `Qwen3_VisionBlock` | 412 | Single vision transformer layer (attn + MLP). Uses `Qwen2_5_VisionAttention` (imported from qwen2_5_vl.py) |
| `Qwen3_VisionPatchMerger` | 466 | Spatial downsampler: groups `spatial_merge_size²` adjacent patch tokens → one LLM token via two FC layers |
| `Qwen3_VisionTransformer` | 518 | Vision encoder / ViT backbone. Iterates `self.blocks`, applies deepstack mergers, then final `self.merger` |
| `Qwen3VLProcessingInfo` | 854 | HF processor wrappers (image, video processors, data parser) |
| `Qwen3VLDummyInputsBuilder` | 1047 | Builds synthetic inputs for memory profiling / cuda graph warmup |
| `Qwen3VLMultiModalProcessor` | 1195 | Converts raw images/videos → token-id placeholder sequences (Stage 1). Computes `num_tokens` per image and builds `PromptReplacement` lists |
| `Qwen3LLMModel` | 1501 | Thin subclass of `Qwen3Model` with EVS-aware forward |
| `Qwen3LLMForCausalLM` | 1554 | Thin wrapper for LM head over `Qwen3LLMModel` |
| `Qwen3VLForConditionalGeneration` | 1592 | Top-level model. Owns `self.visual` (Qwen3_VisionTransformer) + `self.language_model`. Implements `embed_multimodal`, `embed_input_ids`, `forward`, MRoPE interface, encoder cuda graph interface |

## Stage 1: input processor (token-count)

The image token count is computed in **two places**:

### 1a. `Qwen3VLMultiModalProcessor._get_prompt_updates` (line 1343)

- **Inner function**: `get_image_replacement_qwen3vl` (defined at line 1360, closure inside `_get_prompt_updates`)
- Signature: `def get_image_replacement_qwen3vl(item_idx: int) -> list[int]`
- Called by: `PromptReplacement` callback during prompt tokenization
- Inputs: `item_idx` → looks up `out_mm_kwargs["image"][item_idx]["image_grid_thw"]` (a `(3,)` tensor of `[t, h, w]`)
- Returns: list of `image_token_id` repeated `num_tokens` times
- Implementation (lines 1361–1366):
  ```python
  out_item = out_mm_kwargs["image"][item_idx]
  grid_thw = out_item["image_grid_thw"].data
  assert isinstance(grid_thw, torch.Tensor)

  num_tokens = int(grid_thw.prod()) // merge_length
  return [hf_processor.image_token_id] * num_tokens
  ```
  where `merge_length = image_processor.merge_size**2` (line 1358).
  So: `num_tokens = t * h * w // spatial_merge_size²`

### 1b. `Qwen3VLForConditionalGeneration.get_encoder_cudagraph_per_item_output_tokens` (line 1858)

- Signature: `def get_encoder_cudagraph_per_item_output_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]`
- Returns: `[t * (h // m) * (w // m) for t, h, w in grid_thw]`
  where `m = self.visual.spatial_merge_size`
- This is identical math: `t * h/m * w/m = t*h*w / m²`
- Called by CudaGraph engine for encoder scheduling

**Phase 3.2 patch target**: the closure `get_image_replacement_qwen3vl` at line 1365 (the `num_tokens =` expression). This is the single place that controls how many `<|image_pad|>` tokens appear in the prompt string.

## Stage 2: vision tower

- **Tower class**: `Qwen3_VisionTransformer` (line 518)
- **Constructor**: lines 519–614
- **Forward**: line 783, signature:
  ```python
  def forward(self, x: torch.Tensor, grid_thw: torch.Tensor | list[list[int]], *, encoder_metadata: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
  ```
- **Block iteration** (line 805):
  ```python
  for layer_num, blk in enumerate(self.blocks):
      hidden_states = blk(hidden_states, cu_seqlens=..., rotary_pos_emb_cos=..., ...)
  ```
- **Vision block class**: `Qwen3_VisionBlock` (line 412), forward at line 444
- **Attention class inside block**: `Qwen2_5_VisionAttention` (imported from `qwen2_5_vl.py`, line 309 there)
  - Uses `MMEncoderAttention` internally (imported from `vllm.model_executor.layers.attention`)
  - `MMEncoderAttention` wraps FlashAttention — there is **no explicit `attn_weights = Q @ K^T` materialisation**. The kernel fuses QK^T, softmax, and V multiplication. Attention weights are NOT accessible as intermediate tensors.
  - `Qwen2_5_VisionAttention.forward` calls `self.attn(query=q, key=k, value=v, cu_seqlens=..., max_seqlen=...)` at line 405 (qwen2_5_vl.py)
- **`fullatt_block_indexes`**: NOT present in Qwen3-VL. Qwen3 uses **uniform full attention** for all blocks — no window-attention switching. (The `fullatt_block_indexes` pattern is from Qwen2-VL's hybrid window/full attention.) The `Qwen3VLVisionConfig` does not carry this attribute; the config is read from `transformers.models.qwen3_vl` via `Qwen3VLConfig` and `Qwen3VLVisionConfig`.
- **deepstack**: if `vision_config.deepstack_visual_indexes` is set, intermediate block outputs are saved at those layer indices and passed through additional `Qwen3_VisionPatchMerger`s (line 814–819). Standard Qwen3-VL-8B-Instruct has an empty list.
- Final output: `self.merger(hidden_states)` (line 820), then cat with deepstack features if any. Returns shape `[total_patches_after_merge, out_hidden_size]`.

## get_multimodal_embeddings (or equivalent)

vLLM v0.20.0 splits this into two methods on `Qwen3VLForConditionalGeneration`:

### `embed_multimodal` (line 2670)
- Signature: `def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None`
- Parses and validates inputs (line 2671), returns `None` if no multimodal data
- Iterates over modalities, calling `_process_image_input` or `_process_video_input`
- Applies EVS postprocessing (`_postprocess_image_embeds_evs` / `_postprocess_video_embeds_evs`)
- Multi-image handling: each image's embeddings are kept as **separate tensors** in a list (`multimodal_embeddings.extend(image_embeddings)`). The final return is a `tuple` of per-image (or per-video) tensors.
- Returns: `tuple[torch.Tensor, ...]` where each tensor is shape `[num_tokens_for_item, hidden_dim]`

### `_process_image_input` (line 2088)
- Signature: `def _process_image_input(self, image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]`
- Vision tower invocation line: **2103** — `image_embeds = self.visual(pixel_values, grid_thw=grid_thw)`
  - (or line 2100 via `run_dp_sharded_mrope_vision_model` for data-parallel path)
- Splits result into per-image tensors at line 2107–2108:
  ```python
  sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
  return image_embeds.split(sizes)
  ```
- **Returns** per-image split tuple (line 2108)

## get_input_embeddings / merge

- **Method**: `Qwen3VLForConditionalGeneration.embed_input_ids` (line 2741)
- Signature: `def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None) -> torch.Tensor`
- **Merge mechanism**: calls `_merge_multimodal_embeddings` (imported from `vllm.model_executor.models.utils`, line 135) at line 2771:
  ```python
  inputs_embeds = _merge_multimodal_embeddings(
      inputs_embeds=inputs_embeds,
      multimodal_embeddings=multimodal_embeddings,
      is_multimodal=is_multimodal,
  )
  ```
  `is_multimodal` is a boolean mask over the input_ids sequence marking placeholder positions. The merge **scatters** each visual tensor into the placeholder positions in order. The number of placeholder tokens for image `i` must exactly equal `len(multimodal_embeddings[i])`.

## Position encoding

- **Where computed**: in the **LLM decoder** (not the vision tower), via MRoPE.
- `Qwen3VLForConditionalGeneration._get_mrope_input_positions` (static method, line 2510) builds a 3D position tensor `(3, seq_len)` covering text tokens and visual tokens. Visual tokens get 2D grid indices (h, w axes) while the temporal axis uses frame index.
- The vision transformer itself uses its own internal 2D rotary embedding (`self.rotary_pos_emb`, line 565) for patch-level attention — but this is internal to the ViT and discarded after encoding.
- **Pruning implication**: If visual tokens in `embed_multimodal` are pruned (subset selected), `_get_mrope_input_positions` is called with the **actual placeholder token sequence** (already shortened at Stage 1). Provided Stage 1 and Stage 2 agree on the same pruned token count, MRoPE will assign correct positions to the pruned tokens. However, `_get_mrope_input_positions` uses `actual_num_tokens` from `_iter_mm_grid_hw` which reads the actual placeholder length — so it adapts automatically if the placeholder is shorter. **No separate re-indexing is needed** as long as Stage 1 token count = Stage 2 embedding count.

## Stage 1 patching plan (Phase 3.2)

- **Insertion point**: closure `get_image_replacement_qwen3vl` inside `Qwen3VLMultiModalProcessor._get_prompt_updates`, line 1365
- The expression `num_tokens = int(grid_thw.prod()) // merge_length` controls placeholder length.
- Patch shape: after computing `num_tokens`, scale it by `(1 - pruning_rate)` rounded to nearest integer, then floor to at least 1:
  ```python
  # --- AgilePruner patch: compressed placeholder ---
  pruning_rate = _get_agilepruner_rate()   # reads VLLM_AGILEPRUNER_RATE env var
  if pruning_rate > 0.0:
      num_tokens = max(1, int(round(num_tokens * (1.0 - pruning_rate))))
  # --- end patch ---
  return [hf_processor.image_token_id] * num_tokens
  ```
  Helper `_get_agilepruner_rate()` can be a module-level cached function reading `os.environ.get("VLLM_AGILEPRUNER_RATE", "0")`.

## Stage 4.1 patching plan (attention capture)

**Problem**: `Qwen2_5_VisionAttention` uses `MMEncoderAttention` (FlashAttention kernel) — no intermediate attention weight tensor is materialised. We cannot extract per-head attention weights from it directly.

**Approach options**:
1. **Surrogate scores only**: Skip attention capture entirely. Use `compute_surrogate_cls_score` (surrogate via patch norms / activation magnitudes) as the selection score in Phase 4.2. This avoids touching the attention kernel.
2. **Hook on block outputs**: Register a forward hook on each `Qwen3_VisionBlock` to record the hidden states after each block; derive importance from hidden state norms (proxy for attention salience).
3. **Replace MMEncoderAttention with a softmax implementation**: For the final few blocks (e.g., last 4), replace the fused kernel with a manual QK^T→softmax→V to materialise weights. High complexity.

**Recommended**: Option 1 (surrogate scores). No patch to the attention class is needed.

- If Option 1 adopted: no changes needed in the vision block or attention class.
- If Option 2 desired: add `forward_hook` registered in `Qwen3_VisionTransformer.__init__` after building `self.blocks`; store last block's hidden states in `self._last_hidden_states`.

## Stage 4.2 patching plan (apply selection)

- **Insertion point**: `Qwen3VLForConditionalGeneration._process_image_input` (line 2088), immediately after line 2108 (`return image_embeds.split(sizes)`) — actually replace the return with pruned split.
- More precisely: insert after `image_embeds.split(sizes)` (line 2107), before the `return` (line 2108):
  ```python
  # --- AgilePruner patch: select top-k visual tokens per image ---
  from vllm.model_executor.models.agilepruner import agilepruner_select, compute_erank
  pruning_rate = _get_agilepruner_rate()
  if pruning_rate > 0.0:
      pruned = []
      for emb in image_embeds.split(sizes):
          scores = compute_surrogate_cls_score(emb)  # [num_tokens]
          selected = agilepruner_select(emb, scores, pruning_rate)  # [k, hidden]
          pruned.append(selected)
      return tuple(pruned)
  return image_embeds.split(sizes)
  # --- end patch ---
  ```
- Imports needed: `from vllm.model_executor.models.agilepruner import agilepruner_select, compute_surrogate_cls_score`
- `_get_agilepruner_rate` should be defined at module top level (cached env var read).

## Open questions / risks

1. **Stage 1 / Stage 2 count mismatch**: The `_merge_multimodal_embeddings` scatter will silently produce wrong results (or crash) if `len(multimodal_embeddings[i])` ≠ number of placeholder tokens for image `i`. Both patches MUST use the same `pruning_rate` and the same rounding formula. A single `_get_agilepruner_rate()` function shared between Stage 1 and Stage 2 code paths ensures this, but the rounding `max(1, round(n * (1-r)))` must be identical in both.

2. **Data-parallel path**: When `self.use_data_parallel` is True (line 2098), `_process_image_input` returns early via `run_dp_sharded_mrope_vision_model` (line 2099–2101) without going through `image_embeds.split`. The Phase 4.2 patch must also intercept the DP path, or assert that DP is disabled when pruning is active.

3. **Flash attention — no weights**: `Qwen2_5_VisionAttention` uses fused FlashAttention via `MMEncoderAttention`. There is no materialised attention weight tensor available for Phase 4.1 attention-based scoring. Score computation must use surrogate methods (patch norms, MLP activations, or deepstack intermediate features).

4. **EVS / video pruning interaction**: `_postprocess_video_embeds_evs` (line 2176) appends 5-channel MRoPE position data to video embeddings. If AgilePruner also prunes video tokens, care must be taken to also subset the appended position channels. For images, `_postprocess_image_embeds_evs` only activates when `is_multimodal_pruning_enabled` — that flag should be checked to avoid double-pruning.

5. **CudaGraph buffer shapes**: `get_encoder_cudagraph_per_item_output_tokens` (line 1858) is used to pre-allocate cuda graph replay buffers. After Phase 3.2 reduces placeholder count, `get_encoder_cudagraph_per_item_output_tokens` must return the pruned count too, otherwise buffer shapes will be wrong. A corresponding patch there may be needed.

6. **`fullatt_block_indexes` is absent**: Unlike Qwen2-VL, Qwen3-VL uses full attention for all blocks. No block-conditional logic to patch around.
