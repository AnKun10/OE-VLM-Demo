# vllm-patches/

AgilePruner visual-token pre-pruning files for vLLM, applied as a runtime
file-overlay onto the pre-installed vLLM in the Vast.ai pod image.

## Pin

This patch-set is pinned to **vLLM v0.20.0** (the version baked into
`vastai/vllm:v0.20.0-cuda-13.0`). See `PIN.txt`. Applying these files to a
different vLLM version is unsupported and may break vLLM at import time.

## Layout

```
vllm-patches/
├── PIN.txt                                    pinned vLLM version
├── README.md                                  this file
├── vllm/
│   ├── engine/arg_utils.py                    patched: + 4 CLI flags
│   └── model_executor/models/
│       ├── qwen3_vl.py                        patched: stages 1+2 pruning
│       └── agilepruner.py                     new: pure-math helpers
└── tests/models/test_agilepruner.py           12 CPU unit tests
```

The vast template `vast-templates/oe-vlm-demo-agilepruner/onstart.sh`
overlays these files into the pre-installed vLLM site-packages at boot.
Originals are saved as `*.orig` (one-time, idempotent).

## CLI flags

The patched `arg_utils.py` adds:

| Flag | Default | Description |
|---|---|---|
| `--agilepruner-enable` | `False` | Master switch. Off → identical to upstream. |
| `--agilepruner-ratio` | `0.5` | K = round(ratio × N) per image. Floor 4. |
| `--agilepruner-tau-max` | `0.25` | Cap on similarity threshold τ. |
| `--agilepruner-erank-avg` | `95.0` | Dataset-mean erank for normalising τ. |

## Algorithm

Adapted from AgilePruner (Baek et al., ICLR 2026,
https://arxiv.org/abs/2603.01236). Three pieces:

- **Surrogate CLS:** column-mean of attention (excluding diagonal) at the
  last full-attention layer of the Qwen3-VL vision tower.
- **erank:** post-projector embedding effective rank via N×N covariance.
- **Adaptive iterative threshold:** greedily select highest-score alive
  token; prune cosine-similar neighbours. τ_i = min(order_i × tau_base,
  tau_max) where tau_base = (erank_input / erank_avg) × 0.01.

## MRoPE position handling

vLLM v0.20.0 assigns 2D-MRoPE positions to visual tokens in `_iter_mm_grid_hw` →
`_get_mrope_input_positions`. The upstream image branch hardcodes
`actual_num_tokens = grid_h * grid_w`, assuming `mm_position.length` equals
the grid product. AgilePruner violates this by compressing the placeholder
count to K < grid product.

Our patches fix the resulting crash:

- `_iter_mm_grid_hw` IMAGE branch now yields `mm_feature.mm_position.length`
  (mirroring the video branch's existing behaviour).
- `_get_mrope_input_positions` gains an `elif actual_num_tokens <
  expected_tokens_per_frame:` branch that emits exactly K position columns
  (rather than the full grid).

**Known limitation (Bug #2):** the K position columns emitted here are the
FIRST-K row-major grid positions, NOT the actual (h, w) coordinates of the
retained tokens. The K retained embeddings are in attention-rank order so
the position-to-content mapping is scrambled. This is acceptable for
descriptive prompts (e.g., "what colors are in this image?") but may
degrade accuracy on spatial-reasoning prompts (e.g., "where is the red
shape?").

We considered the "EVS piggyback" path — set `is_multimodal_pruning_enabled
=True` to route through vLLM's `recompute_mrope_positions` (the same code
path video EVS uses for sparse-MRoPE position recovery). This worked
architecturally but activated a dormant upstream bug in
`_compute_deepstack_embeds`: the +5 channels appended to image embeddings
break the `torch.split([visual_dim, multiscale_dim])` operation. See the
git history for the reverted Phase 5 + Phase 2-redo commits and our
investigation notes in `NOTES_evs_video_pipeline.md`.

When AgilePruner is DISABLED (`--agilepruner-enable=False`), all the
patches above are no-ops and behaviour is identical to upstream.

## Calibration caveat

`--agilepruner-erank-avg 95.0` is the LLaVA training-set value from the
paper (Appendix D). It is **not calibrated for Qwen3-VL**. The pruning
still works correctly; only the adaptive-threshold scale is paper-default.
A follow-up may calibrate on a Qwen3-VL distribution.

## Local tests

The tests in `tests/models/test_agilepruner.py` run on CPU (no GPU, no
vLLM install required):

```bash
cd <repo-root>/vllm-patches
PYTHONPATH=. pytest tests/models/test_agilepruner.py -v
```

Expected: 17 passes.

## Out of scope (phase 1)

- Per-request configuration via `extra_body`.
- Adaptive token count by erank ratio (paper §B.5).
- Video frame pruning.
- Calibrating `erank_avg` for Qwen3-VL.
- Upstream contribution to vLLM.
