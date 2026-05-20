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

## MRoPE position handling (EVS piggyback)

vLLM v0.20.0 assigns 2D-MRoPE positions to visual tokens in two phases:

- **Phase 1** (`get_mrope_input_positions`, runs before vision encoder):
  generates a dense grid of positions based on `image_grid_thw`. With
  AgilePruner active, the placeholder length is K < grid product, which
  would normally crash Phase 1. Our patches (`_iter_mm_grid_hw` image
  branch + new `elif actual < expected` partial-grid branch) make Phase 1
  emit the right COUNT of positions without crashing; the position VALUES
  here are placeholders.

- **Phase 2** (`recompute_mrope_positions` in `gpu_model_runner._gather_mm_embeddings`,
  runs after vision encoder): reads the LAST 5 CHANNELS of each visual
  embedding to compute the actual MRoPE positions. We attach these channels
  in `_process_image_input` via `append_mrope_position_channels`. The
  5-channel layout is:

  | Slot | Field | Our value (pruned image) |
  |---|---|---|
  | 0 | t (frame index) | 0 |
  | 1 | h (height coord in post-merger grid) | original h_pos of kept token |
  | 2 | w (width coord) | original w_pos of kept token |
  | 3 | is_vision_start | 0 |
  | 4 | is_video (routing flag) | **1** |

  Slot 4 = 1 is a deliberate routing trick: it tells `recompute_mrope_positions`
  to use the sparse-MRoPE branch (originally written for video EVS). It does
  NOT semantically claim the content is video — the LLM still receives
  image embeddings. Upstream may rename this channel in future versions.

This piggyback is the minimum-invasion approach. An alternative would be to
add a separate `is_pruned_image` flag in `_recompute_mrope_positions`, which
requires modifying multiple upstream files; we kept the patch surface small.

When AgilePruner is DISABLED (`--agilepruner-enable=False`), the
`is_multimodal_pruning_enabled` flag is also False, Phase 2 is skipped,
and the image path produces stock `(N, D)` embeddings — fully transparent
to upstream behaviour.

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

Expected: 12 passes.

## Out of scope (phase 1)

- Per-request configuration via `extra_body`.
- Adaptive token count by erank ratio (paper §B.5).
- Video frame pruning.
- Calibrating `erank_avg` for Qwen3-VL.
- Upstream contribution to vLLM.
