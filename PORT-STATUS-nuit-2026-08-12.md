# Telemak↔Python parity — night 2026-08-11/12

Autonomous goal: align Telemak (Swift fork) with the Python engine — support
minimax, hy3, kimi-linear, deepseek-flash, mimo-2.5, ornith-1, inkling-small;
feature parity; fix the Qwen3.6 thinking-loop.

## Gap analysis (model coverage)

| Target | model_type | Telemak status |
|---|---|---|
| minimax (M2/M3/VL) | minimax / minimax_m3 / minimax_m3_vl | **already supported** (LLM + VLM factories) |
| hy3 | hy_v3 | **already supported** (HYV3.swift) |
| Ornith-1 (397B) | **qwen3_5_moe** | **already supported** — it's a Qwen3.5-MoE finetune |
| mimo-2.5 | mimo_v2 | alias added; **weight fix pending** (see below) |
| deepseek-flash | deepseek_v4 | **missing** — Telemak has v3 only. DEFER (epic) |
| kimi-linear | kimi_linear | **missing** — port scoped below |
| inkling-small | inkling_mm_model | **missing** — DEFER (multimodal, native package) |

## DONE + validated tonight

1. **LoopGuard anti-loop** (`Libraries/MLXLMCommon/LoopGuard.swift` + wired into
   `TokenIterator.next()` in `Evaluate.swift`). Token-id degenerate-loop detector,
   parity with the Python `_detect_loop_large`. Anchor-trick: short period (≤64)
   ×12, large period (65..1024) ×4 → ends generation like maxTokens. Default on,
   `TELEMAK_ANTILOOP=0` disables.
   - **Validated**: 4/4 deterministic unit tests (swiftc against the real source):
     large-block runaway trips, normal varied does not, tight loop trips,
     sub-threshold 3× repeat does not. Compiles into Telemak (Debug+Release green).
   - This is the Qwen3.6 thinking-loop fix (the bench saw 1878c ×5 → this cuts it).
   - Commit `53d2227` on `feat/model-gap`.

2. **mimo_v2 alias → mimo_v2_flash** (`LLMModelFactory.swift`). Resolves the
   model_type (was `unsupportedModelType("mimo_v2")`). Commit `53d2227`.

## Blocked / scoped follow-ups

- **mimo-2.5 weight fix**: MiMo-V2.5 genuinely differs from the flash variant.
  Config has `add_swa_attention_sink_bias:True`, `add_full_attention_sink_bias:False`
  → SWA layers carry `attention_sink_bias`, full-attn layers do NOT. Layer 0 of the
  8bit checkpoint has NO sink bias, but `MiMoV2FlashAttention` sets `hasSinks` such
  that it requires the weight → `keyNotFound`. Fix = make the per-layer sink
  determination match the checkpoint's SWA/full pattern (or tolerate an absent
  `attention_sink_bias` by keeping the default). Not a clean alias — needs the
  MiMoV2Flash model generalised.

- **kimi_linear port** (composable, low-risk — all blocks exist in the fork):
  - Reference: stock `mlx_lm/models/kimi_linear.py` (611 lines) — pulled to
    scratchpad. Hybrid arch, 27 layers:
    - full-attn layers → **MLA** (kv_lora_rank=512, q_lora, qk_nope/qk_rope,
      v_head_dim, embed_q/unembed_out `MultiLinear` absorb). **Template =
      `GLM4MoELiteAttention` in `GLM4MOELite.swift`** — near-identical MLA-absorb.
    - kda layers → **KimiDeltaAttention** = `ShortConv1d` (depthwise conv1d,
      kernel 4) on q/k/v + `gated_delta_update`. **Template =
      `Qwen3NextGatedDeltaNet`** + the fork's `GatedDelta.swift` (`gatedDeltaUpdate`).
    - MoE → grouped-topk sigmoid router + shared experts + `e_score_correction_bias`
      (identical to `GLM4MOE`/`DeepseekV3` grouped select).
    - Layer routing by explicit `linear_attn_config.kda_layers` /
      `full_attn_layers` lists (NOT a modular interval like Qwen3Next).
  - **Not attempted tonight**: cannot E2E-validate on .29 (the kimi-linear shards
    are not on .29's disk — same as Qwen3.6). A ~500-line hybrid MLA+KDA port
    compiled-but-unvalidated risks confident-but-wrong numerics; do it on a node
    that HAS the model so load + coherence can be checked.

- **deepseek_v4 (flash)**: DEFER. Novel arch (Indexer, Compressor,
  HyperConnection, PoolingCache; 1787 lines Python) and "never elucidated beyond
  0.32 tok/s" even in Python. Not one-night, low value.

- **inkling_mm_model (small)**: DEFER. Multimodal; Python serves it via a native
  `inkling_mlx` package (not mlx_lm) — no clean Swift port path.

## Dev-loop gotchas discovered (important)

- **xcodebuild ignores `swift package edit`.** The edit override only affects the
  SwiftPM CLI (`.build`); xcodebuild resolves from `.xcbuild/SourcePackages`
  (the GitHub pin). Symptom: builds "succeed" but contain NONE of your fork
  changes. **Fix**: point `telemak/Package.swift` at a local **path** dependency.
  The identity must stay `mlx-swift-lm`, so the path basename must be
  `mlx-swift-lm` → created symlink `~/Claude/code/mlx-swift-lm ->
  mlx-swift-lm-odyssai` and set `.package(path: ".../mlx-swift-lm")`.
  **This Package.swift edit is DEV-ONLY — revert to the github url/branch pin
  before committing telemak, and push the fork branch so the pin resolves it.**

- **CODESIGNING "Invalid Page" crash** when hand-copying only the binary: the
  Developer-ID-signed `telemak` and the `mlx-swift_Cmlx.bundle` (metallib) are a
  matched signed set. Copy BOTH (use `deploy-all.sh`, which does). Copying only
  the rebuilt binary over an old bundle → the binary mmaps a mismatched bundle →
  kernel kills it (empty log).

- **`deploy-all.sh --canary ultra-512` launchd bootstrap fails** ("Bootstrap
  failed: 5") because .29's deepseek slot plists are `.off`. The binary+bundle DO
  get placed correctly before that step, so: run deploy-all (ignore the launchd
  error), then launch manually: `.29` config = `Release.deepseek`, **port 8013**,
  `TELEMAK_MODELS_DIR=/Volumes/models/odysseus`.

- **Model resolution prefers the HF cache over the odysseus dir.** A valid HF-repo
  id (`inferencerlabs/...`) resolves to `~/.cache/huggingface/hub/models--...`
  even when that snapshot is incomplete. `.29` is MISSING full shards for
  Qwen3.6-35B and kimi-linear (config present, weights absent) — a real blocker
  for E2E testing there. `odyssai/...` ids (not on HF) fall through to the
  odysseus dir.
