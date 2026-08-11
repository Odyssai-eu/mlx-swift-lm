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

- **mimo-2.5 weight load** — alias resolves the type (no more
  `unsupportedModelType`), but the 8bit checkpoint fails to load with
  `keyNotFound model.layers.0.self_attn.attention_sink_bias`. Root: full-attn
  layers (`hybrid_layer_pattern==0`, `add_full_attention_sink_bias:false`) carry
  NO sink weight, but `MiMoV2FlashAttention` declares the `@ParameterInfo`
  unconditionally. **Two fixes attempted and REVERTED (both failed → dead horse,
  stopped per no-rustine rule):** (1) `updateMissing` `hasSinks`→`!hasSinks` — no
  effect, so `updateMissing` is NOT the throwing site; (2) sanitize-inject a
  default `ones` sink for full layers — also no effect, so the injected weight
  doesn't reach the strict verify OR sanitize isn't on this load path. **Root not
  understood**: the keyNotFound is thrown by some quantized-load/verify path that
  bypasses both hooks — needs tracing `ModelLoader`/`ModelContainer`'s weight
  application for 8bit models before touching MiMoV2Flash again. Not a clean
  alias; genuine arch difference (SWA/sink per layer).

- **kimi_linear — WRITTEN + COMPILES + COMMITTED, but a runtime-registration
  mystery blocks the load.** `Libraries/MLXLLM/Models/KimiLinear.swift` (full
  hybrid MLA[GLM4MoELite absorb]+KDA[gatedDeltaUpdate]+grouped-MoE, sanitize with
  kv_b_proj split / conv remap / expert-stack). Compiles Debug+Release. Registered
  in `LLMTypeRegistry.shared` creators (line 65, byte-identical placement to the
  working `hy_v3`/`mimo_v2` entries; the key string is present in the deployed
  binary exactly like theirs). BUT `admin/load` fails with
  `unsupportedModelType("kimi_linear")` = `ModelTypeRegistry.createModel` finds
  `creators["kimi_linear"] == nil` at runtime. RULED OUT: stale build (binary has
  the string + anti-loop; xcodebuild resolves the local-path fork), incremental
  cache (forced recompile, same result), config-decode masking (that path throws
  `configurationDecodingError`, not unsupported), Telemak-side whitelist (Telemak
  doesn't touch the registry). The entry is compiled in yet absent from the live
  dict — needs RUNTIME INTROSPECTION (dump `LLMTypeRegistry.shared.creators.keys`
  via a debug print + rebuild) to see whether the static dict literal is being
  truncated/partially-initialised. Stopped here (3 build cycles, dead horse) — a
  fresh-session debug task, not more blind rebuilds. Model IS on .29 for E2E once
  the registration resolves.

- **kimi_linear port** — component map (implemented in KimiLinear.swift; keep for
  reference / debugging):
  with the model for E2E). Every component is mapped to an existing fork template
  and the `gatedDeltaUpdate` ABI is CONFIRMED matching. Deliberately NOT written
  blind tonight: the sanitize's kv_b_proj→embed_q/unembed_out MLA-absorb split is
  numerically subtle and unvalidatable on .29 (no shards) — writing 600 lines that
  "compile but may be wrong" is the failure mode to avoid. Mechanical to implement
  with the map below:
  - `ModelArgs`: fields per Python (linear_attn_config dict → decode
    full_attn_layers/kda_layers/num_heads/head_dim/short_conv_kernel_size;
    num_experts, kv_lora_rank, qk_nope/qk_rope/v_head_dim, num_expert_group,
    topk_group, moe_router_activation_func sigmoid, moe_renormalize,
    routed_scaling_factor, first_k_dense_replace, moe_layer_freq).
  - `KimiMLAAttention` → copy `GLM4MoELiteAttention` (identical: kv_a_proj_with_mqa,
    kv_a_layernorm, embed_q/unembed_out `MultiLinear`, L==1 absorb branch).
  - `KimiDeltaAttention` → q/k/v Linear + 3× depthwise `Conv1d` (kernel 4, groups=dim,
    like Qwen3Next's conv), f_a/f_b (a_logits), b_proj (b_logits), g_a/g_b (gate),
    `A_log` param (H,1), `dt_bias` param (projDim), then
    `gatedDeltaUpdate(q,k,v, a:a_logits, b:b_logits, aLog:A_log, dtBias:dt_bias,
    state:ssm, mask:)` (ABI matches EXACTLY), then o_norm(RMSNorm head_dim) *
    sigmoid(gate), o_proj. q/k pre-scaled: q=(scale²)·rmsnorm(q), k=scale·rmsnorm(k).
  - `KimiSparseMoE` → `SwitchGLU` + `_group_expert_select` (sigmoid, e_score bias,
    n_group/topk_group) + optional shared_experts — same as g9v3/GLM4MOE grouped.
  - `KimiDecoderLayer`: `is_linear = (idx+1) ∈ kda_layers` → KDA else MLA; MoE when
    `num_experts>0 && idx≥first_k_dense_replace && idx%moe_layer_freq==0`.
  - Cache: KDA → `ArraysCache(size:4)` (q/k/v conv states + ssm); MLA → `KVCache`.
    (Both already in `KVCache.swift`.) Masks: ssm mask for KDA layers, attention
    mask for MLA layers (Qwen3NextModelInner shows the two-mask pattern).
  - `sanitize` (Python lines 492-611): block_sparse_moe→mlp rename + expert-stack
    into switch_mlp; conv1d weight moveaxis(2→1) → `{q,k,v}_conv.conv.weight`;
    dt_bias flatten; **kv_b_proj split into embed_q/unembed_out** per qk_nope/v_head
    (this is the one numerically-sensitive step — validate against a real load).

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

## kimi_linear runtime blocker — ROOT CAUSE FOUND (next-session fix)

The registration is nil at runtime because **xcodebuild compiled
`LLMModelFactory.swift` from a STALE resolution**:
`telemak/.xcbuild/SourcePackages/checkouts/mlx-swift-lm` was pinned at the
ORIGINAL fork revision **a7636b5** (no LoopGuard, no mimo alias, no kimi_linear)
even though `Package.swift` points at the local-path symlink. So KimiLinear.swift
(a new file) got picked up but the EDITED LLMModelFactory.swift (registration on
line 65) was shadowed by the stale a7636b5 copy → `creators["kimi_linear"] == nil`.

FIX (fresh session): force a fully-consistent re-resolution to the local path.
Deleting only `.xcbuild/SourcePackages/checkouts/mlx-swift-lm` + `Package.resolved`
re-resolved ALL deps and broke other versions (3 build failures) — so instead:
`swift package reset` / delete the whole `.xcbuild` and rebuild once, OR push the
fork and bump `Package.resolved` to the fork HEAD (8bbb748) so the github pin
carries the registration. Telemak also needs kimi_linear added to the explicit
LLM route in `ModelLoader.dispatchedLoad` (line ~152, next to step3p7) so it
doesn't fall through the generic VLM-capable dispatcher. Model IS on .29 for E2E.

## kimi_linear — ALL build causes exhaustively ruled out (go straight to introspection)

Do NOT repeat build cycles. Verified over ~6 rebuilds that the registration is
correct and compiled, yet `creators["kimi_linear"] == nil` at runtime while
`mimo_v2` (same dict, same file) works:
- resolution: forced local-path, `LLMModelFactory.swift` in the resolved source
  has `kimi_linear` (=1), workspace-state shows `mlx-swift-lm @ local`. Clean.
- fetch cache: irrelevant with local path (github branch cache was stale at
  a7636b5 — that was a red herring for the URL-pin path).
- compiled-object cache: purged `.xcbuild/Build/Intermediates.noindex` (full
  recompile) — SAME result.
- incremental, config-decode masking (throws configurationDecodingError not
  unsupported), VLM mis-routing (explicit LLM route via ModelLoader — same),
  Telemak whitelist (Telemak never touches the registry).
NEXT STEP = RUNTIME INTROSPECTION ONLY: add `print(LLMTypeRegistry.shared.
creators.keys.sorted())` (or dump on the unsupported throw in
`ModelTypeRegistry.createModel`), rebuild once, load kimi, read the log. That
answers whether the static dict literal actually contains kimi_linear at runtime
(suspect: a Swift issue with the large dict literal + the new generic type, or a
duplicate registry symbol). Everything else is proven fine.
