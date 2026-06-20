//
//  HYV3MTP.swift
//  LLM
//
//  Hidden-state ABI on HYV3Model for Hy3 MoE-MTP speculative decoding.
//
//  Exposes the trunk's **PRE-final-norm** hidden — what the MTP head recycles
//  (it re-norms via its own `hnorm`; the trunk's `model.norm` is only the
//  shared-head logit-time norm). Contract resolved from vLLM `hy_v3_mtp.py`.
//
//  Hy3 is GQA full-attention (no SSM/linear-attn), so cache rollback on a
//  rejected speculative block is a plain KV trim done by the iterator — no
//  per-step capture buffer is needed (unlike Qwen3.5's GatedDeltaNet path).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension HYV3ModelInner {
    /// embed + all decoder layers, returning the last-layer hidden BEFORE the
    /// final `norm` — the state the MTP head consumes. (`callAsFunction` is
    /// `norm(...)` of this; kept as its own loop to avoid touching the trunk's
    /// forward.)
    func hiddenPreNorm(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return h
    }
}

extension HYV3Model {
    /// Run the trunk; return logits (full path: `norm` → `lm_head`) and the
    /// PRE-norm hidden (for the MTP head). Covers prefill AND batched verify —
    /// Hy3 has no SSM, so one method serves both.
    public func forwardWithHidden(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> (logits: MLXArray, hidden: MLXArray) {
        let pre = model.hiddenPreNorm(inputs, cache: cache)
        let post = model.norm(pre)
        let logits = lmHead?(post) ?? model.embedTokens.asLinear(post)
        return (logits, pre)
    }

    /// Embedding lookup — shared with the MTP draft.
    public func embed(_ inputs: MLXArray) -> MLXArray {
        model.embedTokens(inputs)
    }

    /// LM head for a hidden that is ALREADY finally-normed (the MTP draft applies
    /// its own `final_layernorm` before calling this; hy_v3's shared head adds no
    /// further norm). So: `lm_head` only — NO `model.norm`.
    public func applyLMHead(_ hidden: MLXArray) -> MLXArray {
        lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    /// Fresh KV caches for the trunk (one per layer). Plain `KVCacheSimple` —
    /// full attention, no Mamba/SSM plumbing.
    public func newMTPCache() -> [KVCache] {
        (0 ..< model.layers.count).map { _ in KVCacheSimple() }
    }
}
