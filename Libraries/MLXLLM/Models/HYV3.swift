//
//  HYV3.swift
//  LLM
//
//  HunYuan-3 (Tencent) — model_type "hy_v3", arch HYV3ForCausalLM.
//  Port of the Odysseus mlx-lm reference (mlx_lm/models/hy_v3.py) to mlx-swift.
//
//  HYV3 = DeepSeek-V3-style MoE (sigmoid router + expert-correction bias +
//  shared expert + first_k_dense_replace) grafted onto a standard GQA attention
//  with qk_norm (Qwen3 style) — NO MLA. The InferencerLabs quant is already in
//  the fused mlx layout (switch_mlp / router.gate / router.expert_bias /
//  shared_mlp), so — unlike GLM4-MoE / MiniMax-M3 — NO expert-stacking sanitize
//  is needed; only the trailing MTP head is dropped.
//
//  Weight key paths (verified against the checkpoint index.json) are mirrored
//  exactly from the Python module hierarchy so the loader binds 1:1:
//    model.layers.N.self_attn.{q,k,v,o}_proj, .{q,k}_norm
//    model.layers.0.mlp.{gate,up,down}_proj                  (dense, first_k)
//    model.layers.N.mlp.router.gate, .router.expert_bias     (MoE, N>=1)
//    model.layers.N.mlp.switch_mlp.{gate,up,down}_proj        (fused experts)
//    model.layers.N.mlp.shared_mlp.{gate,up,down}_proj
//    model.{embed_tokens,norm}, lm_head
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Routing (DeepSeek-V3, no expert groups)

/// sigmoid → select on biased scores → weight with the UNbiased scores →
/// (optional) renorm top-k → × routed_scaling_factor. Mirrors `expert_select`
/// in hy_v3.py: the bias only steers selection, never the combine weight.
private func hyv3ExpertSelect(
    gates: MLXArray, expertBias: MLXArray, topK: Int,
    routedScalingFactor: Float, normTopkProb: Bool
) -> (MLXArray, MLXArray) {
    let scores = sigmoid(gates.asType(.float32))
    let originalScores = scores
    let selectionScores = scores + expertBias.asType(.float32)

    let inds = argPartition(-selectionScores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    var weights = takeAlong(originalScores, inds, axis: -1)
    if topK > 1, normTopkProb {
        weights = weights / weights.sum(axis: -1, keepDims: true)
    }
    weights = weights * routedScalingFactor
    return (inds, weights)
}

// MARK: - Attention (GQA + qk_norm, no MLA)

class HYV3Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: HYV3Configuration) {
        self.nHeads = args.numAttentionHeads
        self.nKVHeads = args.numKeyValueHeads
        self.headDim = args.headDim
        self.scale = pow(Float(args.headDim), -0.5)

        _qProj.wrappedValue = Linear(args.hiddenSize, nHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(args.hiddenSize, nKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(args.hiddenSize, nKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(nHeads * headDim, args.hiddenSize, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        self.rope = initializeRope(
            dims: headDim,
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var q = qProj(x).reshaped(B, L, nHeads, -1)
        var k = kProj(x).reshaped(B, L, nKVHeads, -1)
        var v = vProj(x).reshaped(B, L, nKVHeads, -1)

        // qk_norm: per-head RMSNorm over head_dim, applied in [B,L,H,D] layout.
        q = qNorm(q).transposed(0, 2, 1, 3)
        k = kNorm(k).transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        q = applyRotaryPosition(rope, to: q, offset: offset)
        k = applyRotaryPosition(rope, to: k, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: q,
            keys: k,
            values: v,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - Dense MLP (SwiGLU) — first_k layer and shared expert

class HYV3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Router (quantized gate + f32 correction bias)

class HYV3Router: Module {
    let topK: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray

    init(_ args: HYV3Configuration) {
        self.topK = args.numExpertsPerTok
        self.normTopkProb = args.routeNorm
        self.routedScalingFactor = args.routerScalingFactor

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _expertBias.wrappedValue = MLXArray.zeros([args.numExperts])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        hyv3ExpertSelect(
            gates: gate(x),
            expertBias: expertBias,
            topK: topK,
            routedScalingFactor: routedScalingFactor,
            normTopkProb: normTopkProb
        )
    }
}

// One-shot sub-op profiling of the MoE block on the first multi-token call.
nonisolated(unsafe) private var _hyv3MoEProfiled = false

// MARK: - MoE block

class HYV3MoE: Module, UnaryLayer {
    @ModuleInfo(key: "router") var router: HYV3Router
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_mlp") var sharedMLP: HYV3MLP

    init(_ args: HYV3Configuration) {
        _router.wrappedValue = HYV3Router(args)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )
        let sharedHidden = args.moeIntermediateSize * args.numSharedExperts
        _sharedMLP.wrappedValue = HYV3MLP(hiddenSize: args.hiddenSize, intermediateSize: sharedHidden)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // PROFILE (one-shot): first multi-token MoE call — time each sub-op.
        if !_hyv3MoEProfiled, x.dim(1) >= 2, x.dim(1) <= 4 {
            _hyv3MoEProfiled = true
            func ms(_ t: Date) -> String { String(format: "%.2f", Date().timeIntervalSince(t) * 1000) }
            var rows = ["op,ms", "seqLen,\(x.dim(1))"]
            var t = Date()
            let (inds, weights) = router(x); eval(inds, weights)
            rows.append("router,\(ms(t))"); t = Date()
            var y = switchMLP(x, inds); eval(y)
            rows.append("switchMLP,\(ms(t))"); t = Date()
            y = (y * weights[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype); eval(y)
            rows.append("combine,\(ms(t))"); t = Date()
            let sh = sharedMLP(x); eval(sh)
            rows.append("sharedMLP,\(ms(t))")
            try? rows.joined(separator: "\n")
                .write(toFile: "/tmp/hy3-moe-profile.csv", atomically: true, encoding: .utf8)
            return y + sh
        }
        let (inds, weights) = router(x)
        var y = switchMLP(x, inds)
        // weights are f32 (carry routed_scaling_factor); promote, sum, cast back.
        y = (y * weights[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        return y + sharedMLP(x)
    }
}

// MARK: - Decoder layer

class HYV3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: HYV3Attention
    let mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: HYV3Configuration, layerIdx: Int) {
        _selfAttn.wrappedValue = HYV3Attention(args)

        if layerIdx >= args.firstKDenseReplace {
            self.mlp = HYV3MoE(args)
        } else {
            self.mlp = HYV3MLP(hiddenSize: args.hiddenSize, intermediateSize: args.intermediateSize)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Inner model

public class HYV3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [HYV3DecoderLayer]
    let norm: RMSNorm

    init(_ args: HYV3Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { idx in
            HYV3DecoderLayer(args, layerIdx: idx)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Causal LM

public class HYV3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: HYV3ModelInner
    let configuration: HYV3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: HYV3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.numKeyValueHeads, count: args.hiddenLayers)
        self.model = HYV3ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }

        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // The InferencerLabs quant is already in the fused mlx layout
        // (switch_mlp / router.gate / router.expert_bias / shared_mlp), so no
        // expert stacking is required. Drop only the trailing MTP head
        // (num_nextn_predict_layers → an extra `model.layers.<hiddenLayers>`
        // block + any .mtp/nextn keys) and rotary inv_freq buffers.
        let mtpLayerPrefix = "model.layers.\(configuration.hiddenLayers)"
        var sanitized = weights.filter { (key, _) in
            !key.hasPrefix(mtpLayerPrefix)
                && !key.contains("rotary_emb.inv_freq")
                && !key.contains(".mtp")
                && !key.contains("nextn")
                && !key.hasPrefix("model.mtp")
        }

        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        return sanitized
    }
}

// MARK: - Configuration

public struct HYV3Configuration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var numExperts: Int
    var numExpertsPerTok: Int
    var numSharedExperts: Int
    var firstKDenseReplace: Int
    var routerScalingFactor: Float
    var routeNorm: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case firstKDenseReplace = "first_k_dense_replace"
        case routerScalingFactor = "router_scaling_factor"
        case routeNorm = "route_norm"
    }

    // rope_theta lives in a nested `rope_parameters` block (transformers 5.6),
    // not the legacy top-level key. Kept out of the main CodingKeys (it maps to
    // no stored property) so Encodable still auto-synthesizes; read via a
    // separate container below.
    private enum RopeOuterKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
    }
    private enum RopeParamsKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "hy_v3"
        self.vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        self.moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        self.hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        self.numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        self.numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        self.headDim = try c.decode(Int.self, forKey: .headDim)
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144

        // Prefer rope_parameters.rope_theta, fall back to top-level, then default.
        var theta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        let ropeOuter = try decoder.container(keyedBy: RopeOuterKeys.self)
        if let rp = try? ropeOuter.nestedContainer(
            keyedBy: RopeParamsKeys.self, forKey: .ropeParameters)
        {
            theta = (try? rp.decodeIfPresent(Float.self, forKey: .ropeTheta)) ?? theta
        }
        self.ropeTheta = theta ?? 11_158_840.0

        self.ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.numExperts = try c.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        self.numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 1
        self.firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
        self.routerScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routerScalingFactor) ?? 1.0
        self.routeNorm = try c.decodeIfPresent(Bool.self, forKey: .routeNorm) ?? true
    }
}

// MARK: - LoRA

extension HYV3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
