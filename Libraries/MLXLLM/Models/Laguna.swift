//
//  Laguna.swift
//  LLM
//
//  Port of poolside/Laguna-S-2.1 / Laguna-XS-2.1 (sparse MoE) for OdyssAI-X.
//
//  References (verified 2026-07-25):
//  - PipeNetwork laguna.py (Apache-2.0), vendored in the OdyssAI-X repo at
//    scripts/mlx_models/laguna.py — the mlx-lm implementation the
//    pipenetwork/Laguna-S-2.1-MLX-* quants were produced with.
//  - Blaizzy mlx-vlm 0.6.3 models/laguna/language.py — the implementation the
//    mlx-community/AtomicChat Laguna-XS-2.1 quants were produced with.
//  Both agree on the math; they differ only in checkpoint key layout, which
//  sanitize() below reconciles (see the table there).
//
//  Architecture = Qwen3-MoE plus:
//  * softplus attention output gating (per-head `g_proj`)
//  * per-head Q/K RMSNorm before RoPE
//  * interleaved full / sliding-window (512) attention, one mask per type
//  * query-head count varies BY LAYER (e.g. S: 48 full / 72 sliding;
//    XS: 48 / 64); KV heads stay constant
//  * two RoPEs: full-attention layers use partial-rotary (0.5) YaRN;
//    sliding layers plain RoPE (theta 1e4, full rotary)
//  * sigmoid router + aux-loss-free e_score_correction_bias (DeepSeek-V3
//    style), routed_scaling_factor, always-on shared expert
//  * layer 0 is a dense MLP (mlp_only_layers=[0])
//
//  YaRN note: the configs carry `attention_factor`, but BOTH Python
//  references drop it and let YarnRoPE derive mscale from `factor`
//  (0.1·ln(factor)+1). We match that behaviour bit-for-bit by using stock
//  initializeRope — do NOT "fix" this to honor attention_factor (XS declares
//  1.0 there; honoring it would diverge from every existing Laguna runtime).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct LagunaConfiguration: Codable, Sendable {
    var modelType: String = "laguna"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var vocabularySize: Int
    var rmsNormEps: Float
    // MoE
    var numExperts: Int
    var numExpertsPerToken: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var decoderSparseStep: Int
    var normTopkProb: Bool
    var moeRoutedScalingFactor: Float
    var moeRouterLogitSoftcapping: Float?
    var mlpOnlyLayers: [Int]
    // attention
    var attentionHeadsPerLayer: [Int]?
    var gating: String
    var slidingWindow: Int?
    var layerTypes: [String]?
    var ropeParameters: [String: [String: StringOrNumber]]?
    var maxPositionEmbeddings: Int
    var tieWordEmbeddings: Bool

    var resolvedLayerTypes: [String] {
        layerTypes ?? Array(repeating: "full_attention", count: hiddenLayers)
    }

    func headsForLayer(_ layerIdx: Int) -> Int {
        attentionHeadsPerLayer?[layerIdx] ?? attentionHeads
    }

    /// Rope sub-config for one attention type, mirroring laguna.py's lookup:
    /// sliding layers read `rope_parameters.sliding_attention`, full layers
    /// `rope_parameters.full_attention` (falling back to the whole dict when
    /// it is not nested).
    func ropeConfig(sliding: Bool) -> [String: StringOrNumber] {
        guard let rp = ropeParameters else { return [:] }
        if sliding, let swa = rp["sliding_attention"] { return swa }
        if let full = rp["full_attention"] { return full }
        return [:]
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case decoderSparseStep = "decoder_sparse_step"
        case normTopkProb = "norm_topk_prob"
        case moeRoutedScalingFactor = "moe_routed_scaling_factor"
        case moeRouterLogitSoftcapping = "moe_router_logit_softcapping"
        case mlpOnlyLayers = "mlp_only_layers"
        case attentionHeadsPerLayer = "num_attention_heads_per_layer"
        case gating
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "laguna"
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        self.headDim = try c.decode(Int.self, forKey: .headDim)
        self.vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        self.rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        self.numExperts = try c.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerToken = try c.decode(Int.self, forKey: .numExpertsPerToken)
        self.moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        self.sharedExpertIntermediateSize = try c.decode(
            Int.self, forKey: .sharedExpertIntermediateSize)
        self.decoderSparseStep = try c.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? false
        self.moeRoutedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .moeRoutedScalingFactor) ?? 1.0
        // XS ships `null` here; S ships 0.0 — both mean "no softcap".
        self.moeRouterLogitSoftcapping = try c.decodeIfPresent(
            Float.self, forKey: .moeRouterLogitSoftcapping)
        self.mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? [0]
        self.attentionHeadsPerLayer = try c.decodeIfPresent(
            [Int].self, forKey: .attentionHeadsPerLayer)
        // `gating` is the string "per-head" / "per-element" in shipped configs,
        // but a raw HF export could carry a bool (True == per-element in the
        // Python reference's semantics).
        if let s = try? c.decodeIfPresent(String.self, forKey: .gating) {
            self.gating = s ?? "per-head"
        } else if let b = try? c.decodeIfPresent(Bool.self, forKey: .gating) {
            self.gating = (b ?? true) ? "per-element" : ""
        } else {
            self.gating = "per-head"
        }
        self.slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        self.layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
        self.ropeParameters = try c.decodeIfPresent(
            [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
        self.maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1_048_576
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

// MARK: - Attention

private class LagunaAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let isSliding: Bool
    let gatingEnabled: Bool
    let gatePerHead: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "g_proj") var wg: Linear?

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: LagunaConfiguration, layerIdx: Int) {
        let dim = args.hiddenSize
        let heads = args.headsForLayer(layerIdx)
        self.nHeads = heads
        self.nKVHeads = args.kvHeads
        let headDim = args.headDim
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, args.kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, args.kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        self.gatingEnabled = !args.gating.isEmpty
        self.gatePerHead = args.gating == "per-head"
        if gatingEnabled {
            let gOut = gatePerHead ? heads : heads * headDim
            _wg.wrappedValue = Linear(dim, gOut, bias: false)
        } else {
            _wg.wrappedValue = nil
        }

        self.isSliding = args.resolvedLayerTypes[layerIdx] == "sliding_attention"

        let ropeCfg = args.ropeConfig(sliding: isSliding)
        let theta = ropeCfg["rope_theta"]?.asFloat() ?? 10000.0
        let partial = ropeCfg["partial_rotary_factor"]?.asFloat() ?? 1.0
        let dims = Int(Float(headDim) * partial)
        let ropeType: String =
            if case .string(let s) = ropeCfg["rope_type"] ?? .string("default") { s } else {
                "default"
            }
        if ropeType == "default" || ropeType == "linear" {
            self.rope = RoPE(dimensions: dims, traditional: false, base: theta, scale: 1.0)
        } else {
            self.rope = initializeRope(
                dims: dims, base: theta, traditional: false,
                scalingConfig: ropeCfg,
                maxPositionEmbeddings: args.maxPositionEmbeddings)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, nHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, nKVHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        var out = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, nHeads * headDim)

        // Softplus output gate, before o_proj; fp32 like both references.
        if gatingEnabled, let wg {
            let g = softplus(wg(x).asType(.float32)).asType(out.dtype)
            if gatePerHead {
                out = (out.reshaped(B, L, nHeads, headDim) * g[.ellipsis, .newAxis])
                    .reshaped(B, L, -1)
            } else {
                out = out * g
            }
        }

        return wo(out)
    }
}

// MARK: - MLP / MoE

private class LagunaMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private class LagunaSparseMoeBlock: Module, UnaryLayer {
    let numExperts: Int
    let topK: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float
    let softcap: Float

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaMLP

    init(_ args: LagunaConfiguration) {
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerToken
        self.normTopkProb = args.normTopkProb
        self.routedScalingFactor = args.moeRoutedScalingFactor
        self.softcap = args.moeRouterLogitSoftcapping ?? 0.0

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _eScoreCorrectionBias.wrappedValue = zeros([args.numExperts])
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts)
        _sharedExpert.wrappedValue = LagunaMLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.sharedExpertIntermediateSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Router is precision-sensitive (near-tied sigmoid scores flip expert
        // choice and compound over 47 MoE layers) — fp32 like both references.
        var logits = gate(x).asType(.float32)
        if softcap > 0 {
            logits = tanh(logits / softcap) * softcap
        }
        let scores = sigmoid(logits)
        let choice = scores + eScoreCorrectionBias.asType(.float32)

        let k = topK
        let inds = MLX.argPartition(-choice, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var weights = MLX.takeAlong(scores, inds, axis: -1)
        if normTopkProb {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        weights = (weights * routedScalingFactor).asType(x.dtype)

        let y = switchMLP(x, inds)
        return (y * weights[.ellipsis, .newAxis]).sum(axis: -2) + sharedExpert(x)
    }
}

// MARK: - Decoder / trunk

private class LagunaDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: LagunaAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    fileprivate let mlp: UnaryLayer
    let isSliding: Bool

    init(_ args: LagunaConfiguration, layerIdx: Int) {
        _selfAttn.wrappedValue = LagunaAttention(args, layerIdx: layerIdx)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        let isMoE =
            !args.mlpOnlyLayers.contains(layerIdx) && args.numExperts > 0
            && (layerIdx + 1) % args.decoderSparseStep == 0
        if isMoE {
            self.mlp = LagunaSparseMoeBlock(args)
        } else {
            self.mlp = LagunaMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        }
        self.isSliding = args.resolvedLayerTypes[layerIdx] == "sliding_attention"
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class LagunaModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [LagunaDecoderLayer]
    let norm: RMSNorm
    let args: LagunaConfiguration
    private let firstFullIdx: Int
    private let firstSwaIdx: Int
    private let hasSwa: Bool

    init(_ args: LagunaConfiguration) {
        self.args = args
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { LagunaDecoderLayer(args, layerIdx: $0) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        let types = args.resolvedLayerTypes
        self.firstFullIdx = types.firstIndex(of: "full_attention") ?? 0
        self.firstSwaIdx = types.firstIndex(of: "sliding_attention") ?? 0
        self.hasSwa = types.contains("sliding_attention")
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        // One mask per attention type, computed against a cache of that type
        // (their offsets differ once the rotating window has wrapped).
        let fullMask = createAttentionMask(h: h, cache: cache?[firstFullIdx])
        let swaMask: MLXFast.ScaledDotProductAttentionMaskMode =
            hasSwa
            ? createAttentionMask(
                h: h, cache: cache?[firstSwaIdx], windowSize: args.slidingWindow)
            : fullMask

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: layer.isSliding ? swaMask : fullMask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class LagunaModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: LagunaModelInner
    let configuration: LagunaConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: LagunaConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = LagunaModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Sliding layers get a rotating window cache; full layers a standard one.
    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        configuration.resolvedLayerTypes.map { t in
            if t == "sliding_attention", let window = configuration.slidingWindow {
                return RotatingKVCache(maxSize: window, keep: 0)
            }
            return StandardKVCache()
        }
    }

    /// Reconciles the two checkpoint lineages onto this module tree:
    ///
    /// | key                                   | lineage             | action |
    /// |---------------------------------------|---------------------|--------|
    /// | `language_model.` prefix on everything | Blaizzy (XS quants) | strip  |
    /// | `mlp.gate.proj.{weight,scales,biases}` | Blaizzy             | → `mlp.gate.*` |
    /// | `mlp.gate.e_score_correction_bias`     | Blaizzy             | → `mlp.e_score_correction_bias` |
    /// | `mlp.experts.N.*` (per-expert)         | raw HF export       | stack → `mlp.switch_mlp.*` |
    /// | `mlp.experts.e_score_correction_bias`  | raw HF export       | → `mlp.e_score_correction_bias` |
    ///
    /// PipeNetwork quants (Laguna-S) already match the module tree 1:1.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = [String: MLXArray]()
        w.reserveCapacity(weights.count)
        for (key, value) in weights {
            var k = key
            if k.hasPrefix("language_model.") {
                k.removeFirst("language_model.".count)
            }
            if k.contains(".mlp.gate.proj.") {
                k = k.replacingOccurrences(of: ".mlp.gate.proj.", with: ".mlp.gate.")
            }
            if k.hasSuffix(".mlp.gate.e_score_correction_bias") {
                k = String(k.dropLast(".gate.e_score_correction_bias".count))
                    + ".e_score_correction_bias"
            }
            if k.hasSuffix(".mlp.experts.e_score_correction_bias") {
                k = String(k.dropLast(".experts.e_score_correction_bias".count))
                    + ".e_score_correction_bias"
            }
            w[k] = value
        }

        if configuration.tieWordEmbeddings {
            w["lm_head.weight"] = nil
            w["lm_head.scales"] = nil
            w["lm_head.biases"] = nil
        }

        // Raw HF exports carry per-expert projections; stack into SwitchGLU 3D.
        if w["model.layers.1.mlp.experts.0.gate_proj.weight"] != nil {
            for l in 0 ..< configuration.hiddenLayers {
                let prefix = "model.layers.\(l).mlp"
                guard w["\(prefix).experts.0.gate_proj.weight"] != nil else { continue }
                for n in ["gate_proj", "up_proj", "down_proj"] {
                    let stacked = (0 ..< configuration.numExperts).map { e in
                        w.removeValue(forKey: "\(prefix).experts.\(e).\(n).weight")!
                    }
                    w["\(prefix).switch_mlp.\(n).weight"] = MLX.stacked(stacked)
                }
            }
        }

        return w
    }
}

// MARK: - LoRA

extension LagunaModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
