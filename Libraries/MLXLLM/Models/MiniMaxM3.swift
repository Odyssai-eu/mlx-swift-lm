//
//  MiniMaxM3.swift
//  mlx-swift-lm
//
//  Text-only MiniMax-M3 port. WU1 intentionally uses full causal attention
//  for every layer; the MSA indexer is left out until the 128k-context follow-up.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private func miniMaxM3SwiGLUOAI(
    up: MLXArray, gate: MLXArray, alpha: Float = 1.702, limit: Float = 7.0
) -> MLXArray {
    let clippedGate = clip(gate, max: MLXArray(limit))
    let clippedUp = clip(up, min: MLXArray(-limit), max: MLXArray(limit))
    let glu = clippedGate * sigmoid(clippedGate * alpha)
    return (clippedUp + 1.0) * glu
}

class MiniMaxM3RMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1.0 + weight, eps: eps)
    }
}

class MiniMaxM3Attention: Module {
    let args: MiniMaxM3Configuration
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: MiniMaxM3RMSNorm

    let rope: RoPE

    init(_ args: MiniMaxM3Configuration) {
        self.args = args
        self.numAttentionHeads = args.attentionHeads
        self.numKeyValueHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(args.hiddenSize, numAttentionHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(args.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(args.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(numAttentionHeads * headDim, args.hiddenSize, bias: false)

        _qNorm.wrappedValue = MiniMaxM3RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = MiniMaxM3RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        self.rope = RoPE(
            dimensions: args.rotaryDim,
            traditional: false,
            base: args.ropeTheta,
            scale: 1.0
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var q = wq(x).reshaped(B, L, numAttentionHeads, headDim)
        var k = wk(x).reshaped(B, L, numKeyValueHeads, headDim)
        var v = wv(x).reshaped(B, L, numKeyValueHeads, headDim)

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

        return wo(output)
    }
}

class MiniMaxM3DenseMLP: Module, UnaryLayer {
    let alpha: Float
    let limit: Float

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: MiniMaxM3Configuration, intermediateSize: Int) {
        self.alpha = args.swigluAlpha
        self.limit = args.swigluLimit

        _gateProj.wrappedValue = Linear(args.hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(args.hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, args.hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(
            miniMaxM3SwiGLUOAI(
                up: upProj(x),
                gate: gateProj(x),
                alpha: alpha,
                limit: limit
            ))
    }
}

class MiniMaxM3MoEGate: Module {
    let topK: Int

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ args: MiniMaxM3Configuration) {
        self.topK = args.numExpertsPerTok

        _weight.wrappedValue = MLXArray.zeros([args.numLocalExperts, args.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([args.numLocalExperts])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = x.asType(.float32).matmul(weight.asType(.float32).T)
        let originalScores = sigmoid(logits)
        let selectionScores = originalScores + eScoreCorrectionBias.asType(.float32)

        let k = topK
        let inds = argPartition(-selectionScores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var scores = takeAlong(originalScores, inds, axis: -1)
        scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
        return (inds, scores)
    }
}

class MiniMaxM3SparseMoeBlock: Module, UnaryLayer {
    let gate: MiniMaxM3MoEGate
    let routedScalingFactor: Float

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: MiniMaxM3DenseMLP

    init(_ args: MiniMaxM3Configuration) {
        self.gate = MiniMaxM3MoEGate(args)
        self.routedScalingFactor = args.routedScalingFactor

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.intermediateSize,
            numExperts: args.numLocalExperts,
            gatedActivation: { up, gate in
                miniMaxM3SwiGLUOAI(
                    up: up,
                    gate: gate,
                    alpha: args.swigluAlpha,
                    limit: args.swigluLimit
                )
            }
        )
        _sharedExperts.wrappedValue = MiniMaxM3DenseMLP(
            args, intermediateSize: args.sharedIntermediateSize)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shared = sharedExperts(x)
        let (inds, weights) = gate(x)
        var y = switchMLP(x, inds)
        y = (y * weights.asType(y.dtype)[.ellipsis, .newAxis]).sum(axis: -2)
        return y * routedScalingFactor + shared
    }
}

class MiniMaxM3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxM3Attention
    @ModuleInfo(key: "mlp") var mlp: MiniMaxM3DenseMLP?
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MiniMaxM3SparseMoeBlock?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MiniMaxM3RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: MiniMaxM3RMSNorm

    init(_ args: MiniMaxM3Configuration, layerIndex: Int) {
        _selfAttn.wrappedValue = MiniMaxM3Attention(args)

        if args.isMoELayer(layerIndex) {
            _blockSparseMoe.wrappedValue = MiniMaxM3SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = MiniMaxM3DenseMLP(
                args, intermediateSize: args.denseIntermediateSize)
        }

        _inputLayerNorm.wrappedValue = MiniMaxM3RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = MiniMaxM3RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        if let blockSparseMoe {
            return h + blockSparseMoe(postAttentionLayerNorm(h))
        }
        guard let mlp else {
            fatalError("MiniMax-M3 decoder layer has neither dense MLP nor MoE.")
        }
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class MiniMaxM3ModelInner: Module {
    let args: MiniMaxM3Configuration

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    fileprivate let layers: [MiniMaxM3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: MiniMaxM3RMSNorm

    init(_ args: MiniMaxM3Configuration) {
        self.args = args

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { layerIndex in
            MiniMaxM3DecoderLayer(args, layerIndex: layerIndex)
        }
        _norm.wrappedValue = MiniMaxM3RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ inputs: MLXArray?, cache: [KVCache]?, inputEmbedding: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbedding {
            h = inputEmbedding
        } else if let inputs {
            h = embedTokens(inputs)
        } else {
            fatalError("one of inputs or inputEmbedding must be non-nil")
        }

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class MiniMaxM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MiniMaxM3ModelInner
    let configuration: MiniMaxM3Configuration
    let modelType: String

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: MiniMaxM3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.modelType = args.modelType
        self.model = MiniMaxM3ModelInner(args)

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

    /// Embedding-injection variant used by the MiniMax-M3-VL wrapper: the vision
    /// path splices image features into the token embeddings and feeds them here.
    public func callAsFunction(
        _ inputs: MLXArray?, cache: [KVCache]?, inputEmbedding: MLXArray?
    ) -> MLXArray {
        let out = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights: [String: MLXArray] = [:]

        for (rawKey, value) in weights {
            if rawKey.hasPrefix("vision_tower.")
                || rawKey.hasPrefix("multi_modal_projector.")
                || rawKey.hasPrefix("patch_merge_mlp.")
                || rawKey.contains(".self_attn.index_")
                || rawKey.hasPrefix("model.mtp")
            {
                continue
            }

            var key = rawKey
            if key.hasPrefix("language_model.") {
                key.removeFirst("language_model.".count)
            }
            key = key.replacingOccurrences(
                of: ".block_sparse_moe.e_score_correction_bias",
                with: ".block_sparse_moe.gate.e_score_correction_bias"
            )
            sanitizedWeights[key] = value
        }

        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let dtype = weight.dtype
            let bs = 128
            let (m, n) = (weight.dim(0), weight.dim(1))
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs

            var paddedWeight = padded(
                weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            paddedWeight = paddedWeight.reshaped(
                [(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = paddedWeight * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
                .asType(dtype)
        }

        var dequantizedWeights: [String: MLXArray] = [:]
        for (key, value) in sanitizedWeights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = sanitizedWeights[weightKey] {
                    dequantizedWeights[weightKey] = dequant(weight: weight, scaleInv: value)
                }
            } else if dequantizedWeights[key] == nil {
                dequantizedWeights[key] = value
            }
        }

        sanitizedWeights = dequantizedWeights.isEmpty ? sanitizedWeights : dequantizedWeights

        if sanitizedWeights["model.layers.3.block_sparse_moe.experts.0.w1.weight"] == nil {
            return sanitizedWeights
        }

        for layerIndex in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layerIndex).block_sparse_moe"
            for (orig, updated) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(orig).\(key)"
                    if sanitizedWeights[firstKey] != nil {
                        let toJoin = (0 ..< configuration.numLocalExperts).map { expertIndex in
                            sanitizedWeights.removeValue(
                                forKey: "\(prefix).experts.\(expertIndex).\(orig).\(key)"
                            )!
                        }
                        sanitizedWeights["\(prefix).switch_mlp.\(updated).\(key)"] =
                            MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitizedWeights
    }
}

public struct MiniMaxM3Configuration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var partialRotaryFactor: Float
    var rotaryDim: Int
    var maxPositionEmbeddings: Int
    var denseIntermediateSize: Int
    var intermediateSize: Int
    var sharedIntermediateSize: Int
    var numLocalExperts: Int
    var numExpertsPerTok: Int
    var routedScalingFactor: Float
    var swigluAlpha: Float
    var swigluLimit: Float
    var tieWordEmbeddings: Bool
    var moeLayerFreq: [Int]?
    var mlpLayerTypes: [String]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case rotaryDim = "rotary_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case denseIntermediateSize = "dense_intermediate_size"
        case intermediateSize = "intermediate_size"
        case sharedIntermediateSize = "shared_intermediate_size"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case routedScalingFactor = "routed_scaling_factor"
        case swigluAlpha = "swiglu_alpha"
        case swigluLimit = "swiglu_limit"
        case tieWordEmbeddings = "tie_word_embeddings"
        case moeLayerFreq = "moe_layer_freq"
        case mlpLayerTypes = "mlp_layer_types"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "minimax_m3"
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? 200064
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 6144
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 60
        self.attentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
            ?? 64
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        self.rmsNormEps =
            try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
            ?? 1e-6
        self.ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? 5_000_000
        self.partialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        self.rotaryDim =
            try container.decodeIfPresent(Int.self, forKey: .rotaryDim)
            ?? Int(Float(self.headDim) * self.partialRotaryFactor)
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1_048_576
        self.denseIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .denseIntermediateSize) ?? 12288
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        self.sharedIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedIntermediateSize) ?? 3072
        self.numLocalExperts =
            try container.decodeIfPresent(Int.self, forKey: .numLocalExperts) ?? 128
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 4
        self.routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2.0
        self.swigluAlpha =
            try container.decodeIfPresent(Float.self, forKey: .swigluAlpha)
            ?? 1.702
        self.swigluLimit =
            try container.decodeIfPresent(Float.self, forKey: .swigluLimit)
            ?? 7.0
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.moeLayerFreq = try container.decodeIfPresent([Int].self, forKey: .moeLayerFreq)
        self.mlpLayerTypes = try container.decodeIfPresent([String].self, forKey: .mlpLayerTypes)
    }

    func isMoELayer(_ index: Int) -> Bool {
        if let mlpLayerTypes, index < mlpLayerTypes.count {
            return mlpLayerTypes[index] == "sparse"
        }
        if let moeLayerFreq, index < moeLayerFreq.count {
            return moeLayerFreq[index] != 0
        }
        return index >= 3
    }
}

extension MiniMaxM3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
