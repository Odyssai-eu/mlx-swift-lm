//
//  KimiLinear.swift
//  LLM
//
//  Port of mlx-lm kimi_linear.py — Kimi-Linear-48B-A3B (KimiLinearForCausalLM).
//  Hybrid: full-attention layers use MLA (absorb, embed_q/unembed_out MultiLinear
//  — identical to GLM4-MoE-Lite), linear layers use Kimi Delta Attention (KDA)
//  = short depthwise conv on q/k/v + gated_delta recurrence. MoE = grouped-topk
//  sigmoid router + shared experts. Layer routing by linear_attn_config.kda_layers
//  (1-indexed: is_linear = (idx+1) in kda_layers).
//
//  Reuses the fork's MultiLinear/QuantizedMultiLinear (GLM4MOELite.swift) and
//  gatedDeltaUpdate (GatedDelta.swift). The MLA absorb + kv_b_proj split mirror
//  GLM4-MoE-Lite exactly.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private func kimiCallMultiLinear(_ module: Module, _ x: MLXArray) -> MLXArray {
    if let m = module as? MultiLinear { return m(x) }
    if let q = module as? QuantizedMultiLinear { return q(x) }
    fatalError("embed_q/unembed_out must be MultiLinear or QuantizedMultiLinear")
}

private func kimiRMSNormNoWeight(_ x: MLXArray, eps: Float) -> MLXArray {
    let v = x.square().mean(axis: -1, keepDims: true)
    return x * rsqrt(v + eps)
}

// MARK: - MLP

class KimiMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: KimiLinearConfiguration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        let dim = hiddenSize ?? args.hiddenSize
        let hidden = intermediateSize ?? args.intermediateSize
        _gateProj.wrappedValue = Linear(dim, hidden, bias: false)
        _upProj.wrappedValue = Linear(dim, hidden, bias: false)
        _downProj.wrappedValue = Linear(hidden, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Grouped-topk router

private func kimiGroupExpertSelect(
    gates: MLXArray, bias: MLXArray, topK: Int, nGroup: Int, topkGroup: Int,
    routedScalingFactor: Float, renormalize: Bool, scoreFunction: String
) -> (MLXArray, MLXArray) {
    var scores = scoreFunction == "sigmoid"
        ? sigmoid(gates.asType(.float32))
        : softmax(gates.asType(.float32), axis: -1, precise: true)
    let originalScores = scores
    scores = scores + bias.asType(scores.dtype)

    if nGroup > 1 {
        scores = unflatten(scores, axis: -1, shape: [nGroup, -1])
        let groupScores = top(scores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
        let k = nGroup - topkGroup
        let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
        scores = putAlong(scores, stopGradient(groupIdx), values: MLXArray(0.0), axis: -2)
        scores = flattened(scores, start: -2, end: -1)
    }

    let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    scores = takeAlong(originalScores, inds, axis: -1)
    if topK > 1, renormalize {
        scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
    }
    return (inds, scores * routedScalingFactor)
}

class KimiSparseMoE: Module, UnaryLayer {
    let args: KimiLinearConfiguration
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray
    @ModuleInfo(key: "shared_experts") var sharedExperts: KimiMLP?

    init(_ args: KimiLinearConfiguration) {
        self.args = args
        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts)
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([args.numExperts], dtype: .float32)
        _sharedExperts.wrappedValue = args.numSharedExperts > 0
            ? KimiMLP(args, intermediateSize: args.moeIntermediateSize * args.numSharedExperts)
            : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, weights) = kimiGroupExpertSelect(
            gates: gate(x), bias: eScoreCorrectionBias,
            topK: args.numExpertsPerToken, nGroup: args.numExpertGroup,
            topkGroup: args.topkGroup, routedScalingFactor: args.routedScalingFactor,
            renormalize: args.moeRenormalize, scoreFunction: args.moeRouterActivationFunc)
        var out = switchMLP(x, inds)
        out = (out * weights[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts { out = out + sharedExperts(x) }
        return out
    }
}

// MARK: - MLA attention (full-attention layers), mirrors GLM4-MoE-Lite absorb

class KimiMLAAttention: Module {
    let numHeads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let qHeadDim: Int
    let vHeadDim: Int
    let kvLoraRank: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let rope: RoPE

    init(_ args: KimiLinearConfiguration) {
        self.numHeads = args.numAttentionHeads
        self.qkNopeHeadDim = args.qkNopeHeadDim ?? args.headDim
        self.qkRopeHeadDim = args.qkRopeHeadDim ?? 0
        self.qHeadDim = qkNopeHeadDim + qkRopeHeadDim
        self.vHeadDim = args.vHeadDim ?? args.headDim
        self.kvLoraRank = args.kvLoraRank
        self.scale = pow(Float(qHeadDim), -0.5)

        _qProj.wrappedValue = Linear(args.hiddenSize, numHeads * qHeadDim, bias: false)
        _kvAProjWithMqa.wrappedValue = Linear(args.hiddenSize, kvLoraRank + qkRopeHeadDim, bias: false)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: args.rmsNormEps)
        _embedQ.wrappedValue = MultiLinear(
            inputDims: qkNopeHeadDim, outputDims: kvLoraRank, numHeads: numHeads)
        _unembedOut.wrappedValue = MultiLinear(
            inputDims: kvLoraRank, outputDims: vHeadDim, numHeads: numHeads)
        _oProj.wrappedValue = Linear(numHeads * vHeadDim, args.hiddenSize, bias: false)
        self.rope = RoPE(dimensions: qkRopeHeadDim, traditional: false, base: args.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var q = qProj(x).reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = splitQ[0]
        var qPe = splitQ[1]

        let compressed = kvAProjWithMqa(x)
        let splitKv = split(compressed, indices: [kvLoraRank], axis: -1)
        var kvLatent = kvALayerNorm(splitKv[0])
        var kPe = splitKv[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        qPe = applyRotaryPosition(rope, to: qPe, offset: offset)
        kPe = applyRotaryPosition(rope, to: kPe, offset: offset)

        kvLatent = expandedDimensions(kvLatent, axis: 1)
        qNope = kimiCallMultiLinear(embedQ, qNope)

        var keys = concatenated([kvLatent, kPe], axis: -1)
        var values = kvLatent
        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }
        let queries = concatenated([qNope, qPe], axis: -1)

        var output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        output = kimiCallMultiLinear(unembedOut, output)
        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(output)
    }
}

// MARK: - Short depthwise conv (KDA)

class KimiShortConv1d: Module {
    let kernelSize: Int
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(channels: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        _conv.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: kernelSize, groups: channels, bias: false)
    }

    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let s = state ?? MLXArray.zeros([x.dim(0), kernelSize - 1, x.dim(2)], dtype: x.dtype)
        let convInput = concatenated([s, x], axis: 1)
        let out = silu(conv(convInput))
        let nKeep = kernelSize - 1
        let newState = convInput[0..., (convInput.dim(1) - nKeep)..., 0...]
        return (out, newState)
    }
}

// MARK: - Kimi Delta Attention (linear layers)

class KimiDeltaAttention: Module {
    let numHeads: Int
    let headDim: Int
    let projectionDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_conv") var qConv: KimiShortConv1d
    @ModuleInfo(key: "k_conv") var kConv: KimiShortConv1d
    @ModuleInfo(key: "v_conv") var vConv: KimiShortConv1d
    @ModuleInfo(key: "f_a_proj") var fAProj: Linear
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ args: KimiLinearConfiguration) {
        let cfg = args.linearAttnConfig
        self.numHeads = cfg.numHeads
        self.headDim = cfg.headDim
        self.projectionDim = numHeads * headDim
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(args.hiddenSize, projectionDim, bias: false)
        _kProj.wrappedValue = Linear(args.hiddenSize, projectionDim, bias: false)
        _vProj.wrappedValue = Linear(args.hiddenSize, projectionDim, bias: false)
        _qConv.wrappedValue = KimiShortConv1d(channels: projectionDim, kernelSize: cfg.shortConvKernelSize)
        _kConv.wrappedValue = KimiShortConv1d(channels: projectionDim, kernelSize: cfg.shortConvKernelSize)
        _vConv.wrappedValue = KimiShortConv1d(channels: projectionDim, kernelSize: cfg.shortConvKernelSize)
        _fAProj.wrappedValue = Linear(args.hiddenSize, headDim, bias: false)
        _fBProj.wrappedValue = Linear(headDim, projectionDim, bias: false)
        _bProj.wrappedValue = Linear(args.hiddenSize, numHeads, bias: false)
        _gAProj.wrappedValue = Linear(args.hiddenSize, headDim, bias: false)
        _gBProj.wrappedValue = Linear(headDim, projectionDim, bias: false)
        _aLog.wrappedValue = MLXArray.zeros([numHeads])
        _dtBias.wrappedValue = MLXArray.zeros([projectionDim])
        _oNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _oProj.wrappedValue = Linear(projectionDim, args.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let (B, T) = (x.dim(0), x.dim(1))

        let (qConvOut, qNew) = qConv(qProj(x), state: cache?[0])
        let (kConvOut, kNew) = kConv(kProj(x), state: cache?[1])
        let (vConvOut, vNew) = vConv(vProj(x), state: cache?[2])
        if let cache { cache[0] = qNew; cache[1] = kNew; cache[2] = vNew }

        var q = qConvOut.reshaped(B, T, numHeads, headDim)
        var k = kConvOut.reshaped(B, T, numHeads, headDim)
        let v = vConvOut.reshaped(B, T, numHeads, headDim)
        q = (scale * scale) * kimiRMSNormNoWeight(q, eps: 1e-6)
        k = scale * kimiRMSNormNoWeight(k, eps: 1e-6)

        let aLogits = fBProj(fAProj(x)).reshaped(B, T, numHeads, headDim)
        let bLogits = bProj(x).reshaped(B, T, numHeads)

        let (out, newSsm) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: aLogits, b: bLogits,
            aLog: aLog.reshaped(numHeads, 1), dtBias: dtBias.reshaped(numHeads, headDim),
            state: cache?[3], mask: nil)
        if let cache { cache[3] = newSsm }

        let gate = gBProj(gAProj(x)).reshaped(B, T, numHeads, headDim)
        let normed = oNorm(out.reshaped(B, T, numHeads, headDim)) * sigmoid(gate)
        return oProj(normed.reshaped(B, T, -1))
    }
}

// MARK: - Decoder layer

class KimiDecoderLayer: Module {
    let isLinear: Bool
    @ModuleInfo(key: "self_attn") var selfAttn: Module
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: KimiLinearConfiguration, layerIdx: Int) {
        self.isLinear = args.linearAttnConfig.kdaLayers.contains(layerIdx + 1)
        _selfAttn.wrappedValue = isLinear ? KimiDeltaAttention(args) : KimiMLAAttention(args)
        let isMoe = args.numExperts > 0 && layerIdx >= args.firstKDenseReplace
            && layerIdx % args.moeLayerFreq == 0
        _mlp.wrappedValue = isMoe ? KimiSparseMoE(args) : KimiMLP(args)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let y: MLXArray
        if isLinear {
            y = (selfAttn as! KimiDeltaAttention)(normed, cache: cache as? ArraysCache)
        } else {
            y = (selfAttn as! KimiMLAAttention)(normed, mask: mask, cache: cache)
        }
        let h = x + y
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Model

class KimiLinearModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [KimiDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    let attnIdx: Int

    init(_ args: KimiLinearConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.numHiddenLayers).map { KimiDecoderLayer(args, layerIdx: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        let kda = Set(args.linearAttnConfig.kdaLayers)
        self.attnIdx = (0 ..< args.numHiddenLayers).first { !kda.contains($0 + 1) } ?? 0
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache.map { [$0[attnIdx]] })
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

public class KimiLinearModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: KimiLinearConfiguration
    @ModuleInfo(key: "model") var model: KimiLinearModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: KimiLinearConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabSize
        self.kvHeads = Array(repeating: args.numKeyValueHeads, count: args.numHiddenLayers)
        _model.wrappedValue = KimiLinearModelInner(args)
        _lmHead.wrappedValue = args.tieWordEmbeddings
            ? nil : Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead { return lmHead(out) }
        return model.embedTokens.asLinear(out)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        model.layers.map { $0.isLinear ? ArraysCache(size: 4) : KVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights.filter { !$0.key.hasPrefix("model.mtp") }
        if configuration.tieWordEmbeddings { w["lm_head.weight"] = nil }

        for layerIdx in 0 ..< configuration.numHiddenLayers {
            let prefix = "model.layers.\(layerIdx)"

            // MoE expert stacking: block_sparse_moe/experts -> mlp.switch_mlp
            for (src, dst) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                let sp = "\(prefix).block_sparse_moe"
                if w["\(sp).experts.0.\(src).weight"] != nil {
                    let toJoin = (0 ..< configuration.numExperts).compactMap {
                        w.removeValue(forKey: "\(sp).experts.\($0).\(src).weight")
                    }
                    w["\(prefix).mlp.switch_mlp.\(dst).weight"] = MLX.stacked(toJoin)
                }
            }
            for name in ["gate_proj", "up_proj", "down_proj"] {
                if let v = w.removeValue(forKey: "\(prefix).block_sparse_moe.shared_experts.\(name).weight") {
                    w["\(prefix).mlp.shared_experts.\(name).weight"] = v
                }
            }
            if let v = w.removeValue(forKey: "\(prefix).block_sparse_moe.gate.weight") {
                w["\(prefix).mlp.gate.weight"] = v
            }
            if let v = w.removeValue(forKey: "\(prefix).block_sparse_moe.gate.e_score_correction_bias") {
                w["\(prefix).mlp.e_score_correction_bias"] = v
            }

            // KDA conv weights: {q,k,v}_conv1d -> {q,k,v}_conv.conv (moveaxis 2->1)
            for (srcName, dstName) in [("q_conv1d", "q_conv"), ("k_conv1d", "k_conv"), ("v_conv1d", "v_conv")] {
                if var v = w.removeValue(forKey: "\(prefix).self_attn.\(srcName).weight") {
                    if v.ndim == 3 { v = v.movedAxis(source: 2, destination: 1) }
                    w["\(prefix).self_attn.\(dstName).conv.weight"] = v
                }
            }
            let dtKey = "\(prefix).self_attn.dt_bias"
            if let v = w[dtKey], v.ndim > 1 { w[dtKey] = v.reshaped([-1]) }

            // MLA absorb: split kv_b_proj -> embed_q / unembed_out
            if let kvB = w.removeValue(forKey: "\(prefix).self_attn.kv_b_proj.weight") {
                let qkNope = configuration.qkNopeHeadDim ?? configuration.headDim
                let vHead = configuration.vHeadDim ?? configuration.headDim
                let nH = configuration.numAttentionHeads
                let reshaped = kvB.reshaped([nH, qkNope + vHead, configuration.kvLoraRank])
                let parts = split(reshaped, indices: [qkNope], axis: 1)
                w["\(prefix).self_attn.embed_q.weight"] = parts[0]
                w["\(prefix).self_attn.unembed_out.weight"] = parts[1].swappedAxes(1, 2)
            }
        }
        return w
    }
}

extension KimiLinearModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}

// MARK: - Configuration

public struct KimiLinearLinearAttnConfig: Codable, Sendable {
    public let kdaLayers: [Int]
    public let numHeads: Int
    public let headDim: Int
    public let shortConvKernelSize: Int

    enum CodingKeys: String, CodingKey {
        case kdaLayers = "kda_layers"
        case numHeads = "num_heads"
        case headDim = "head_dim"
        case shortConvKernelSize = "short_conv_kernel_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.kdaLayers = try c.decode([Int].self, forKey: .kdaLayers)
        self.numHeads = try c.decode(Int.self, forKey: .numHeads)
        self.headDim = try c.decode(Int.self, forKey: .headDim)
        self.shortConvKernelSize = try c.decodeIfPresent(Int.self, forKey: .shortConvKernelSize) ?? 4
    }
}

public struct KimiLinearConfiguration: Codable, Sendable {
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let intermediateSize: Int
    public let headDim: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float
    public let linearAttnConfig: KimiLinearLinearAttnConfig
    public let numExperts: Int
    public let moeIntermediateSize: Int
    public let kvLoraRank: Int
    public let tieWordEmbeddings: Bool
    public let qkNopeHeadDim: Int?
    public let qkRopeHeadDim: Int?
    public let vHeadDim: Int?
    public let numExpertsPerToken: Int
    public let numSharedExperts: Int
    public let moeRouterActivationFunc: String
    public let moeRenormalize: Bool
    public let routedScalingFactor: Float
    public let firstKDenseReplace: Int
    public let moeLayerFreq: Int
    public let numExpertGroup: Int
    public let topkGroup: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case headDim = "head_dim"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case linearAttnConfig = "linear_attn_config"
        case numExperts = "num_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case kvLoraRank = "kv_lora_rank"
        case tieWordEmbeddings = "tie_word_embeddings"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case numExpertsPerToken = "num_experts_per_token"
        case numSharedExperts = "num_shared_experts"
        case moeRouterActivationFunc = "moe_router_activation_func"
        case moeRenormalize = "moe_renormalize"
        case routedScalingFactor = "routed_scaling_factor"
        case firstKDenseReplace = "first_k_dense_replace"
        case moeLayerFreq = "moe_layer_freq"
        case numExpertGroup = "num_expert_group"
        case topkGroup = "topk_group"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "kimi_linear"
        self.vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        self.numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        self.numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        self.intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        self.headDim = try c.decode(Int.self, forKey: .headDim)
        self.ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        self.rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        self.linearAttnConfig = try c.decode(KimiLinearLinearAttnConfig.self, forKey: .linearAttnConfig)
        self.numExperts = try c.decode(Int.self, forKey: .numExperts)
        self.moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        self.kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        self.tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim)
        self.qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim)
        self.vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim)
        self.numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 1
        self.numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0
        self.moeRouterActivationFunc = try c.decodeIfPresent(String.self, forKey: .moeRouterActivationFunc) ?? "sigmoid"
        self.moeRenormalize = try c.decodeIfPresent(Bool.self, forKey: .moeRenormalize) ?? true
        self.routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        self.firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        self.moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        self.numExpertGroup = try c.decodeIfPresent(Int.self, forKey: .numExpertGroup) ?? 1
        self.topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
    }
}
