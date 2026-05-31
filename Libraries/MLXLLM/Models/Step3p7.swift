//
//  Step3p7.swift
//  mlx-swift-lm
//
//  Text-only Step-3.7 / Step-3p5 port for Step-3.7-Flash checkpoints.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private func boundedSwiGLU(gate: MLXArray, up: MLXArray, limit: Float?) -> MLXArray {
    guard let limit else {
        return silu(gate) * up
    }
    let clippedGate = clip(silu(gate), max: MLXArray(limit))
    let clippedUp = clip(up, min: MLXArray(-limit), max: MLXArray(limit))
    return clippedGate * clippedUp
}

class Step3p7Attention: Module {
    let config: Step3p7TextConfiguration
    let layerIndex: Int
    let isSlidingWindow: Bool
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let scale: Float
    let useHeadWiseAttentionGate: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "g_proj") var gateProj: Linear?

    let rope: RoPELayer

    init(_ config: Step3p7TextConfiguration, layerIndex: Int, isSlidingWindow: Bool) {
        self.config = config
        self.layerIndex = layerIndex
        self.isSlidingWindow = isSlidingWindow

        if isSlidingWindow, let other = config.attentionOtherSetting {
            self.numAttentionHeads = other.numAttentionHeads
            self.numKeyValueHeads = other.numAttentionGroups
            self.headDim = other.headDim ?? config.headDim
        } else {
            self.numAttentionHeads = config.attentionHeads
            self.numKeyValueHeads = config.attentionGroups
            self.headDim = config.headDim
        }
        self.scale = pow(Float(self.headDim), -0.5)
        self.useHeadWiseAttentionGate = config.useHeadWiseAttentionGate

        _wq.wrappedValue = Linear(
            config.hiddenSize, self.numAttentionHeads * self.headDim, bias: false)
        _wk.wrappedValue = Linear(
            config.hiddenSize, self.numKeyValueHeads * self.headDim, bias: false)
        _wv.wrappedValue = Linear(
            config.hiddenSize, self.numKeyValueHeads * self.headDim, bias: false)
        _wo.wrappedValue = Linear(
            self.numAttentionHeads * self.headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: self.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: self.headDim, eps: config.rmsNormEps)
        if self.useHeadWiseAttentionGate {
            _gateProj.wrappedValue = Linear(
                config.hiddenSize, self.numAttentionHeads, bias: false)
        }

        let partial = config.partialRotaryFactor(layerIndex: layerIndex)
        let rotaryDims = Int(Float(self.headDim) * partial)
        var scalingConfig = config.ropeScaling
        if let yarnOnlyTypes = config.yarnOnlyTypes, !yarnOnlyTypes.contains(config.layerTypes[layerIndex]) {
            scalingConfig = nil
        }
        self.rope = initializeRope(
            dims: rotaryDims,
            base: config.ropeTheta(layerIndex: layerIndex),
            traditional: false,
            scalingConfig: scalingConfig,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(wq(x).reshaped(B, L, numAttentionHeads, headDim))
            .transposed(0, 2, 1, 3)
        var keys = kNorm(wk(x).reshaped(B, L, numKeyValueHeads, headDim))
            .transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, numKeyValueHeads, headDim)
            .transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)

        if let gateProj {
            let gate = sigmoid(gateProj(x)).reshaped(B, L, numAttentionHeads, 1)
            output = output * gate
        }

        return wo(output.reshaped(B, L, -1))
    }
}

class Step3p7MLP: Module, UnaryLayer {
    let limit: Float?

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int, limit: Float?) {
        self.limit = limit
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(boundedSwiGLU(gate: gate(x), up: up(x), limit: limit))
    }
}

class Step3p7MoEGate: Module {
    let topK: Int
    let routedScalingFactor: Float
    let normTopKProb: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "router_bias") var routerBias: MLXArray

    init(_ config: Step3p7TextConfiguration) {
        self.topK = config.moeTopK
        self.routedScalingFactor = config.moeRouterScalingFactor
        self.normTopKProb = config.normExpertWeight
        _gate.wrappedValue = Linear(config.hiddenSize, config.moeNumExperts, bias: false)
        _routerBias.wrappedValue = MLXArray.zeros([config.moeNumExperts], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let gateProb = sigmoid(gate(x).asType(.float32))
        let selectionScores = gateProb + routerBias
        let k = topK
        let inds = argPartition(-selectionScores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var scores = takeAlong(gateProb, inds, axis: -1)
        if normTopKProb {
            scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
        }
        scores = scores * routedScalingFactor
        return (inds, scores)
    }
}

class Step3p7MoEBlock: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: Step3p7MoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "share_expert") var sharedExpert: Step3p7MLP

    init(_ config: Step3p7TextConfiguration, moeLimit: Float?, sharedLimit: Float?) {
        _gate.wrappedValue = Step3p7MoEGate(config)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.moeNumExperts
        )
        _sharedExpert.wrappedValue = Step3p7MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.sharedExpertDim,
            limit: sharedLimit
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(x.dtype)
        return y + sharedExpert(x)
    }
}

class Step3p7DecoderLayer: Module {
    let isSlidingWindow: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Step3p7Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    fileprivate let mlp: UnaryLayer

    init(_ config: Step3p7TextConfiguration, layerIndex: Int) {
        self.isSlidingWindow = config.layerTypes[layerIndex] == "sliding_attention"
        _selfAttn.wrappedValue = Step3p7Attention(
            config, layerIndex: layerIndex, isSlidingWindow: self.isSlidingWindow)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.moeLayers.contains(layerIndex) {
            self.mlp = Step3p7MoEBlock(
                config,
                moeLimit: config.swigluLimit(layerIndex: layerIndex),
                sharedLimit: config.swigluSharedLimit(layerIndex: layerIndex)
            )
        } else {
            self.mlp = Step3p7MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                limit: config.swigluSharedLimit(layerIndex: layerIndex)
            )
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class Step3p7ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Step3p7DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let fullAttentionIndex: Int
    let slidingAttentionIndex: Int?
    let slidingWindow: Int?

    init(_ config: Step3p7TextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { index in
            Step3p7DecoderLayer(config, layerIndex: index)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.fullAttentionIndex = config.layerTypes.firstIndex(of: "full_attention") ?? 0
        self.slidingAttentionIndex = config.layerTypes.firstIndex(of: "sliding_attention")
        self.slidingWindow = config.slidingWindow
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let fullMask = createAttentionMask(h: h, cache: cache?[fullAttentionIndex])
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let slidingAttentionIndex {
            slidingMask = createAttentionMask(
                h: h, cache: cache?[slidingAttentionIndex], windowSize: slidingWindow)
        } else {
            slidingMask = .none
        }

        for (index, layer) in layers.enumerated() {
            let mask = layer.isSlidingWindow ? slidingMask : fullMask
            h = layer(h, mask: mask, cache: cache?[index])
        }
        return norm(h)
    }
}

public class Step3p7Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Step3p7ModelInner
    let configuration: Step3p7TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Step3p7TextConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { index in
            let sliding = config.layerTypes[index] == "sliding_attention"
            return sliding ? (config.attentionOtherSetting?.numAttentionGroups ?? config.attentionGroups)
                : config.attentionGroups
        }
        self.model = Step3p7ModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
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
        var weights = weights

        if weights.keys.contains(where: { $0.hasPrefix("language_model.") }) {
            weights = Dictionary(uniqueKeysWithValues: weights.map { key, value in
                if key.hasPrefix("language_model.") {
                    return (String(key.dropFirst("language_model.".count)), value)
                }
                return (key, value)
            })
        }
        if weights.keys.contains(where: { $0.hasPrefix("model.language_model.") }) {
            weights = Dictionary(uniqueKeysWithValues: weights.map { key, value in
                if key.hasPrefix("model.language_model.") {
                    return (String(key.dropFirst("model.language_model.".count)), value)
                }
                return (key, value)
            })
        }

        let unflattened = ModuleParameters.unflattened(weights)
        if let languageModel = unflattened["language_model"] {
            weights = Dictionary(uniqueKeysWithValues: languageModel.flattened())
        }

        weights = weights.filter {
            !$0.key.contains("rotary_emb.inv_freq")
                && !$0.key.hasPrefix("model.layers.\(configuration.hiddenLayers).")
                && !$0.key.hasPrefix("model.layers.\(configuration.hiddenLayers + 1).")
                && !$0.key.hasPrefix("model.layers.\(configuration.hiddenLayers + 2).")
                && !$0.key.contains(".mtp.")
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        let remappings = [
            (".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."),
            (".moe.up_proj.", ".mlp.switch_mlp.up_proj."),
            (".moe.down_proj.", ".mlp.switch_mlp.down_proj."),
            (".moe.gate.", ".mlp.gate.gate."),
            (".moe.router_bias", ".mlp.gate.router_bias"),
            (".share_expert.", ".mlp.share_expert."),
        ]

        let isVanilla = weights.keys.contains { key in
            remappings.contains { src, dst in
                key.contains(src) && !key.contains(dst)
            }
        }

        var newWeights: [String: MLXArray] = [:]
        newWeights.reserveCapacity(weights.count)
        for (originalKey, originalValue) in weights {
            var key = originalKey
            var value = originalValue

            for (src, dst) in remappings {
                if key.contains(src), !key.contains(dst) {
                    key = key.replacingOccurrences(of: src, with: dst)
                    break
                }
            }

            if isVanilla, key.hasSuffix(".weight"), key.contains("norm"), value.ndim == 1 {
                value = value + MLXArray(1, dtype: value.dtype)
            }
            newWeights[key] = value
        }

        return newWeights
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        sanitize(weights: weights)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        model.layers.map { layer in
            if layer.isSlidingWindow, let slidingWindow = configuration.slidingWindow {
                return RotatingKVCache(maxSize: slidingWindow)
            }
            return KVCacheSimple()
        }
    }
}

public struct Step3p7AttentionOtherSetting: Codable, Sendable {
    var numAttentionHeads: Int
    var numAttentionGroups: Int
    var headDim: Int?

    enum CodingKeys: String, CodingKey {
        case numAttentionHeads = "num_attention_heads"
        case numAttentionGroups = "num_attention_groups"
        case headDim = "head_dim"
    }
}

public struct Step3p7TextConfiguration: Codable, Sendable {
    var modelType: String = "step3p7"
    var hiddenSize: Int = 4096
    var intermediateSize: Int = 11264
    var attentionHeads: Int = 64
    var attentionGroups: Int = 8
    var hiddenLayers: Int = 45
    var vocabularySize: Int = 128896
    var rmsNormEps: Float = 1e-5
    var moeIntermediateSize: Int = 1280
    var moeNumExperts: Int = 288
    var moeTopK: Int = 8
    var ropeThetaValues: [Float] = [10000]
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int = 128000
    var sharedExpertDim: Int = 1280
    var headDim: Int = 128
    var layerTypes: [String] = []
    var slidingWindow: Int?
    var tieWordEmbeddings: Bool = false
    var useHeadWiseAttentionGate: Bool = false
    var useMOERouterBias: Bool = true
    var moeRouterActivation: String = "sigmoid"
    var moeRouterScalingFactor: Float = 1.0
    var normExpertWeight: Bool = true
    var needFP32Gate: Bool = true
    var attentionOtherSetting: Step3p7AttentionOtherSetting?
    var swigluLimits: [Float?] = []
    var swigluSharedLimits: [Float?] = []
    var partialRotaryFactors: [Float] = []
    var yarnOnlyTypes: [String]?
    var moeLayers: Set<Int> = Set(3 ..< 45)

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case attentionGroups = "num_attention_groups"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeNumExperts = "moe_num_experts"
        case moeTopK = "moe_top_k"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case sharedExpertDim = "share_expert_dim"
        case sharedExpertDims = "share_expert_dims"
        case headDim = "head_dim"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useHeadWiseAttentionGate = "use_head_wise_attn_gate"
        case useMOERouterBias = "use_moe_router_bias"
        case moeRouterActivation = "moe_router_activation"
        case moeRouterScalingFactor = "moe_router_scaling_factor"
        case normExpertWeight = "norm_expert_weight"
        case needFP32Gate = "need_fp32_gate"
        case attentionOtherSetting = "attention_other_setting"
        case swigluLimits = "swiglu_limits"
        case swigluSharedLimits = "swiglu_limits_shared"
        case partialRotaryFactors = "partial_rotary_factors"
        case yarnOnlyTypes = "yarn_only_types"
        case moeLayersEnum = "moe_layers_enum"
    }

    enum TopLevelCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let topLevel = try decoder.container(keyedBy: TopLevelCodingKeys.self)
        let container =
            if topLevel.contains(.textConfig) {
                try topLevel.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "step3p7"
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11264
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 64
        attentionGroups = try container.decodeIfPresent(Int.self, forKey: .attentionGroups) ?? 8
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 45
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 128896
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1280
        moeNumExperts = try container.decodeIfPresent(Int.self, forKey: .moeNumExperts) ?? 288
        moeTopK = try container.decodeIfPresent(Int.self, forKey: .moeTopK) ?? 8
        ropeThetaValues = try Step3p7TextConfiguration.decodeFloatOrFloatArray(
            container, forKey: .ropeTheta, defaultValue: [10000])
        ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 128000
        sharedExpertDim =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertDim)
            ?? container.decodeIfPresent(Int.self, forKey: .sharedExpertDims)
            ?? 1280
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        if layerTypes.isEmpty {
            layerTypes = (0 ..< hiddenLayers).map { $0 % 4 == 0 ? "full_attention" : "sliding_attention" }
        }
        if layerTypes.count > hiddenLayers {
            layerTypes = Array(layerTypes.prefix(hiddenLayers))
        } else if layerTypes.count < hiddenLayers, let last = layerTypes.last {
            layerTypes += Array(repeating: last, count: hiddenLayers - layerTypes.count)
        }
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        useHeadWiseAttentionGate =
            try container.decodeIfPresent(Bool.self, forKey: .useHeadWiseAttentionGate) ?? false
        useMOERouterBias = try container.decodeIfPresent(Bool.self, forKey: .useMOERouterBias) ?? true
        moeRouterActivation =
            try container.decodeIfPresent(String.self, forKey: .moeRouterActivation) ?? "sigmoid"
        moeRouterScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .moeRouterScalingFactor) ?? 1.0
        normExpertWeight = try container.decodeIfPresent(Bool.self, forKey: .normExpertWeight) ?? true
        needFP32Gate = try container.decodeIfPresent(Bool.self, forKey: .needFP32Gate) ?? true
        attentionOtherSetting =
            try container.decodeIfPresent(Step3p7AttentionOtherSetting.self, forKey: .attentionOtherSetting)
        swigluLimits = try Step3p7TextConfiguration.decodeOptionalFloatArray(
            container, forKey: .swigluLimits)
        swigluSharedLimits = try Step3p7TextConfiguration.decodeOptionalFloatArray(
            container, forKey: .swigluSharedLimits)
        partialRotaryFactors =
            try container.decodeIfPresent([Float].self, forKey: .partialRotaryFactors) ?? []
        if partialRotaryFactors.isEmpty {
            partialRotaryFactors = Array(repeating: 1.0, count: hiddenLayers)
        }
        if partialRotaryFactors.count > hiddenLayers {
            partialRotaryFactors = Array(partialRotaryFactors.prefix(hiddenLayers))
        } else if partialRotaryFactors.count < hiddenLayers, let last = partialRotaryFactors.last {
            partialRotaryFactors += Array(repeating: last, count: hiddenLayers - partialRotaryFactors.count)
        }
        yarnOnlyTypes = try container.decodeIfPresent([String].self, forKey: .yarnOnlyTypes)
        moeLayers = try Step3p7TextConfiguration.decodeMOELayers(container, forKey: .moeLayersEnum)
            ?? Set(3 ..< hiddenLayers)
    }

    public init() {}

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(attentionGroups, forKey: .attentionGroups)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try container.encode(moeNumExperts, forKey: .moeNumExperts)
        try container.encode(moeTopK, forKey: .moeTopK)
        try container.encode(ropeThetaValues, forKey: .ropeTheta)
        try container.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(sharedExpertDim, forKey: .sharedExpertDim)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(layerTypes, forKey: .layerTypes)
        try container.encodeIfPresent(slidingWindow, forKey: .slidingWindow)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(useHeadWiseAttentionGate, forKey: .useHeadWiseAttentionGate)
        try container.encode(useMOERouterBias, forKey: .useMOERouterBias)
        try container.encode(moeRouterActivation, forKey: .moeRouterActivation)
        try container.encode(moeRouterScalingFactor, forKey: .moeRouterScalingFactor)
        try container.encode(normExpertWeight, forKey: .normExpertWeight)
        try container.encode(needFP32Gate, forKey: .needFP32Gate)
        try container.encodeIfPresent(attentionOtherSetting, forKey: .attentionOtherSetting)
        try container.encode(swigluLimits, forKey: .swigluLimits)
        try container.encode(swigluSharedLimits, forKey: .swigluSharedLimits)
        try container.encode(partialRotaryFactors, forKey: .partialRotaryFactors)
        try container.encodeIfPresent(yarnOnlyTypes, forKey: .yarnOnlyTypes)
        try container.encode(Array(moeLayers).sorted(), forKey: .moeLayersEnum)
    }

    func ropeTheta(layerIndex: Int) -> Float {
        if ropeThetaValues.count == 1 { return ropeThetaValues[0] }
        return ropeThetaValues[min(layerIndex, ropeThetaValues.count - 1)]
    }

    func partialRotaryFactor(layerIndex: Int) -> Float {
        partialRotaryFactors[min(layerIndex, partialRotaryFactors.count - 1)]
    }

    func swigluLimit(layerIndex: Int) -> Float? {
        guard layerIndex < swigluLimits.count else { return nil }
        guard let value = swigluLimits[layerIndex], value != 0 else { return nil }
        return value
    }

    func swigluSharedLimit(layerIndex: Int) -> Float? {
        guard layerIndex < swigluSharedLimits.count else { return nil }
        guard let value = swigluSharedLimits[layerIndex], value != 0 else { return nil }
        return value
    }

    private static func decodeFloatOrFloatArray(
        _ container: KeyedDecodingContainer<CodingKeys>, forKey key: CodingKeys, defaultValue: [Float]
    ) throws -> [Float] {
        if let array = try? container.decode([Float].self, forKey: key) {
            return array
        }
        if let value = try? container.decode(Float.self, forKey: key) {
            return [value]
        }
        return defaultValue
    }

    private static func decodeOptionalFloatArray(
        _ container: KeyedDecodingContainer<CodingKeys>, forKey key: CodingKeys
    ) throws -> [Float?] {
        if let values = try? container.decode([Float?].self, forKey: key) {
            return values
        }
        return []
    }

    private static func decodeMOELayers(
        _ container: KeyedDecodingContainer<CodingKeys>, forKey key: CodingKeys
    ) throws -> Set<Int>? {
        if let values = try? container.decode([Int].self, forKey: key) {
            return Set(values)
        }
        if let value = try? container.decode(String.self, forKey: key) {
            let layers = value.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            return Set(layers)
        }
        return nil
    }
}

extension Step3p7Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
