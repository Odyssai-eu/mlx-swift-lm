//
//  Gemma4Assistant.swift
//  mlx-swift-lm
//
//  Native Gemma 4 sidecar MTP drafter.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Gemma4AssistantConfiguration: Codable, Sendable {
    var modelType: String = "gemma4_assistant"
    var textConfig: Gemma4TextConfiguration
    var backboneHiddenSize: Int
    var vocabSize: Int = 262144

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case backboneHiddenSize = "backbone_hidden_size"
        case vocabSize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "gemma4_assistant"
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        if let backboneHiddenSize = try container.decodeIfPresent(
            Int.self, forKey: .backboneHiddenSize)
        {
            self.backboneHiddenSize = backboneHiddenSize
        } else if let textConfig = try container.decodeIfPresent(
            Gemma4TextConfiguration.self, forKey: .textConfig)
        {
            self.backboneHiddenSize = textConfig.hiddenSize
        } else {
            self.backboneHiddenSize = 1536
        }

        if var textConfig = try container.decodeIfPresent(
            Gemma4TextConfiguration.self, forKey: .textConfig)
        {
            textConfig.vocabSize = self.vocabSize
            self.textConfig = textConfig
        } else {
            self.textConfig = try Gemma4TextConfiguration(from: decoder)
            self.textConfig.vocabSize = self.vocabSize
        }
    }

    public func targetLayerIndices(
        targetLayerTypes: [String],
        targetNumKVSharedLayers: Int = 0
    ) throws -> [Int] {
        let nonSharedCount = max(targetLayerTypes.count - targetNumKVSharedLayers, 0)
        var lastTargetByType = [String: Int]()
        for (idx, layerType) in targetLayerTypes.prefix(nonSharedCount).enumerated() {
            lastTargetByType[layerType] = idx
        }

        return try textConfig.layerTypes.map { layerType in
            guard let targetIdx = lastTargetByType[layerType] else {
                throw Gemma4AssistantError.noTargetLayer(attentionType: layerType)
            }
            return targetIdx
        }
    }
}

public struct Gemma4AssistantSharedKV {
    public let keys: MLXArray
    public let values: MLXArray
    public let positionOffset: RoPEOffset?

    public init(keys: MLXArray, values: MLXArray, positionOffset: RoPEOffset? = nil) {
        self.keys = keys
        self.values = values
        self.positionOffset = positionOffset
    }

    public init?(cache: KVCache) {
        let state = cache.state
        guard state.count == 2 else { return nil }
        self.keys = state[0]
        self.values = state[1]
        self.positionOffset = cache.ropeOffset
    }
}

public struct Gemma4AssistantOutput {
    public let draftHiddenStates: MLXArray
    public let backboneHiddenStates: MLXArray
    public let logits: MLXArray
}

public enum Gemma4AssistantError: Error, CustomStringConvertible {
    case noTargetLayer(attentionType: String)
    case targetCacheMissing(index: Int)
    case targetCacheEmpty(index: Int)

    public var description: String {
        switch self {
        case .noTargetLayer(let attentionType):
            "no target Gemma4 layer found for assistant attention type '\(attentionType)'"
        case .targetCacheMissing(let index):
            "target Gemma4 cache missing at layer \(index)"
        case .targetCacheEmpty(let index):
            "target Gemma4 cache at layer \(index) has no K/V state"
        }
    }
}

private class Gemma4AssistantAttention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let effectiveHeadDim: Int
    let nHeads: Int
    let scale: Float = 1.0

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo var rope: RoPELayer

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.effectiveHeadDim = isSliding ? config.headDim : config.globalHeadDim
        self.nHeads = config.numAttentionHeads

        self._qProj.wrappedValue = Linear(
            config.hiddenSize, nHeads * effectiveHeadDim, bias: false)
        self._oProj.wrappedValue = Linear(
            nHeads * effectiveHeadDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)

        if isSliding {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.slidingRopeTheta, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
        } else {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.fullRopeTheta, traditional: false,
                scalingConfig: [
                    "type": .string("proportional"),
                    "partial_rotary_factor": .float(config.fullPartialRotaryFactor),
                ],
                maxPositionEmbeddings: nil)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        sharedKV: Gemma4AssistantSharedKV,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        var queries = qProj(x).reshaped(B, L, nHeads, effectiveHeadDim)
        queries = qNorm(queries).transposed(0, 2, 1, 3)
        let offset = sharedKV.positionOffset ?? .scalar(max(sharedKV.keys.dim(2) - L, 0))
        queries = applyRotaryPosition(rope, to: queries, offset: offset)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: sharedKV.keys,
            values: sharedKV.values,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

private class Gemma4AssistantMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma4TextConfiguration) {
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

private class Gemma4AssistantDecoderLayer: Module {
    let layerType: String

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4AssistantAttention
    @ModuleInfo var mlp: Gemma4AssistantMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.layerType = config.layerTypes[layerIdx]
        self._selfAttn.wrappedValue = Gemma4AssistantAttention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma4AssistantMLP(config)
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._layerScalar.wrappedValue = MLXArray.ones([1], dtype: .float16)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        sharedKV: Gemma4AssistantSharedKV,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let residual = x
        let attnOut = selfAttn(inputLayernorm(x), sharedKV: sharedKV, mask: mask)
        var out = residual + postAttentionLayernorm(attnOut)

        let residual2 = out
        out = mlp(preFeedforwardLayernorm(out))
        out = residual2 + postFeedforwardLayernorm(out)

        return out * layerScalar
    }
}

private class Gemma4AssistantModelInner: Module {
    let config: Gemma4TextConfiguration
    let backboneHiddenSize: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4AssistantDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: Gemma4TextConfiguration, backboneHiddenSize: Int) {
        self.config = config
        self.backboneHiddenSize = backboneHiddenSize
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            Gemma4AssistantDecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ h: MLXArray,
        sharedKeyValues: [Gemma4AssistantSharedKV]
    ) -> MLXArray {
        precondition(
            sharedKeyValues.count == layers.count,
            "Gemma4Assistant requires one shared KV pair per assistant layer")

        var hiddenStates = h
        for (idx, layer) in layers.enumerated() {
            let sharedKV = sharedKeyValues[idx]
            let mask = Self.attentionMask(
                queryLength: hiddenStates.dim(1),
                keyLength: sharedKV.keys.dim(2),
                slidingWindow: layer.layerType == "sliding_attention" ? config.slidingWindow : nil
            )
            hiddenStates = layer(hiddenStates, sharedKV: sharedKV, mask: mask)
        }

        return norm(hiddenStates)
    }

    private static func attentionMask(
        queryLength: Int,
        keyLength: Int,
        slidingWindow: Int?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if queryLength == 1 {
            return .none
        }

        let offset = max(keyLength - queryLength, 0)
        return .array(createCausalMask(n: queryLength, offset: offset, windowSize: slidingWindow))
    }
}

public class Gemma4AssistantModel: Module, LLMModel {
    public let vocabularySize: Int
    public var layerTypes: [String] { config.textConfig.layerTypes }

    let config: Gemma4AssistantConfiguration
    fileprivate let model: Gemma4AssistantModelInner

    @ModuleInfo(key: "pre_projection") var preProjection: Linear
    @ModuleInfo(key: "post_projection") var postProjection: Linear
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Gemma4AssistantConfiguration) {
        self.config = config
        self.vocabularySize = config.textConfig.vocabSize
        self.model = Gemma4AssistantModelInner(
            config.textConfig, backboneHiddenSize: config.backboneHiddenSize)
        self._preProjection.wrappedValue = Linear(
            config.backboneHiddenSize * 2, config.textConfig.hiddenSize, bias: false)
        self._postProjection.wrappedValue = Linear(
            config.textConfig.hiddenSize, config.backboneHiddenSize, bias: false)
        if !config.textConfig.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false)
        }
        super.init()
    }

    public func draft(
        inputsEmbeds: MLXArray,
        feedbackHiddenStates: MLXArray,
        sharedKeyValues: [Gemma4AssistantSharedKV]
    ) -> Gemma4AssistantOutput {
        let combined = concatenated([inputsEmbeds, feedbackHiddenStates], axis: -1)
        let projected = preProjection(combined)
        let draftHiddenStates = model(projected, sharedKeyValues: sharedKeyValues)
        let backboneHiddenStates = postProjection(draftHiddenStates)

        let logits: MLXArray
        if let lmHead {
            logits = lmHead(draftHiddenStates)
        } else {
            logits = model.embedTokens.asLinear(draftHiddenStates)
        }

        return Gemma4AssistantOutput(
            draftHiddenStates: draftHiddenStates,
            backboneHiddenStates: backboneHiddenStates,
            logits: logits
        )
    }

    public func targetLayerIndices(
        targetLayerTypes: [String],
        targetNumKVSharedLayers: Int = 0
    ) throws -> [Int] {
        try config.targetLayerIndices(
            targetLayerTypes: targetLayerTypes,
            targetNumKVSharedLayers: targetNumKVSharedLayers)
    }

    public func sharedKeyValues(
        from targetCache: [KVCache],
        targetLayerTypes: [String],
        targetNumKVSharedLayers: Int = 0
    ) throws -> [Gemma4AssistantSharedKV] {
        try targetLayerIndices(
            targetLayerTypes: targetLayerTypes,
            targetNumKVSharedLayers: targetNumKVSharedLayers
        ).map { targetIdx in
            guard targetIdx < targetCache.count else {
                throw Gemma4AssistantError.targetCacheMissing(index: targetIdx)
            }
            guard let sharedKV = Gemma4AssistantSharedKV(cache: targetCache[targetIdx]) else {
                throw Gemma4AssistantError.targetCacheEmpty(index: targetIdx)
            }
            return sharedKV
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError(
            "Gemma4AssistantModel is a sidecar drafter. Use draft(inputsEmbeds:feedbackHiddenStates:sharedKeyValues:) with Gemma4 backbone embeddings and cache."
        )
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        []
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("rotary_emb")
                || key.contains("input_max")
                || key.contains("input_min")
                || key.contains("output_max")
                || key.contains("output_min")
                || key.hasPrefix("masked_embedding.")
            {
                continue
            }
            sanitized[key] = value
        }
        return sanitized
    }
}

extension Gemma4AssistantModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers.map { $0.selfAttn }
    }
}
