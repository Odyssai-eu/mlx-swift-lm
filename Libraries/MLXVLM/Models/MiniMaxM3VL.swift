//
//  MiniMaxM3VL.swift
//  mlx-swift-lm
//
//  MiniMax-M3-VL port. The vision tower, the two projectors and the processor
//  are new; the language side reuses the MiniMax-M3 text trunk from MLXLLM via
//  embedding injection (the reference discards grids and visual position masks
//  on the language side and runs plain 1D positions, so no mRoPE plumbing is
//  needed).
//
//  Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/minimax_m3_vl
//

import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Vision

private enum Vision {

    /// Applies the partial rotary embedding used by the M3-VL vision tower.
    ///
    /// `cos`/`sin` cover `rotDim = 2 * 3 * (axisDim / 2)` dims (78 for head
    /// dim 80); the remaining tail dims pass through untouched. This is the
    /// second silent-divergence trap vs Qwen2VL, which rotates the full head.
    static func applyVisionRoPE(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        // x: (seq, heads, headDim); cos/sin: (seq, rotDim)
        let rotDim = cos.dim(-1)
        let cos = cos[0..., .newAxis, 0...]
        let sin = sin[0..., .newAxis, 0...]
        let xRot = x[.ellipsis, ..<rotDim]
        let xPass = x[.ellipsis, rotDim...]
        let rotated = (xRot * cos) + (QwenVL.rotateHalf(xRot) * sin)
        return concatenated([rotated, xPass], axis: -1).asType(x.dtype)
    }

    /// 3-band (t/h/w) rotary frequency builder (vision.py `_rotary_pos_emb`).
    ///
    /// Unlike Qwen2VL (h/w only, temporal tiled with no positional signal),
    /// every band carries a REAL coordinate: the temporal index within the
    /// segment, and absolute h/w patch coordinates laid out in merge-block
    /// order. Bands are equal thirds of `2 * ((headDim / 3) / 2)` dims.
    struct RotaryEmbedding {
        let axisDim: Int
        let inverseFreq: MLXArray

        init(headDim: Int, theta: Float) {
            let ropeDims = 2 * (headDim / 2)
            self.axisDim = 2 * ((ropeDims / 3) / 2)
            self.inverseFreq =
                QwenVL.VisionRotaryEmbedding(
                    dimensions: axisDim, theta: theta
                ).inverseFreq
        }

        /// Precomputed cos/sin tables of shape (totalSeq, 3 * axisDim), one
        /// row per patch across all segments.
        func cosSin(segments: [THW], mergeSize: Int) -> (MLXArray, MLXArray) {
            var freqsPerSegment = [MLXArray]()

            for segment in segments {
                let (t, h, w) = segment.values
                let mergedH = h / mergeSize
                let mergedW = w / mergeSize
                let count = t * h * w

                // Coordinates in merge-block order (t, h/m, w/m, m, m) — the
                // same flattened layout QwenVL.patchify emits. The temporal
                // index restarts at 0 for every segment.
                var tCoords = [Int32]()
                tCoords.reserveCapacity(count)
                var hCoords = [Int32]()
                hCoords.reserveCapacity(count)
                var wCoords = [Int32]()
                wCoords.reserveCapacity(count)

                for ti in 0 ..< t {
                    for hBlock in 0 ..< mergedH {
                        for wBlock in 0 ..< mergedW {
                            for intraH in 0 ..< mergeSize {
                                for intraW in 0 ..< mergeSize {
                                    tCoords.append(Int32(ti))
                                    hCoords.append(Int32(hBlock * mergeSize + intraH))
                                    wCoords.append(Int32(wBlock * mergeSize + intraW))
                                }
                            }
                        }
                    }
                }

                let freqs = concatenated(
                    [
                        outer(MLXArray(tCoords).asType(.float32), inverseFreq),
                        outer(MLXArray(hCoords).asType(.float32), inverseFreq),
                        outer(MLXArray(wCoords).asType(.float32), inverseFreq),
                    ], axis: -1)

                // duplicate the whole block for the rotate-half convention
                freqsPerSegment.append(concatenated([freqs, freqs], axis: -1))
            }

            let freqs = concatenated(freqsPerSegment, axis: 0)
            return (cos(freqs), sin(freqs))
        }
    }

    /// The checkpoint stores the patch embedding in the torch Conv3d layout
    /// [hidden, C, tps, pH, pW]; the reference (vision.py:60-65) runs it as a
    /// matmul on flattened 1176-dim patches — do NOT model this as Conv3d.
    fileprivate class PatchEmbedding: Module, UnaryLayer {
        @ParameterInfo(key: "weight") var weight: MLXArray

        let hiddenSize: Int
        let patchDim: Int

        init(_ config: MiniMaxM3VLConfiguration.VisionConfiguration) {
            self.hiddenSize = config.hiddenSize
            self.patchDim =
                config.numChannels * config.temporalPatchSize * config.patchSize
                * config.patchSize
            self._weight.wrappedValue = MLXArray.zeros([
                config.hiddenSize, config.numChannels, config.temporalPatchSize,
                config.patchSize, config.patchSize,
            ])
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            x.reshaped(-1, patchDim).matmul(weight.reshaped(hiddenSize, patchDim).T)
        }
    }

    fileprivate class Embeddings: Module, UnaryLayer {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: PatchEmbedding

        init(_ config: MiniMaxM3VLConfiguration.VisionConfiguration) {
            self._patchEmbedding.wrappedValue = PatchEmbedding(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            patchEmbedding(x)
        }
    }

    fileprivate class Attention: Module {
        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "out_proj") var wo: Linear

        init(dims: Int, numHeads: Int) {
            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dims, dims, bias: true)
            self._wk.wrappedValue = Linear(dims, dims, bias: true)
            self._wv.wrappedValue = Linear(dims, dims, bias: true)
            self._wo.wrappedValue = Linear(dims, dims, bias: true)

            super.init()
        }

        func callAsFunction(
            _ x: MLXArray, cuSeqlens: [Int], cos: MLXArray, sin: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)

            var q = wq(x).reshaped(sequenceLength, numHeads, -1)
            var k = wk(x).reshaped(sequenceLength, numHeads, -1)
            var v = wv(x).reshaped(sequenceLength, numHeads, -1)

            q = applyVisionRoPE(q, cos: cos, sin: sin)
            k = applyVisionRoPE(k, cos: cos, sin: sin)

            q = q.transposed(1, 0, 2)[.newAxis]
            k = k.transposed(1, 0, 2)[.newAxis]
            v = v.transposed(1, 0, 2)[.newAxis]

            // Per-segment (block-diagonal) attention: each image / video
            // segment attends only within itself. For a single image this
            // collapses to one full SDPA call.
            var outputs = [MLXArray]()
            for i in 1 ..< cuSeqlens.count {
                let (start, end) = (cuSeqlens[i - 1], cuSeqlens[i])
                outputs.append(
                    MLXFast.scaledDotProductAttention(
                        queries: q[0..., 0..., start ..< end, 0...],
                        keys: k[0..., 0..., start ..< end, 0...],
                        values: v[0..., 0..., start ..< end, 0...],
                        scale: scale,
                        mask: .none
                    ))
            }
            let output = outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)

            return wo(output[0].transposed(1, 0, 2).reshaped(sequenceLength, -1))
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            self.fc1 = Linear(dimensions, hiddenDimensions, bias: true)
            self.fc2 = Linear(hiddenDimensions, dimensions, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            // hidden_act "gelu" is the EXACT erf-based GELU — Qwen2VL's
            // `.fast` approximation here would silently skew the features.
            fc2(gelu(fc1(x)))
        }
    }

    fileprivate class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo(key: "mlp") var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: MiniMaxM3VLConfiguration.VisionConfiguration) {
            self._attention.wrappedValue = Attention(
                dims: config.hiddenSize, numHeads: config.numHeads)
            self._layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            self._mlp.wrappedValue = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            self._layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray, cuSeqlens: [Int], cos: MLXArray, sin: MLXArray
        ) -> MLXArray {
            var hidden = x + attention(layerNorm1(x), cuSeqlens: cuSeqlens, cos: cos, sin: sin)
            hidden = hidden + mlp(layerNorm2(hidden))
            return hidden
        }
    }

    fileprivate class Encoder: Module {
        @ModuleInfo(key: "layers") var layers: [EncoderLayer]

        init(_ config: MiniMaxM3VLConfiguration.VisionConfiguration) {
            self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
                EncoderLayer(config)
            }
            super.init()
        }
    }

    fileprivate class VisionTransformer: Module {
        @ModuleInfo(key: "embeddings") var embeddings: Embeddings
        // upstream typo ("layrnorm") — keep it, the checkpoint key matches
        @ModuleInfo(key: "pre_layrnorm") var preLayrnorm: LayerNorm
        @ModuleInfo(key: "encoder") var encoder: Encoder

        let rotaryEmbedding: RotaryEmbedding
        let mergeSize: Int
        let segmentMaxFrames: Int

        init(_ config: MiniMaxM3VLConfiguration.VisionConfiguration) {
            self._embeddings.wrappedValue = Embeddings(config)
            self._preLayrnorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            self._encoder.wrappedValue = Encoder(config)

            self.rotaryEmbedding = RotaryEmbedding(
                headDim: config.hiddenSize / config.numHeads, theta: config.ropeTheta)
            self.mergeSize = config.spatialMergeSize
            self.segmentMaxFrames = config.visionSegmentMaxFrames

            super.init()
        }

        /// Grids with more than `vision_segment_max_frames` frames are split
        /// into segments that get independent temporal coordinates and
        /// independent attention blocks. Still images (t = 1) pass through.
        func segmented(_ frames: [THW]) -> [THW] {
            var segments = [THW]()
            for frame in frames {
                let (t, h, w) = frame.values
                if t <= segmentMaxFrames {
                    segments.append(frame)
                } else {
                    var start = 0
                    while start < t {
                        segments.append(THW(min(segmentMaxFrames, t - start), h, w))
                        start += segmentMaxFrames
                    }
                }
            }
            return segments
        }

        func callAsFunction(_ pixelValues: MLXArray, frames: [THW]) -> MLXArray {
            var hidden = embeddings(pixelValues)
            hidden = preLayrnorm(hidden)

            let segments = segmented(frames)
            let (cosTable, sinTable) = rotaryEmbedding.cosSin(
                segments: segments, mergeSize: mergeSize)

            var cuSeqlens = [0]
            for segment in segments {
                cuSeqlens.append(cuSeqlens.last! + segment.product)
            }

            for layer in encoder.layers {
                hidden = layer(hidden, cuSeqlens: cuSeqlens, cos: cosTable, sin: sinTable)
            }

            // no post-layernorm, no CLS pooling — per-patch features out
            return hidden
        }
    }

    fileprivate class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: VisionTransformer

        init(_ config: MiniMaxM3VLConfiguration.VisionConfiguration) {
            self._visionModel.wrappedValue = VisionTransformer(config)
            super.init()
        }

        var patchEmbeddingDtype: DType {
            visionModel.embeddings.patchEmbedding.weight.dtype
        }

        func callAsFunction(_ pixelValues: MLXArray, frames: [THW]) -> MLXArray {
            visionModel(pixelValues, frames: frames)
        }
    }
}

// MARK: - Projectors

/// Two-linear MLP with exact GELU (minimax_m3_vl.py `MiniMaxProjector`).
///
/// M3-VL projects PER PATCH first (`multi_modal_projector`, 1280 -> 6144) and
/// only then does the 2x2 spatial merge feed `patch_merge_mlp`
/// (24576 -> 6144) — the opposite order of Qwen2VL's PatchMerger.
private class MiniMaxProjector: Module, UnaryLayer {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputDims: Int, hiddenDims: Int, outputDims: Int, bias: Bool) {
        self._linear1.wrappedValue = Linear(inputDims, hiddenDims, bias: bias)
        self._linear2.wrappedValue = Linear(hiddenDims, outputDims, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(x)))
    }
}

// MARK: - Processor

/// MiniMaxM3VL `UserInputProcessor`.
///
/// This is meant to be used with ``MiniMaxM3VL`` and is typically created by
/// ``VLMModelFactory``.
public struct MiniMaxM3VLProcessor: UserInputProcessor {
    private let config: MiniMaxM3VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    // Special tokens (processing_minimax_m3_vl.py). Ids are resolved from the
    // tokenizer with the reference defaults as fallback.
    static let imageToken = "]<]image[>["
    static let visionStartToken = "]<]start of image[>["
    static let visionEndToken = "]<]end of image[>["

    private var imageTokenId: Int {
        tokenizer.convertTokenToId(Self.imageToken) ?? 200025
    }
    private var visionStartTokenId: Int {
        tokenizer.convertTokenToId(Self.visionStartToken) ?? 200029
    }
    private var visionEndTokenId: Int {
        tokenizer.convertTokenToId(Self.visionEndToken) ?? 200030
    }

    public init(_ config: MiniMaxM3VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// smart_resize (processing_minimax_m3_vl.py:61-110).
    ///
    /// Same core rounding/beta math as `QwenVL.targetSize`, with the M3 clamp:
    /// small images are UPSCALED onto the factor grid (`max(factor, ...)`)
    /// instead of throwing.
    static func targetSize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int)
        throws -> (Int, Int)
    {
        if max(height, width) / min(height, width) > 200 {
            throw VLMError.imageProcessingFailure(
                "Absolute aspect ratio must be smaller than 200: \(width) × \(height)")
        }

        var hBar = max(
            factor, Int((Float(height) / Float(factor)).rounded(.toNearestOrEven)) * factor)
        var wBar = max(
            factor, Int((Float(width) / Float(factor)).rounded(.toNearestOrEven)) * factor)

        if hBar * wBar > maxPixels {
            let beta = sqrt(Float(height * width) / Float(maxPixels))
            hBar = max(factor, Int(floor(Float(height) / beta / Float(factor))) * factor)
            wBar = max(factor, Int(floor(Float(width) / beta / Float(factor))) * factor)
        } else if hBar * wBar < minPixels {
            let beta = sqrt(Float(minPixels) / Float(height * width))
            hBar = Int(ceil(Float(height) * beta / Float(factor))) * factor
            wBar = Int(ceil(Float(width) * beta / Float(factor))) * factor
        }

        if hBar <= 0 || wBar <= 0 {
            throw VLMError.imageProcessingFailure(
                "Invalid target dimensions: \(wBar) × \(hBar)")
        }

        return (hBar, wBar)
    }

    func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
        image
            .toSRGB()
            .resampled(to: resizedSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        // first apply the user requested resizing, etc. if any
        let images = images.map { MediaProcessing.apply($0, processing: processing) }

        let size = images[0].extent.size
        let (resizedHeight, resizedWidth) = try Self.targetSize(
            height: Int(size.height), width: Int(size.width),
            factor: config.patchSize * config.mergeSize,
            minPixels: config.minPixels, maxPixels: config.maxPixels)
        let resizedSize = CGSize(width: resizedWidth, height: resizedHeight)

        let processedImages = images.map { image in
            preprocess(image: image, resizedSize: resizedSize).asMLXArray()
        }

        // ordering verified identical to the reference `_patchify`
        return try QwenVL.patchify(
            images: processedImages, mergeSize: config.mergeSize, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }

    /// Id-level splicing: the chat template emits ONE bare image token per
    /// image; each becomes `start + imageToken × (t·h·w / mergeSize²) + end`
    /// (processing_minimax_m3_vl.py `replace_image_token`). No string
    /// round-trips — the `]<]...[>[` token strings do not re-encode reliably.
    private func expandImageTokens(in promptTokens: [Int], frames: [THW]) throws -> [Int] {
        let imageTokenId = self.imageTokenId
        let placeholderCount = promptTokens.filter { $0 == imageTokenId }.count
        guard placeholderCount == frames.count else {
            throw VLMError.processing(
                "Number of image tokens (\(placeholderCount)) does not match number of images (\(frames.count))"
            )
        }

        let mergeLength = config.mergeSize * config.mergeSize
        var result = [Int]()
        result.reserveCapacity(
            promptTokens.count + frames.reduce(0) { $0 + $1.product / mergeLength + 2 })

        var frameIndex = 0
        for token in promptTokens {
            if token == imageTokenId {
                let count = frames[frameIndex].product / mergeLength
                frameIndex += 1
                result.append(visionStartTokenId)
                result.append(contentsOf: Array(repeating: imageTokenId, count: count))
                result.append(visionEndTokenId)
            } else {
                result.append(token)
            }
        }
        return result
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = MiniMaxM3VLMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        if !input.videos.isEmpty {
            throw VLMError.processing(
                "Video input is not yet supported for minimax_m3_vl")
        }

        // Text-only input
        if input.images.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        let imagePixelsAndFrames = try input.images.map {
            try preprocess(images: [$0.asCIImage()], processing: input.processing)
        }
        let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
        let imageFrames = imagePixelsAndFrames.map { $0.1 }
        let processedImage = LMInput.ProcessedImage(
            pixels: imagePixelsConcatenated, frames: imageFrames)

        promptTokens = try expandImageTokens(in: promptTokens, frames: imageFrames)

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage)
    }
}

// MARK: - Model

/// MiniMaxM3VL VLM
///
/// This is typically created by ``VLMModelFactory``.
public class MiniMaxM3VL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionTower: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: MiniMaxM3Model
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector: MiniMaxProjector
    @ModuleInfo(key: "patch_merge_mlp") private var patchMergeMLP: MiniMaxProjector

    public let config: MiniMaxM3VLConfiguration

    public var vocabularySize: Int { languageModel.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.loraLayers
    }

    public init(_ config: MiniMaxM3VLConfiguration) {
        precondition(
            config.visionFeatureLayer == -1 && config.visionFeatureSelectStrategy == "full",
            "minimax_m3_vl port implements only vision_feature_layer == -1 with the \"full\" select strategy"
        )
        precondition(
            config.projectorHiddenAct == "gelu",
            "minimax_m3_vl port implements only the \"gelu\" projector activation")
        precondition(
            config.visionConfiguration.hiddenAct == "gelu",
            "minimax_m3_vl port implements only the \"gelu\" vision activation")

        self.config = config

        let textHiddenSize = config.textOverview.hiddenSize
        let mergeSize = config.visionConfiguration.spatialMergeSize

        self._visionTower.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = MiniMaxM3Model(config.textConfiguration)
        self._multiModalProjector.wrappedValue = MiniMaxProjector(
            inputDims: config.visionConfiguration.hiddenSize,
            hiddenDims: config.projectorHiddenSize,
            outputDims: textHiddenSize,
            bias: config.multimodalProjectorBias)
        self._patchMergeMLP.wrappedValue = MiniMaxProjector(
            inputDims: textHiddenSize * mergeSize * mergeSize,
            hiddenDims: textHiddenSize,
            outputDims: textHiddenSize,
            bias: config.patchMergeBias)

        super.init()
    }

    /// 2x2 spatial merge AFTER the per-patch projection
    /// (minimax_m3_vl.py `_merge_visual_tokens`).
    private func mergeVisualTokens(_ features: MLXArray, frames: [THW]) -> MLXArray {
        let mergeSize = config.visionConfiguration.spatialMergeSize
        let mergeLength = mergeSize * mergeSize
        let featureDim = features.dim(-1)

        var outputs = [MLXArray]()
        var offset = 0
        for frame in frames {
            let (t, h, w) = frame.values
            let length = t * h * w
            let slice = features[offset ..< (offset + length)]
            offset += length

            // patch order is (t, h/m, w/m, m, m) so grouping each merge block
            // of m*m patches is a plain reshape — reference reshapes to
            // (t, h/m, w/m, m, m, D) then (-1, m*m*D), which composes to this
            let merged = slice.reshaped(length / mergeLength, mergeLength * featureDim)
            outputs.append(patchMergeMLP(merged))
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, frames: [THW]?)
        -> MLXArray
    {
        guard let pixelValues, let frames else {
            return languageModel.model.embedTokens(inputIds[.newAxis, .ellipsis])
        }

        let inputEmbeds = languageModel.model.embedTokens(inputIds)

        // per-patch features -> per-patch projection -> 2x2 merge
        var hiddenStates = visionTower(pixelValues, frames: frames)
        hiddenStates = multiModalProjector(hiddenStates)
        let imageFeatures = mergeVisualTokens(hiddenStates, frames: frames)

        // same hard check as the Python scatter (minimax_m3_vl.py:320-325)
        let imageTokenCount = inputIds.asArray(Int.self).filter {
            $0 == config.imageTokenIndex || $0 == config.videoTokenIndex
        }.count
        precondition(
            imageTokenCount == imageFeatures.dim(0),
            "Image features and image tokens do not match: tokens: \(imageTokenCount), features: \(imageFeatures.dim(0))"
        )

        return QwenVL.mergeInputIdsWithImageFeatures(
            inputIds: inputIds, inputEmbeds: inputEmbeds, imageFeatures: imageFeatures,
            imageTokenId: config.imageTokenIndex,
            videoTokenId: config.videoTokenIndex)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        if input.video != nil {
            throw VLMError.processing(
                "Video input is not yet supported for minimax_m3_vl")
        }

        let dtype = visionTower.patchEmbeddingDtype

        var allPixels: MLXArray?
        var allFrames: [THW] = []
        if let imagePixels = input.image?.pixels, let imageFrames = input.image?.frames {
            allPixels = imagePixels.asType(dtype)
            allFrames.append(contentsOf: imageFrames)
        }

        let inputEmbeddings = self.inputEmbeddings(
            inputIds: input.text.tokens, pixelValues: allPixels,
            frames: allFrames.isEmpty ? nil : allFrames)

        let result = languageModel(nil, cache: cache, inputEmbedding: inputEmbeddings)

        return .logits(LMOutput(logits: result))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = [String: MLXArray]()

        for (rawKey, value) in weights {
            var key = rawKey

            // defensive: some conversions keep an outer `model.` prefix
            // (minimax_m3_vl.py:391-402); a no-op for the mlx-community build
            for prefix in [
                "model.language_model.", "model.vision_tower.",
                "model.multi_modal_projector.", "model.patch_merge_mlp.",
            ] {
                if key.hasPrefix(prefix) {
                    key.removeFirst("model.".count)
                    break
                }
            }

            // no MSA in the Swift trunk; MTP heads unused
            if key.contains(".self_attn.index_") || key.hasPrefix("model.mtp")
                || key.hasPrefix("language_model.model.mtp")
            {
                continue
            }

            key = key.replacingOccurrences(
                of: ".block_sparse_moe.e_score_correction_bias",
                with: ".block_sparse_moe.gate.e_score_correction_bias"
            )

            sanitizedWeights[key] = value
        }

        let routedExperts = config.textOverview.numLocalExperts
        let textHiddenSize = config.textOverview.hiddenSize

        // Unfuse switch_mlp.gate_up_proj and unpack the shared expert: the
        // checkpoint fuses gate+up along the output axis and packs the shared
        // expert as expert index `routedExperts` (mlx-vlm
        // `_sanitize_moe_weights` pack_shared path). Output-axis splits are
        // exact for quantized tensors — packing and groups run along the
        // input axis.
        for key in Array(sanitizedWeights.keys) {
            guard let range = key.range(of: ".switch_mlp.gate_up_proj.") else { continue }
            let value = sanitizedWeights.removeValue(forKey: key)!
            let moePrefix = String(key[..<range.lowerBound])
            let suffix = String(key[range.upperBound...])

            let hasPackedShared = value.dim(0) == routedExperts + 1
            precondition(
                hasPackedShared || value.dim(0) == routedExperts,
                "Unexpected expert count \(value.dim(0)) in \(key)")
            let outSplit = value.dim(1) / 2

            let routed = hasPackedShared ? value[0 ..< routedExperts] : value
            sanitizedWeights["\(moePrefix).switch_mlp.gate_proj.\(suffix)"] =
                contiguous(routed[0..., 0 ..< outSplit, 0...])
            sanitizedWeights["\(moePrefix).switch_mlp.up_proj.\(suffix)"] =
                contiguous(routed[0..., outSplit..., 0...])

            if hasPackedShared {
                let shared = value[routedExperts]
                sanitizedWeights["\(moePrefix).shared_experts.gate_proj.\(suffix)"] =
                    contiguous(shared[0 ..< outSplit])
                sanitizedWeights["\(moePrefix).shared_experts.up_proj.\(suffix)"] =
                    contiguous(shared[outSplit...])
            }
        }

        // down_proj: expert index `routedExperts` is the shared expert
        for key in Array(sanitizedWeights.keys) {
            guard let range = key.range(of: ".switch_mlp.down_proj.") else { continue }
            let value = sanitizedWeights[key]!
            guard value.dim(0) == routedExperts + 1 else { continue }
            let moePrefix = String(key[..<range.lowerBound])
            let suffix = String(key[range.upperBound...])

            sanitizedWeights[key] = contiguous(value[0 ..< routedExperts])
            sanitizedWeights["\(moePrefix).shared_experts.down_proj.\(suffix)"] =
                contiguous(value[routedExperts])
        }

        // Dequantize the MoE router gate (8-bit gs64 on this checkpoint):
        // MiniMaxM3MoEGate holds a raw weight array, not a Linear, so the
        // loader cannot wrap it — and leftover scales/biases keys would fail
        // update(verify: .all). Gate logits want high precision anyway.
        for key in Array(sanitizedWeights.keys) {
            guard key.hasSuffix(".block_sparse_moe.gate.weight") else { continue }
            let base = String(key.dropLast(".weight".count))
            guard let scales = sanitizedWeights.removeValue(forKey: "\(base).scales") else {
                continue
            }
            let biases = sanitizedWeights.removeValue(forKey: "\(base).biases")
            let weight = sanitizedWeights[key]!

            let groupSize = textHiddenSize / scales.dim(-1)
            let bits = weight.dim(-1) * 32 / textHiddenSize
            sanitizedWeights[key] = dequantized(
                weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
        }

        return sanitizedWeights
    }
}

// MARK: - Configuration

/// Configuration for ``MiniMaxM3VL``
public struct MiniMaxM3VLConfiguration: Codable, Sendable {

    public struct VisionConfiguration: Codable, Sendable {
        public let hiddenSize: Int
        public let numHeads: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let patchSize: Int
        private let _numChannels: Int?
        public var numChannels: Int { _numChannels ?? 3 }
        private let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-5 }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 10_000 }
        private let _hiddenAct: String?
        public var hiddenAct: String { _hiddenAct ?? "gelu" }
        private let _visionSegmentMaxFrames: Int?
        public var visionSegmentMaxFrames: Int { _visionSegmentMaxFrames ?? 4 }
        private let _imgTokenCompression: ImgTokenCompression?
        public var spatialMergeSize: Int { _imgTokenCompression?.spatialMergeSize ?? 2 }
        public var temporalPatchSize: Int { _imgTokenCompression?.temporalPatchSize ?? 2 }

        public struct ImgTokenCompression: Codable, Sendable {
            let spatialMergeSize: Int?
            let temporalPatchSize: Int?

            enum CodingKeys: String, CodingKey {
                case spatialMergeSize = "spatial_merge_size"
                case temporalPatchSize = "temporal_patch_size"
            }
        }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numHeads = "num_attention_heads"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case patchSize = "patch_size"
            case _numChannels = "num_channels"
            case _layerNormEps = "layer_norm_eps"
            case _ropeTheta = "rope_theta"
            case _hiddenAct = "hidden_act"
            case _visionSegmentMaxFrames = "vision_segment_max_frames"
            case _imgTokenCompression = "img_token_compression_config"
        }
    }

    /// The handful of `text_config` fields MLXVLM reads directly — the full
    /// trunk configuration is decoded separately as ``MiniMaxM3Configuration``
    /// (whose stored properties are internal to MLXLLM).
    public struct TextOverview: Codable, Sendable {
        private let _hiddenSize: Int?
        public var hiddenSize: Int { _hiddenSize ?? 6144 }
        private let _numLocalExperts: Int?
        public var numLocalExperts: Int { _numLocalExperts ?? 128 }

        enum CodingKeys: String, CodingKey {
            case _hiddenSize = "hidden_size"
            case _numLocalExperts = "num_local_experts"
        }
    }

    public let modelType: String
    public let textConfiguration: MiniMaxM3Configuration
    public let textOverview: TextOverview
    public let visionConfiguration: VisionConfiguration
    public let imageTokenIndex: Int
    public let videoTokenIndex: Int
    public let visionStartTokenId: Int
    public let visionEndTokenId: Int
    public let projectorHiddenSize: Int
    public let projectorHiddenAct: String
    public let multimodalProjectorBias: Bool
    public let patchMergeBias: Bool
    public let visionFeatureLayer: Int
    public let visionFeatureSelectStrategy: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenIndex = "image_token_index"
        case videoTokenIndex = "video_token_index"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
        case projectorHiddenSize = "projector_hidden_size"
        case projectorHiddenAct = "projector_hidden_act"
        case multimodalProjectorBias = "multimodal_projector_bias"
        case patchMergeBias = "patch_merge_bias"
        case visionFeatureLayer = "vision_feature_layer"
        case visionFeatureSelectStrategy = "vision_feature_select_strategy"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "minimax_m3_vl"
        // nested sub-dictionaries — unlike the LLM-side flat decode
        self.textConfiguration = try container.decode(
            MiniMaxM3Configuration.self, forKey: .textConfiguration)
        self.textOverview = try container.decode(TextOverview.self, forKey: .textConfiguration)
        self.visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)

        // defaults per mlx-vlm config.py ModelConfig
        self.imageTokenIndex =
            try container.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 200025
        self.videoTokenIndex =
            try container.decodeIfPresent(Int.self, forKey: .videoTokenIndex) ?? 200026
        self.visionStartTokenId =
            try container.decodeIfPresent(Int.self, forKey: .visionStartTokenId) ?? 200029
        self.visionEndTokenId =
            try container.decodeIfPresent(Int.self, forKey: .visionEndTokenId) ?? 200030
        self.projectorHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 6144
        self.projectorHiddenAct =
            try container.decodeIfPresent(String.self, forKey: .projectorHiddenAct) ?? "gelu"
        self.multimodalProjectorBias =
            try container.decodeIfPresent(Bool.self, forKey: .multimodalProjectorBias) ?? true
        self.patchMergeBias =
            try container.decodeIfPresent(Bool.self, forKey: .patchMergeBias) ?? true
        self.visionFeatureLayer =
            try container.decodeIfPresent(Int.self, forKey: .visionFeatureLayer) ?? -1
        self.visionFeatureSelectStrategy =
            try container.decodeIfPresent(String.self, forKey: .visionFeatureSelectStrategy)
            ?? "full"
    }

    public func encode(to encoder: any Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        // textOverview is a partial re-read of text_config; not encoded separately
        try container.encode(textConfiguration, forKey: .textConfiguration)
        try container.encode(visionConfiguration, forKey: .visionConfiguration)
        try container.encode(imageTokenIndex, forKey: .imageTokenIndex)
        try container.encode(videoTokenIndex, forKey: .videoTokenIndex)
        try container.encode(visionStartTokenId, forKey: .visionStartTokenId)
        try container.encode(visionEndTokenId, forKey: .visionEndTokenId)
        try container.encode(projectorHiddenSize, forKey: .projectorHiddenSize)
        try container.encode(projectorHiddenAct, forKey: .projectorHiddenAct)
        try container.encode(multimodalProjectorBias, forKey: .multimodalProjectorBias)
        try container.encode(patchMergeBias, forKey: .patchMergeBias)
        try container.encode(visionFeatureLayer, forKey: .visionFeatureLayer)
        try container.encode(visionFeatureSelectStrategy, forKey: .visionFeatureSelectStrategy)
    }
}

/// Configuration for ``MiniMaxM3VLProcessor``
public struct MiniMaxM3VLProcessorConfiguration: Codable, Sendable {

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int

    private let _minPixels: Int?
    private let _maxPixels: Int?

    public var minPixels: Int { _minPixels ?? 3136 }
    public var maxPixels: Int { _maxPixels ?? 451_584 }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case _minPixels = "min_pixels"
        case _maxPixels = "max_pixels"
    }
}

/// Message Generator for MiniMaxM3VL.
///
/// The chat template maps `{"type": "image"}` content items to the bare image
/// token (one per image); the processor then expands them at the id level.
public struct MiniMaxM3VLMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        [
            "role": message.role.rawValue,
            "content": [
                ["type": "text", "text": message.content]
            ]
                + message.images.map { _ in
                    ["type": "image"]
                }
                + message.videos.map { _ in
                    ["type": "video"]
                },
        ]
    }
}
