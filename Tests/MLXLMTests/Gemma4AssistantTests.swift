//
//  Gemma4AssistantTests.swift
//  mlx-swift-lm
//

import Foundation
import Testing

@testable import MLXLLM

struct Gemma4AssistantTests {
    @Test("gemma4_assistant config decodes nested text config")
    func testConfigurationDecoding() throws {
        let json =
            """
            {
              "model_type": "gemma4_assistant",
              "backbone_hidden_size": 5376,
              "vocab_size": 262144,
              "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "intermediate_size": 8192,
                "num_attention_heads": 32,
                "head_dim": 256,
                "global_head_dim": 512,
                "num_key_value_heads": 16,
                "num_global_key_value_heads": 4,
                "num_kv_shared_layers": 4,
                "hidden_size_per_layer_input": 0,
                "attention_k_eq_v": true,
                "layer_types": [
                  "sliding_attention",
                  "sliding_attention",
                  "sliding_attention",
                  "full_attention"
                ],
                "rope_parameters": {
                  "sliding_attention": { "rope_theta": 10000 },
                  "full_attention": {
                    "rope_theta": 1000000,
                    "partial_rotary_factor": 0.25
                  }
                }
              }
            }
            """

        let config = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: Data(json.utf8))

        #expect(config.modelType == "gemma4_assistant")
        #expect(config.backboneHiddenSize == 5376)
        #expect(config.textConfig.hiddenSize == 1024)
        #expect(config.textConfig.numHiddenLayers == 4)
        #expect(config.textConfig.layerTypes.last == "full_attention")
    }

    @Test("gemma4_assistant maps draft layers to last target layer of same type")
    func testTargetLayerMapping() throws {
        let config = try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self,
            from: Data(
                """
                {
                  "model_type": "gemma4_assistant",
                  "backbone_hidden_size": 5376,
                  "text_config": {
                    "hidden_size": 1024,
                    "num_hidden_layers": 4,
                    "intermediate_size": 8192,
                    "num_attention_heads": 32,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "num_key_value_heads": 16,
                    "num_global_key_value_heads": 4,
                    "num_kv_shared_layers": 4,
                    "layer_types": [
                      "sliding_attention",
                      "sliding_attention",
                      "sliding_attention",
                      "full_attention"
                    ]
                  }
                }
                """.utf8))

        var targetLayerTypes = [String]()
        for idx in 0 ..< 60 {
            targetLayerTypes.append(idx % 6 == 5 ? "full_attention" : "sliding_attention")
        }

        let mapping = try config.targetLayerIndices(targetLayerTypes: targetLayerTypes)

        #expect(mapping == [58, 58, 58, 59])
    }

    @Test("Gemma4 LLM strips VLM language_model checkpoint prefix")
    func testGemma4LLMSanitizesVLMTextWeights() throws {
        #expect(
            Gemma4Model.sanitizeWeightKey("language_model.model.embed_tokens.weight")
                == "language_model.model.embed_tokens.weight")
        #expect(
            Gemma4Model.sanitizeWeightKey("model.language_model.model.embed_tokens.weight")
                == "language_model.model.embed_tokens.weight")
        #expect(Gemma4Model.sanitizeWeightKey("embed_vision.embedding_projection.weight") == nil)
    }
}
