// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for JSON format: <tag>{"name": "...", "arguments": {...}}</tag>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/default.py
public struct JSONToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let start = startTag, let end = endTag else { return nil }

        // Find the JSON content between tags
        var text = content

        // Strip tags if present
        if let startRange = text.range(of: start) {
            text = String(text[startRange.upperBound...])
        }
        if let endRange = text.range(of: end) {
            text = String(text[..<endRange.lowerBound])
        }

        let jsonStr = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard
            let data = jsonStr.data(using: .utf8),
            let normalizedData = normalizedToolCallData(from: data),
            let function = try? JSONDecoder().decode(ToolCall.Function.self, from: normalizedData)
        else { return nil }

        return ToolCall(function: function)
    }

    private func normalizedToolCallData(from data: Data) -> Data? {
        guard let root = try? JSONSerialization.jsonObject(with: data) else {
            return nil
        }

        var jsonObject: [String: Any]
        if let object = root as? [String: Any] {
            jsonObject = object
        } else if let array = root as? [[String: Any]], let first = array.first {
            jsonObject = first
        } else {
            return nil
        }

        if let toolCalls = jsonObject["tool_calls"] as? [[String: Any]],
           let first = toolCalls.first
        {
            jsonObject = first
        }

        if let function = jsonObject["function"] as? [String: Any] {
            jsonObject = function
        }

        if jsonObject["arguments"] == nil,
           let parameters = jsonObject["parameters"]
        {
            jsonObject["arguments"] = parameters
        }

        if jsonObject["arguments"] == nil {
            jsonObject["arguments"] = [:]
        }

        if let stringifiedArguments = jsonObject["arguments"] as? String {
            guard
                let argumentsData = stringifiedArguments.data(using: .utf8),
                let argumentsObject = try? JSONSerialization.jsonObject(with: argumentsData)
                    as? [String: Any]
            else { return nil }
            jsonObject["arguments"] = argumentsObject
        }

        return try? JSONSerialization.data(withJSONObject: jsonObject)
    }
}
