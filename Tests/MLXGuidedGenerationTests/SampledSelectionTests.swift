// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXGuidedGeneration

/// Optional sampler in ``GuidedGenerationLoop/applyMaskAndSample``: `nil`
/// keeps greedy selection; a supplied `LogitSampler` selects from the
/// already-masked, already-biased logits.
@Suite
struct SampledSelectionTests {

    private func maskArray(allowing allowed: [Int], vocab: Int = 256) -> MLXArray {
        var words = [UInt32](repeating: 0, count: vocab / 32)
        for token in allowed { words[token / 32] |= (UInt32(1) << (token % 32)) }
        return words.withUnsafeBufferPointer {
            GuidedGenerationLoop.bitmaskToMLXArray($0.baseAddress!, maskBitCount: vocab, totalCount: vocab)
        }
    }

    private func logits(_ values: [Int: Float], vocab: Int = 256) -> MLXArray {
        var floats = [Float](repeating: 0, count: vocab)
        for (token, value) in values { floats[token] = value }
        return MLXArray(floats)[.newAxis, .newAxis, 0...]
    }

    private let qwenNonThinking = { (seed: UInt64) in
        TopPSampler(temperature: 0.7, topP: 0.8, topK: 20, minP: 0, seed: seed)
    }

    @Test
    func nilSamplerIsGreedyAndMatchesArgMaxSampler() {
        let raw = logits([7: 3, 9: 5, 200: 50])
        let mask = maskArray(allowing: [7, 9])
        let greedy = GuidedGenerationLoop.applyMaskAndSample(logits: raw, maskArray: mask)
        let explicit = GuidedGenerationLoop.applyMaskAndSample(logits: raw, maskArray: mask, sampler: ArgMaxSampler())
        #expect(greedy == 9)
        #expect(explicit == greedy)
    }

    @Test
    func samplerNeverSelectsAGrammarDisallowedToken() {
        // Token 200 dominates the raw logits but is masked out.
        let raw = logits([10: 1.0, 11: 0.9, 12: 0.8, 200: 100])
        let mask = maskArray(allowing: [10, 11, 12])
        let sampler = TopPSampler(temperature: 1.0, seed: 3)
        for _ in 0 ..< 300 {
            let token = GuidedGenerationLoop.applyMaskAndSample(logits: raw, maskArray: mask, sampler: sampler)
            #expect([10, 11, 12].contains(Int(token)))
        }
    }

    @Test
    func samplerCanSelectANonArgMaxAllowedToken() {
        let raw = logits([10: 1.0, 11: 0.9])
        let mask = maskArray(allowing: [10, 11])
        let sampler = qwenNonThinking(11)
        let drawn = Set((0 ..< 200).map { _ in
            GuidedGenerationLoop.applyMaskAndSample(logits: raw, maskArray: mask, sampler: sampler)
        })
        #expect(drawn == [10, 11])
    }

    @Test
    func samplerSeesTheClosingBias() {
        // Hard-zone style bias leaves only token 5 viable among allowed ones.
        let raw = logits([4: 2, 5: 0, 6: 2])
        var bias = [Float](repeating: -10_000, count: 256)
        bias[5] = 0
        let sampler = TopPSampler(temperature: 1.0, seed: 5)
        for _ in 0 ..< 100 {
            let token = GuidedGenerationLoop.applyMaskAndSample(
                logits: raw, maskArray: maskArray(allowing: [4, 5, 6]),
                closingBias: MLXArray(bias), sampler: sampler)
            #expect(token == 5)
        }
    }

    // MARK: - Full constrained decode without a model

    private func byteTokenizer() throws -> GrammarTokenizer {
        try GrammarTokenizer(
            vocab: (0 ..< 256).map { String(format: "<0x%02X>", $0) },
            vocabType: .byteFallback,
            eosTokenId: 255)
    }

    /// Drives the constraint exactly as the loop does (mask → select →
    /// stop on an allowed EOS → commit, appending fast-forward tokens) over
    /// flat logits, so only the grammar and the sampler decide the output.
    private func decode(schema: String, sampler: (any LogitSampler)?) throws -> String {
        let constraint = try GrammarConstraint(tokenizer: try byteTokenizer(), jsonSchema: schema)
        let flat = logits([:])
        var bytes: [UInt8] = []
        for _ in 0 ..< 200 {
            let mask = try constraint.computeMask()
            if mask.isTerminated { break }
            let maskArray = GuidedGenerationLoop.buildMaskArray(for: mask, vocabSize: 256, logitDim: 256)
            let token = GuidedGenerationLoop.applyMaskAndSample(logits: flat, maskArray: maskArray, sampler: sampler)
            if mask.needsApply && token == 255 { break }
            let commit = try constraint.commitToken(Int32(token))
            bytes.append(UInt8(token))
            bytes.append(contentsOf: commit.tokens.map { UInt8($0) })
            if commit.isTerminated { break }
        }
        return String(decoding: bytes.filter { $0 != 255 }, as: UTF8.self)
    }

    private let enumSchema = #"{"type": "string", "enum": ["alpha", "beta", "gamma", "delta"]}"#
    private let allowed: Set<String> = [#""alpha""#, #""beta""#, #""gamma""#, #""delta""#]

    @Test
    func greedyConstrainedDecodeIsDeterministicAndValid() throws {
        let first = try decode(schema: enumSchema, sampler: nil)
        #expect(allowed.contains(first))
        #expect(try decode(schema: enumSchema, sampler: nil) == first)
        #expect(try decode(schema: enumSchema, sampler: ArgMaxSampler()) == first)
    }

    @Test
    func sampledConstrainedDecodeIsValidReproducibleAndSeedSensitive() throws {
        var outputs: [UInt64: String] = [:]
        for seed in UInt64(1) ... 12 {
            let output = try decode(schema: enumSchema, sampler: TopPSampler(temperature: 1.0, seed: seed))
            #expect(allowed.contains(output), "seed \(seed) produced \(output)")
            #expect(try decode(schema: enumSchema, sampler: TopPSampler(temperature: 1.0, seed: seed)) == output)
            outputs[seed] = output
        }
        #expect(Set(outputs.values).count > 1, "different seeds should reach different allowed values")
    }

    @Test
    func sampledObjectDecodeSatisfiesTheSchema() throws {
        let schema = #"{"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": false}"#
        for seed in UInt64(1) ... 6 {
            let output = try decode(schema: schema, sampler: qwenNonThinking(seed))
            let object = try JSONSerialization.jsonObject(with: Data(output.utf8)) as? [String: Any]
            #expect(object?.keys.sorted() == ["ok"], "seed \(seed) produced \(output)")
            #expect(object?["ok"] is Bool)
        }
    }
}
