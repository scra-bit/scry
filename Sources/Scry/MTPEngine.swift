import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - MTP Support Detection

/// Detects and reports MTP capability for a loaded model.
public struct MTPDetector: Sendable {

    /// Check if a model supports MTP decoding based on its metadata.
    public static func detect(metadata: ModelMetadata) -> MTPCapability {
        switch metadata.mtpVariant {
        case .qwen:
            return .supported(
                variant: .qwen,
                draftTokens: metadata.mtpNumLayers,  // typically 1
                description: "Qwen MTP (\(metadata.mtpNumLayers) draft layer\(metadata.mtpNumLayers > 1 ? "s" : ""))"
            )
        case .gemma4:
            return .supported(
                variant: .gemma4,
                draftTokens: 1,
                description: "Gemma 4 integrated drafter (KV-shared)"
            )
        case .step:
            return .supported(
                variant: .step,
                draftTokens: metadata.mtpNumLayers,  // typically 3
                description: "Step MTP (\(metadata.mtpNumLayers) prediction layers)"
            )
        case .none:
            return .notSupported
        }
    }
}

// MARK: - MTP Capability

public enum MTPCapability: Sendable {
    case supported(variant: MTPVariant, draftTokens: Int, description: String)
    case notSupported

    public var isSupported: Bool {
        if case .supported = self { return true }
        return false
    }

    public var variant: MTPVariant? {
        if case .supported(let v, _, _) = self { return v }
        return nil
    }

    public var description: String {
        switch self {
        case .supported(_, _, let desc): return desc
        case .notSupported: return "No MTP support"
        }
    }
}

// MARK: - MTP Configuration

/// Configuration for MTP decoding.
public struct MTPConfiguration: Sendable {
    public let variant: MTPVariant
    public let numDraftTokens: Int
    public let useGreedyAcceptance: Bool

    public init(variant: MTPVariant, numDraftTokens: Int = 1, useGreedyAcceptance: Bool = true) {
        self.variant = variant
        self.numDraftTokens = numDraftTokens
        self.useGreedyAcceptance = useGreedyAcceptance
    }

    /// Auto-configure from model metadata.
    public static func auto(from metadata: ModelMetadata) -> MTPConfiguration? {
        let capability = MTPDetector.detect(metadata: metadata)
        guard case .supported(let variant, let draftTokens, _) = capability else {
            return nil
        }
        return MTPConfiguration(
            variant: variant,
            numDraftTokens: draftTokens,
            useGreedyAcceptance: true  // Start greedy, stochastic is Phase 2
        )
    }
}

// MARK: - MTP Round Result

/// Result of a single MTP draft-verify round.
struct MTPRoundResult {
    let acceptedTokens: [Int]      // Tokens confirmed as correct (including the verified base token)
    let correctionToken: Int?       // Token from backbone at rejection point (nil if all accepted)
    let numDrafted: Int
    let numAccepted: Int

    var allTokens: [Int] {
        var result = acceptedTokens
        if let correction = correctionToken {
            result.append(correction)
        }
        return result
    }
}

// MARK: - MTP Engine

/// Manages MTP (Multi-Token Prediction) decoding.
///
/// This engine implements the draft-verify-trim loop for models with built-in
/// prediction heads. It composes with the existing generation infrastructure
/// rather than replacing it.
///
/// For models without built-in MTP, fall back to either:
/// - `SpeculativeTokenIterator` (external draft model, already in mlx-swift-lm)
/// - Standard `TokenIterator` (no speculation)
public final class MTPEngine: @unchecked Sendable {
    private let config: MTPConfiguration
    private var stats = MTPGenerationStats()

    public init(config: MTPConfiguration) {
        self.config = config
    }

    /// Get accumulated stats for this engine's lifetime.
    public var generationStats: MTPGenerationStats { stats }

    /// Reset accumulated stats.
    public func resetStats() {
        stats = MTPGenerationStats()
    }

    // MARK: - Greedy Verification

    /// Verify draft tokens against backbone logits using greedy acceptance.
    ///
    /// - Parameters:
    ///   - draftTokens: Tokens proposed by the MTP head
    ///   - verifyLogits: Backbone logits at each draft position (shape: [numDraft, vocabSize])
    /// - Returns: Number of accepted tokens and correction token
    public func greedyVerify(
        draftTokens: [Int],
        verifyLogits: MLXArray
    ) -> (accepted: Int, correctionToken: Int) {
        let numDraft = draftTokens.count
        var accepted = 0

        for i in 0..<numDraft {
            let logitsAtPosition = verifyLogits[i]
            let backboneChoice = Int(argMax(logitsAtPosition).item(Int32.self))

            if backboneChoice == draftTokens[i] {
                accepted += 1
            } else {
                // Draft rejected at position i — backbone's choice is the correction
                stats.totalDrafted += numDraft
                stats.totalAccepted += accepted
                stats.rounds += 1
                return (accepted, backboneChoice)
            }
        }

        // All drafts accepted — correction is the next backbone prediction after all drafts
        let lastLogits = verifyLogits[numDraft - 1]
        let nextToken = Int(argMax(lastLogits).item(Int32.self))
        stats.totalDrafted += numDraft
        stats.totalAccepted += accepted
        stats.rounds += 1
        return (accepted, nextToken)
    }

    // MARK: - Cache Trimming

    /// Calculate how many tokens to trim from backbone and drafter caches after a verify round.
    ///
    /// The backbone was fed [verified + draft_0 + ... + draft_{n-1}], so after verification
    /// we need to trim (numDraft - accepted) from the backbone cache.
    /// The drafter cache trims differently because the correction token hasn't been fed to it yet.
    public func trimAmounts(numDraft: Int, accepted: Int) -> (backboneTrim: Int, drafterTrim: Int) {
        let backboneTrim = numDraft - accepted
        let drafterTrim = max(numDraft - accepted - 1, 0)
        return (backboneTrim, drafterTrim)
    }

    // MARK: - Decision: Use MTP?

    /// Decide whether to use MTP for a given request context.
    /// MTP should be disabled for concurrent serving (it's a single-sequence optimization).
    public static func shouldUseMTP(
        metadata: ModelMetadata,
        concurrentRequests: Int,
        temperature: Float
    ) -> Bool {
        // No MTP support
        guard metadata.mtpVariant != .none else { return false }

        // MTP doesn't help with concurrent requests (batch decode is better)
        guard concurrentRequests <= 1 else { return false }

        // For now, only greedy acceptance (temperature == 0 or very low)
        // Phase 2: stochastic acceptance for temperature > 0
        // We still enable it at non-zero temp, just with lower expected acceptance rate
        return true
    }
}
