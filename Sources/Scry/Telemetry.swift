import Foundation
import MLX

// MARK: - Generation Stats

/// Performance statistics for a single generation request.
public struct GenerationStats: Sendable {
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let promptTimeSeconds: Double
    public let generateTimeSeconds: Double
    public let tokensPerSecond: Double
    public let promptTokensPerSecond: Double
    public let ttftMilliseconds: Double
    public let mtpStats: MTPGenerationStats?

    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        promptTimeSeconds: Double,
        generateTimeSeconds: Double,
        mtpStats: MTPGenerationStats? = nil
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTimeSeconds = promptTimeSeconds
        self.generateTimeSeconds = generateTimeSeconds
        self.tokensPerSecond = generateTimeSeconds > 0
            ? Double(generationTokenCount) / generateTimeSeconds : 0
        self.promptTokensPerSecond = promptTimeSeconds > 0
            ? Double(promptTokenCount) / promptTimeSeconds : 0
        self.ttftMilliseconds = promptTimeSeconds * 1000
        self.mtpStats = mtpStats
    }

    /// Human-readable summary line.
    public var summary: String {
        var parts = [
            "\(generationTokenCount) tokens",
            String(format: "%.1f tok/s", tokensPerSecond),
            String(format: "TTFT: %.0fms", ttftMilliseconds),
        ]
        if let mtp = mtpStats, mtp.totalDrafted > 0 {
            parts.append(String(format: "MTP: %.0f%% accepted", mtp.acceptanceRate * 100))
        }
        return "[\(parts.joined(separator: ", "))]"
    }
}

// MARK: - MTP Generation Stats

public struct MTPGenerationStats: Sendable {
    public var totalDrafted: Int = 0
    public var totalAccepted: Int = 0
    public var rounds: Int = 0

    public var acceptanceRate: Double {
        totalDrafted > 0 ? Double(totalAccepted) / Double(totalDrafted) : 0
    }

    public var avgTokensPerRound: Double {
        rounds > 0 ? Double(totalAccepted + rounds) / Double(rounds) : 1
    }

    public init() {}
}

// MARK: - Bandwidth Efficiency

/// Compute bandwidth utilization for a decode run.
public struct BandwidthAnalysis: Sendable {
    public let weightBytes: Int
    public let decodeTokPerSec: Double
    public let hardwareBandwidthGBps: Double
    public let achievedBandwidthGBps: Double
    public let utilization: Double  // 0.0 - 1.0
    public let theoreticalMaxTokPerSec: Double

    public init(weightBytes: Int, decodeTokPerSec: Double, hardwareBandwidthGBps: Double) {
        self.weightBytes = weightBytes
        self.decodeTokPerSec = decodeTokPerSec
        self.hardwareBandwidthGBps = hardwareBandwidthGBps
        self.achievedBandwidthGBps = (Double(weightBytes) * decodeTokPerSec) / 1_073_741_824
        self.theoreticalMaxTokPerSec = (hardwareBandwidthGBps * 1_073_741_824) / Double(weightBytes)
        self.utilization = decodeTokPerSec / self.theoreticalMaxTokPerSec
    }

    public var summary: String {
        return """
        Bandwidth:  \(String(format: "%.0f%%", utilization * 100)) utilization \
        (\(formatBytes(weightBytes)) × \(String(format: "%.1f", decodeTokPerSec)) / \
        \(String(format: "%.0f", hardwareBandwidthGBps)) GB/s)
        Theoretical max: \(String(format: "%.1f", theoreticalMaxTokPerSec)) tok/s
        Achieved:        \(String(format: "%.1f", decodeTokPerSec)) tok/s \
        (\(String(format: "%.0f%%", utilization * 100)) of theoretical)
        """
    }

    private func formatBytes(_ bytes: Int) -> String {
        String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
    }
}

// MARK: - Server Telemetry

/// Accumulates metrics across requests for server health reporting.
public actor ServerTelemetry {
    private var requestsTotal: Int = 0
    private var requestsActive: Int = 0
    private var tokensGeneratedTotal: Int = 0
    private var recentStats: [GenerationStats] = []
    private let maxRecentStats = 100
    private let startTime = Date()

    public init() {}

    /// Record the start of a request.
    public func requestStarted() {
        requestsActive += 1
    }

    /// Record the completion of a request with its stats.
    public func requestCompleted(stats: GenerationStats) {
        requestsTotal += 1
        requestsActive -= 1
        tokensGeneratedTotal += stats.generationTokenCount
        recentStats.append(stats)
        if recentStats.count > maxRecentStats {
            recentStats.removeFirst()
        }
    }

    /// Record request failure.
    public func requestFailed() {
        requestsActive -= 1
    }

    /// Get current health snapshot.
    public func healthSnapshot(modelID: String, memoryUsed: Int, memoryAvailable: Int, mtpEnabled: Bool) -> HealthResponse {
        let avgTokPerSec = recentStats.isEmpty ? 0.0 :
            recentStats.reduce(0.0) { $0 + $1.tokensPerSecond } / Double(recentStats.count)
        let avgTTFT = recentStats.isEmpty ? 0.0 :
            recentStats.reduce(0.0) { $0 + $1.ttftMilliseconds } / Double(recentStats.count)
        let avgMTPAcceptance: Double? = {
            let mtpStats = recentStats.compactMap { $0.mtpStats }
            guard !mtpStats.isEmpty else { return nil }
            return mtpStats.reduce(0.0) { $0 + $1.acceptanceRate } / Double(mtpStats.count)
        }()

        return HealthResponse(
            model: modelID,
            uptimeSeconds: Date().timeIntervalSince(startTime),
            requestsTotal: requestsTotal,
            requestsActive: requestsActive,
            tokensGeneratedTotal: tokensGeneratedTotal,
            avgDecodeTokPerSec: avgTokPerSec,
            avgTTFTMs: avgTTFT,
            memoryUsedBytes: memoryUsed,
            memoryAvailableBytes: memoryAvailable,
            mtpEnabled: mtpEnabled,
            mtpAcceptanceRate: avgMTPAcceptance
        )
    }
}

// MARK: - Health Response

public struct HealthResponse: Codable, Sendable {
    public let model: String
    public let uptimeSeconds: Double
    public let requestsTotal: Int
    public let requestsActive: Int
    public let tokensGeneratedTotal: Int
    public let avgDecodeTokPerSec: Double
    public let avgTTFTMs: Double
    public let memoryUsedBytes: Int
    public let memoryAvailableBytes: Int
    public let mtpEnabled: Bool
    public let mtpAcceptanceRate: Double?

    enum CodingKeys: String, CodingKey {
        case model
        case uptimeSeconds = "uptime_seconds"
        case requestsTotal = "requests_total"
        case requestsActive = "requests_active"
        case tokensGeneratedTotal = "tokens_generated_total"
        case avgDecodeTokPerSec = "avg_decode_tok_per_sec"
        case avgTTFTMs = "avg_ttft_ms"
        case memoryUsedBytes = "memory_used_bytes"
        case memoryAvailableBytes = "memory_available_bytes"
        case mtpEnabled = "mtp_enabled"
        case mtpAcceptanceRate = "mtp_acceptance_rate"
    }
}

// MARK: - Benchmark Runner

/// Runs standardized benchmarks for the `bench` command.
public struct BenchmarkRunner: Sendable {
    public let profile: HardwareProfile

    public init(profile: HardwareProfile) {
        self.profile = profile
    }

    /// The fixed benchmark prompt for reproducible results.
    public static let benchPrompt = """
    Write a detailed explanation of how transformer neural networks work, covering attention \
    mechanisms, positional encoding, layer normalization, and the feed-forward network. Be \
    thorough and technical.
    """

    /// Format a complete benchmark report.
    public func formatReport(
        modelID: String,
        weightBytes: Int,
        prefillResults: [GenerationStats],
        decodeResults: [GenerationStats],
        ttftResults: [GenerationStats],
        mtpDecodeResults: [GenerationStats]?
    ) -> String {
        let prefillMedian = median(prefillResults.map { $0.promptTokensPerSecond })
        let decodeMedian = median(decodeResults.map { $0.tokensPerSecond })
        let ttftMedian = median(ttftResults.map { $0.ttftMilliseconds })

        let bandwidth = BandwidthAnalysis(
            weightBytes: weightBytes,
            decodeTokPerSec: decodeMedian,
            hardwareBandwidthGBps: profile.estimatedBandwidthGBps
        )

        let kvBytes = decodeResults.first.map {
            Int(Double($0.generationTokenCount) * 0.001 * 1_073_741_824)
        } ?? 0
        let totalMemory = weightBytes + kvBytes
        let memPct = profile.totalMemoryBytes > 0
            ? Double(totalMemory) / Double(profile.totalMemoryBytes) * 100 : 0

        var lines: [String] = []
        lines.append("")
        lines.append("scry bench — \(modelID) on \(profile.chipName) (\(String(format: "%.0f", profile.totalMemoryGB)) GB)")
        lines.append("")
        lines.append(String(format: "Prefill:    %.0f tok/s  (median of %d runs)", prefillMedian, prefillResults.count))
        lines.append(String(format: "Decode:     %.1f tok/s (median of %d runs)", decodeMedian, decodeResults.count))
        lines.append(String(format: "TTFT:       %.0f ms     (median of %d runs)", ttftMedian, ttftResults.count))

        if let mtpResults = mtpDecodeResults, !mtpResults.isEmpty {
            let mtpMedian = median(mtpResults.map { $0.tokensPerSecond })
            let speedup = decodeMedian > 0 ? mtpMedian / decodeMedian : 0
            let acceptance = mtpResults.compactMap { $0.mtpStats?.acceptanceRate }
            let avgAcceptance = acceptance.isEmpty ? 0 : acceptance.reduce(0, +) / Double(acceptance.count)
            lines.append("")
            lines.append(String(format: "MTP Decode: %.1f tok/s (%.2fx speedup, %.0f%% acceptance)",
                                mtpMedian, speedup, avgAcceptance * 100))
        }

        lines.append("")
        lines.append(String(format: "Memory:     %@ weights (%.1f%% of %.0f GB)",
                            formatBytes(weightBytes), memPct, profile.totalMemoryGB))
        lines.append(bandwidth.summary)
        lines.append("")

        return lines.joined(separator: "\n")
    }

    private func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 0 {
            return (sorted[mid - 1] + sorted[mid]) / 2
        }
        return sorted[mid]
    }

    private func formatBytes(_ bytes: Int) -> String {
        String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
    }
}
