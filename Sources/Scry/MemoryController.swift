import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - Memory Configuration

/// The output of memory analysis — consumed by GenerationEngine to configure generation.
public struct MemoryConfiguration: Sendable {
    public let kvBits: Int?          // nil = fp16, 4 = quantized
    public let kvGroupSize: Int      // typically 64
    public let maxKVSize: Int?       // context cap, nil = unlimited
    public let prefillStepSize: Int
    public let wiredBaseBytes: Int   // weights + workspace for ticket creation

    public init(
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        maxKVSize: Int? = nil,
        prefillStepSize: Int = 512,
        wiredBaseBytes: Int = 0
    ) {
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.maxKVSize = maxKVSize
        self.prefillStepSize = prefillStepSize
        self.wiredBaseBytes = wiredBaseBytes
    }
}

// MARK: - Memory Measurement

/// Results from a real measurement pass (via WiredMemoryUtils.tune or manual).
public struct MemoryMeasurement: Sendable {
    public let weightBytes: Int
    public let kvBytesPerToken: Double   // KV bytes per token (from measurement at known token count)
    public let workspaceBytes: Int
    public let measurementTokenCount: Int

    /// Estimate KV bytes for a given context length and bit width.
    public func estimateKVBytes(contextLength: Int, bits: Int? = nil) -> Int {
        // Scale linearly from measurement
        let baseBytes = kvBytesPerToken * Double(contextLength)
        guard let bits = bits else { return Int(baseBytes) }
        // If measurement was at fp16 (16 bits) and we want quantized
        return Int(baseBytes * Double(bits) / 16.0)
    }

    public init(weightBytes: Int, kvBytesPerToken: Double, workspaceBytes: Int, measurementTokenCount: Int) {
        self.weightBytes = weightBytes
        self.kvBytesPerToken = kvBytesPerToken
        self.workspaceBytes = workspaceBytes
        self.measurementTokenCount = measurementTokenCount
    }
}

// MARK: - Memory Pressure Level

public enum MemoryPressureLevel: String, Sendable {
    case nominal    // > 2GB headroom
    case warning    // < 2GB headroom
    case critical   // < 500MB headroom
}

// MARK: - Memory Controller

/// Decides memory budgets, KV cache strategy, and monitors pressure at runtime.
public final class MemoryController: Sendable {
    private let profile: HardwareProfile

    public init(profile: HardwareProfile) {
        self.profile = profile
    }

    // MARK: - Static Estimation (Pre-flight)

    /// Estimate whether a model will fit, before loading it.
    public func preFlight(estimatedWeightBytes: Int, desiredContext: Int, metadata: ModelMetadata) -> (fits: Bool, message: String) {
        let kvBytesPerToken = estimateKVBytesPerToken(metadata: metadata, bits: profile.defaultKVBits ?? 16)
        let kvBytes = kvBytesPerToken * desiredContext
        let workspace = Int(Double(estimatedWeightBytes) * 0.12)
        let total = estimatedWeightBytes + kvBytes + workspace
        let available = profile.availableModelMemoryBytes

        if total <= available {
            let headroom = available - total
            return (true, "Fits with \(formatBytes(headroom)) headroom")
        } else {
            let overage = total - available
            return (false, "Exceeds budget by \(formatBytes(overage)). Needs \(formatBytes(total)), have \(formatBytes(available))")
        }
    }

    // MARK: - Runtime Configuration (After Loading)

    /// Configure memory strategy from a real measurement + hardware profile.
    public func configure(measurement: MemoryMeasurement, metadata: ModelMetadata) -> MemoryConfiguration {
        let available = profile.maxRecommendedWorkingSetBytes - measurement.weightBytes - measurement.workspaceBytes

        // Priority candidates: (maxContext, kvBits)
        let candidates: [(Int, Int?)] = candidatesForTier(profile.tier)

        for (maxContext, kvBits) in candidates {
            let kvNeeded = measurement.estimateKVBytes(contextLength: maxContext, bits: kvBits)
            let safeAvailable = Int(Double(available) * 0.9)  // 10% safety margin
            if kvNeeded < safeAvailable {
                return MemoryConfiguration(
                    kvBits: kvBits,
                    kvGroupSize: 64,
                    maxKVSize: maxContext,
                    prefillStepSize: profile.prefillStepSize,
                    wiredBaseBytes: measurement.weightBytes + measurement.workspaceBytes
                )
            }
        }

        // Fallback: minimum viable configuration
        return MemoryConfiguration(
            kvBits: 4,
            kvGroupSize: 64,
            maxKVSize: 2048,
            prefillStepSize: profile.tier == .constrained ? 128 : 256,
            wiredBaseBytes: measurement.weightBytes + measurement.workspaceBytes
        )
    }

    /// Configure from static estimation only (skip tune pass).
    public func configureFromEstimate(metadata: ModelMetadata) -> MemoryConfiguration {
        let estimatedWeight = metadata.estimatedWeightBytes
        let workspace = Int(Double(estimatedWeight) * 0.12)
        let measurement = MemoryMeasurement(
            weightBytes: estimatedWeight,
            kvBytesPerToken: Double(estimateKVBytesPerToken(metadata: metadata, bits: 16)),
            workspaceBytes: workspace,
            measurementTokenCount: 1
        )
        return configure(measurement: measurement, metadata: metadata)
    }

    // MARK: - Pressure Monitoring

    /// Check current memory pressure level.
    public func currentPressureLevel() -> MemoryPressureLevel {
        let active = MLX.GPU.activeMemory
        let maxRecommended = profile.maxRecommendedWorkingSetBytes
        let headroom = maxRecommended - Int(active)

        if headroom < 500 * 1024 * 1024 {
            return .critical
        } else if headroom < 2 * 1024 * 1024 * 1024 {
            return .warning
        }
        return .nominal
    }

    /// Create a pressure monitor task that calls the handler on level changes.
    /// Returns a Task that can be cancelled to stop monitoring.
    public func startPressureMonitor(
        interval: Duration = .seconds(2),
        handler: @escaping @Sendable (MemoryPressureLevel) -> Void
    ) -> Task<Void, Never> {
        Task.detached { [weak self] in
            var lastLevel: MemoryPressureLevel = .nominal
            var nominalCount = 0

            while !Task.isCancelled {
                try? await Task.sleep(for: interval)
                guard let self = self else { break }

                let level = self.currentPressureLevel()

                // Hysteresis: require 3 consecutive nominal readings to transition from warning→nominal
                if lastLevel == .warning && level == .nominal {
                    nominalCount += 1
                    if nominalCount < 3 { continue }
                } else {
                    nominalCount = 0
                }

                if level != lastLevel {
                    lastLevel = level
                    handler(level)
                }
            }
        }
    }

    // MARK: - Private Helpers

    private func candidatesForTier(_ tier: HardwareTier) -> [(Int, Int?)] {
        switch tier {
        case .constrained:
            return [(4096, 4), (2048, 4)]
        case .standard:
            return [(8192, nil), (8192, 4), (4096, nil)]
        case .highEnd:
            return [(32768, nil), (16384, nil)]
        case .server:
            return [(131072, nil), (65536, nil), (32768, nil)]
        }
    }

    private func estimateKVBytesPerToken(metadata: ModelMetadata, bits: Int) -> Int {
        // KV cache per token = 2 (K+V) × numLayers × numKVHeads × headDim × bytesPerElement
        let bytesPerElement = bits == 4 ? 1 : 2  // fp16 = 2 bytes, 4-bit = 0.5 bytes (round up)
        return 2 * metadata.numLayers * metadata.numKVHeads * metadata.headDim * bytesPerElement
    }

    private func formatBytes(_ bytes: Int) -> String {
        let gb = Double(bytes) / 1_073_741_824
        if gb >= 1 {
            return String(format: "%.1f GB", gb)
        }
        return String(format: "%.0f MB", Double(bytes) / 1_048_576)
    }
}
