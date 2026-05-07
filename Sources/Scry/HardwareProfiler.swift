import Foundation
import Metal
import MLX

// MARK: - Hardware Tier

/// Classification of the machine's capability tier.
/// Determined once at startup; downstream subsystems branch on this instead of raw numbers.
public enum HardwareTier: String, Sendable, CaseIterable {
    case constrained  // ≤16 GB
    case standard     // 24-48 GB
    case highEnd      // 64-128 GB
    case server       // 192+ GB
}

// MARK: - Hardware Profile

/// Immutable snapshot of the current machine's hardware capabilities.
/// Produced once at startup by `HardwareProfiler.profile()`.
public struct HardwareProfile: Sendable {
    public let chipName: String
    public let chipFamily: ChipFamily
    public let gpuCoreCount: Int
    public let totalMemoryBytes: Int
    public let maxRecommendedWorkingSetBytes: Int
    public let availableModelMemoryBytes: Int
    public let estimatedBandwidthGBps: Double
    public let tier: HardwareTier

    // Derived convenience
    public var totalMemoryGB: Double { Double(totalMemoryBytes) / 1_073_741_824 }
    public var availableModelMemoryGB: Double { Double(availableModelMemoryBytes) / 1_073_741_824 }

    /// Theoretical peak decode tok/s for a model of the given weight size in bytes.
    public func theoreticalTokPerSec(weightBytes: Int) -> Double {
        guard weightBytes > 0 else { return 0 }
        return (estimatedBandwidthGBps * 1_073_741_824) / Double(weightBytes)
    }

    /// Recommended prefill step size for this tier.
    public var prefillStepSize: Int {
        switch tier {
        case .constrained: return 256
        case .standard:    return 512
        case .highEnd:     return 1024
        case .server:      return 2048
        }
    }

    /// Default max context length for this tier.
    public var defaultMaxContext: Int {
        switch tier {
        case .constrained: return 4096
        case .standard:    return 8192
        case .highEnd:     return 32768
        case .server:      return 131072
        }
    }

    /// Default KV quantization bits for this tier (nil = fp16).
    public var defaultKVBits: Int? {
        switch tier {
        case .constrained: return 4
        default:           return nil
        }
    }
}

// MARK: - Chip Family

public enum ChipFamily: String, Sendable {
    case m1, m1Pro, m1Max, m1Ultra
    case m2, m2Pro, m2Max, m2Ultra
    case m3, m3Pro, m3Max, m3Ultra
    case m4, m4Pro, m4Max, m4Ultra
    case unknown
}

// MARK: - Hardware Profiler

public struct HardwareProfiler: Sendable {

    /// Profile the current machine. Call once at startup.
    public static func profile() -> HardwareProfile {
        let chipName = readChipName()
        let chipFamily = parseChipFamily(chipName)
        let totalMemory = Int(ProcessInfo.processInfo.physicalMemory)
        let (gpuCores, maxWorkingSet) = queryMetal()
        let bandwidth = estimatedBandwidth(for: chipFamily)
        let osHeadroom = computeOSHeadroom(totalMemory: totalMemory)
        let available = min(maxWorkingSet, totalMemory - osHeadroom)
        let tier = classifyTier(totalMemory: totalMemory)

        return HardwareProfile(
            chipName: chipName,
            chipFamily: chipFamily,
            gpuCoreCount: gpuCores,
            totalMemoryBytes: totalMemory,
            maxRecommendedWorkingSetBytes: maxWorkingSet,
            availableModelMemoryBytes: max(available, 0),
            estimatedBandwidthGBps: bandwidth,
            tier: tier
        )
    }

    // MARK: - Chip Name

    private static func readChipName() -> String {
        var size: Int = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        guard size > 0 else { return "Unknown" }
        var name = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &name, &size, nil, 0)
        return String(cString: name)
    }

    // MARK: - Chip Family Parsing

    static func parseChipFamily(_ name: String) -> ChipFamily {
        let lower = name.lowercased()
        guard lower.contains("apple") else { return .unknown }

        // Order matters: check "ultra" before "max" before "pro" before base.
        let generations: [(prefix: String, base: ChipFamily, pro: ChipFamily, max: ChipFamily, ultra: ChipFamily)] = [
            ("m4", .m4, .m4Pro, .m4Max, .m4Ultra),
            ("m3", .m3, .m3Pro, .m3Max, .m3Ultra),
            ("m2", .m2, .m2Pro, .m2Max, .m2Ultra),
            ("m1", .m1, .m1Pro, .m1Max, .m1Ultra),
        ]

        for gen in generations {
            if lower.contains(gen.prefix) {
                if lower.contains("ultra") { return gen.ultra }
                if lower.contains("max")   { return gen.max }
                if lower.contains("pro")   { return gen.pro }
                return gen.base
            }
        }
        return .unknown
    }

    // MARK: - Metal Query

    private static func queryMetal() -> (gpuCores: Int, maxWorkingSet: Int) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return (0, 0)
        }
        // GPU core count isn't directly exposed by Metal; approximate from maxThreadgroupMemoryLength
        // or use a rough heuristic. For now, return 0 — it's advisory only.
        let maxWorkingSet = Int(device.recommendedMaxWorkingSetSize)
        return (0, maxWorkingSet)
    }

    // MARK: - Bandwidth Table

    /// Memory bandwidth in GB/s, from Apple spec sheets.
    static func estimatedBandwidth(for family: ChipFamily) -> Double {
        switch family {
        // M1
        case .m1:      return 68.25
        case .m1Pro:   return 200
        case .m1Max:   return 400
        case .m1Ultra: return 800
        // M2
        case .m2:      return 100
        case .m2Pro:   return 200
        case .m2Max:   return 400
        case .m2Ultra: return 800
        // M3
        case .m3:      return 100
        case .m3Pro:   return 150
        case .m3Max:   return 400
        case .m3Ultra: return 800
        // M4
        case .m4:      return 120
        case .m4Pro:   return 273
        case .m4Max:   return 546
        case .m4Ultra: return 819
        // Unknown — conservative estimate
        case .unknown: return 100
        }
    }

    // MARK: - OS Headroom

    /// Estimate how much memory the OS and background processes use.
    /// ~25% of the first 16GB, tapering off, capped at 8GB.
    static func computeOSHeadroom(totalMemory: Int) -> Int {
        let headroom = min(
            Int(Double(totalMemory) * 0.25),
            8 * 1024 * 1024 * 1024
        )
        return headroom
    }

    // MARK: - Tier Classification

    static func classifyTier(totalMemory: Int) -> HardwareTier {
        let gb = totalMemory / (1024 * 1024 * 1024)
        switch gb {
        case ..<20:   return .constrained
        case 20..<56: return .standard
        case 56..<160: return .highEnd
        default:      return .server
        }
    }
}
