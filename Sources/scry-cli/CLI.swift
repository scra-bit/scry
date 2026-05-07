import ArgumentParser
import Foundation
import Scry

@main
struct ScryCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "scry",
        abstract: "Local LLM inference on Apple Silicon",
        version: "0.1.0",
        subcommands: [
            RunCommand.self,
            ChatCommand.self,
            PullCommand.self,
            ListCommand.self,
            BenchCommand.self,
            ServeCommand.self,
        ],
        defaultSubcommand: ChatCommand.self
    )
}

// MARK: - Shared Options

struct ModelOptions: ParsableArguments {
    @Option(name: .long, help: "Model ID (HuggingFace ID or local path)")
    var model: String?

    @Option(name: .long, help: "Draft model for speculative decoding")
    var draftModel: String?
}

struct GenerationOptions: ParsableArguments {
    @Option(name: .long, help: "Sampling temperature (0 = greedy)")
    var temperature: Float = 0.6

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 4096

    @Option(name: .long, help: "Top-p (nucleus) sampling")
    var topP: Float = 0.9

    @Option(name: .long, help: "System prompt")
    var system: String?
}

// MARK: - Helpers

/// Resolve model ID: use --model flag, or find cached model, or recommend one.
func resolveModelID(explicit: String?, profile: HardwareProfile) async -> String {
    if let explicit = explicit {
        return explicit
    }

    // Check for cached models
    let manager = ModelManager(profile: profile)
    let cached = await manager.listCachedModels()
    if let first = cached.first {
        print("Using cached model: \(first)")
        return first
    }

    // Recommend based on hardware
    let rec = await manager.recommendModel()
    print("Using recommended model: \(rec.modelID) (\(rec.reason))")
    return rec.modelID
}

/// Print a simple progress bar for model downloads.
func makeProgressHandler() -> @Sendable (Double) -> Void {
    let state = ProgressState()
    return { fraction in
        Task { @MainActor in
            state.update(fraction: fraction)
        }
    }
}

@MainActor
final class ProgressState {
    private var lastPrinted: Double = -1

    func update(fraction: Double) {
        let pct = (fraction * 100).rounded()
        if pct != lastPrinted {
            lastPrinted = pct
            let barWidth = 30
            let filled = Int(fraction * Double(barWidth))
            let bar = String(repeating: "=", count: filled) + String(repeating: " ", count: barWidth - filled)
            print("\rDownloading: [\(bar)] \(Int(pct))%", terminator: "")
            fflush(stdout)
            if pct >= 100 {
                print()  // newline after complete
            }
        }
    }
}
