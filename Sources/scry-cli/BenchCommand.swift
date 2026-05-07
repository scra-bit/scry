import ArgumentParser
import Foundation
import Scry

struct BenchCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench",
        abstract: "Run standardized benchmarks"
    )

    @OptionGroup var modelOpts: ModelOptions

    @Option(name: .long, help: "Number of tokens for prefill benchmark")
    var promptLength: Int = 512

    @Option(name: .long, help: "Number of tokens to generate in decode benchmark")
    var genLength: Int = 256

    @Option(name: .long, help: "Number of runs per benchmark")
    var runs: Int = 3

    func run() async throws {
        let profile = HardwareProfiler.profile()
        let modelID = await resolveModelID(explicit: modelOpts.model, profile: profile)

        let engine = GenerationEngine(profile: profile)

        print("Loading \(modelID) for benchmarking...")
        try await engine.setup(modelID: modelID, progressHandler: makeProgressHandler())
        print()

        let hasMTP = engine.currentMetadata?.mtpVariant != .none

        // Warmup run (JIT shader compilation, cache warming)
        print("Warming up...")
        _ = try await engine.generate(
            prompt: "Hello",
            temperature: 0,
            maxTokens: 10
        )

        // Prefill benchmark
        print("Running prefill benchmark (\(runs) runs)...")
        var prefillResults: [GenerationStats] = []
        let longPrompt = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: promptLength / 10)
        for i in 1...runs {
            let (_, stats) = try await engine.generate(
                prompt: longPrompt,
                temperature: 0,
                maxTokens: 1  // Just measure prefill
            )
            prefillResults.append(stats)
            print("  Run \(i): \(String(format: "%.0f", stats.promptTokensPerSecond)) tok/s prefill")
        }

        // Decode benchmark
        print("Running decode benchmark (\(runs) runs)...")
        var decodeResults: [GenerationStats] = []
        for i in 1...runs {
            let (_, stats) = try await engine.generate(
                prompt: BenchmarkRunner.benchPrompt,
                temperature: 0,
                maxTokens: genLength
            )
            decodeResults.append(stats)
            print("  Run \(i): \(String(format: "%.1f", stats.tokensPerSecond)) tok/s decode")
        }

        // TTFT benchmark
        print("Running TTFT benchmark (\(runs + 2) runs)...")
        var ttftResults: [GenerationStats] = []
        let ttftPrompt = "Explain the concept of recursion in programming."
        for i in 1...(runs + 2) {
            let (_, stats) = try await engine.generate(
                prompt: ttftPrompt,
                temperature: 0,
                maxTokens: 1
            )
            ttftResults.append(stats)
            print("  Run \(i): \(String(format: "%.0f", stats.ttftMilliseconds))ms TTFT")
        }

        // MTP benchmark (if supported)
        var mtpResults: [GenerationStats]? = nil
        if hasMTP {
            print("Running MTP decode benchmark (\(runs) runs)...")
            // MTP would use the same engine but with MTP-aware iteration
            // For now, standard decode serves as the baseline
            // MTP iteration will be plugged in when model-specific MTP is fully wired
        }

        // Format report
        let runner = BenchmarkRunner(profile: profile)
        let weightBytes = engine.weightBytes ?? engine.currentMetadata?.estimatedWeightBytes ?? 0
        let report = runner.formatReport(
            modelID: modelID,
            weightBytes: weightBytes,
            prefillResults: prefillResults,
            decodeResults: decodeResults,
            ttftResults: ttftResults,
            mtpDecodeResults: mtpResults
        )
        print(report)
    }
}
