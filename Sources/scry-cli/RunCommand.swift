import ArgumentParser
import Foundation
import Scry

struct RunCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "One-shot generation from a prompt"
    )

    @OptionGroup var modelOpts: ModelOptions
    @OptionGroup var genOpts: GenerationOptions

    @Option(name: .long, help: "Prompt text")
    var prompt: String

    func run() async throws {
        let profile = HardwareProfiler.profile()
        let modelID = await resolveModelID(explicit: modelOpts.model, profile: profile)

        let engine = GenerationEngine(profile: profile)

        print("Loading \(modelID)...")
        try await engine.setup(
            modelID: modelID,
            systemPrompt: genOpts.system,
            progressHandler: makeProgressHandler()
        )

        if let meta = engine.currentMetadata {
            let mtp = MTPDetector.detect(metadata: meta)
            if mtp.isSupported {
                print("MTP: \(mtp.description)")
            }
        }

        print()

        // Stream the response
        var lastStats: GenerationStats?
        for try await event in engine.chat(
            prompt: prompt,
            temperature: genOpts.temperature,
            maxTokens: genOpts.maxTokens,
            topP: genOpts.topP
        ) {
            switch event {
            case .chunk(let text):
                print(text, terminator: "")
                fflush(stdout)
            case .info(let stats):
                lastStats = stats
            }
        }

        print()
        if let stats = lastStats {
            print()
            print(stats.summary)
        }
    }
}
