import ArgumentParser
import Foundation
import Scry

struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start an OpenAI-compatible HTTP server"
    )

    @OptionGroup var modelOpts: ModelOptions
    @OptionGroup var genOpts: GenerationOptions

    @Option(name: .long, help: "Port to listen on")
    var port: Int = 8080

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

        // Print model info
        if let meta = engine.currentMetadata {
            let mtp = MTPDetector.detect(metadata: meta)
            print("Model: \(meta.modelType) | \(meta.numLayers)L | \(meta.quantizationBits)-bit")
            if mtp.isSupported {
                print("MTP: \(mtp.description)")
            }
            if let wb = engine.weightBytes {
                let theoretical = profile.theoreticalTokPerSec(weightBytes: wb)
                print("Weights: \(String(format: "%.1f", Double(wb) / 1_073_741_824)) GB")
                print("Theoretical max: \(String(format: "%.1f", theoretical)) tok/s")
            }
        }

        print()

        let server = ScryServer(engine: engine, modelID: modelID)
        try await server.start(port: port)
    }
}
