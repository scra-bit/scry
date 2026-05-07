import ArgumentParser
import Foundation
import Scry

struct PullCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download a model without loading it"
    )

    @Argument(help: "Model ID (e.g., mlx-community/Llama-3.1-8B-Instruct-4bit)")
    var modelID: String

    func run() async throws {
        let profile = HardwareProfiler.profile()
        let manager = ModelManager(profile: profile)

        // Pre-flight check
        let rec = await manager.recommendModel()
        print("Downloading \(modelID)...")
        print("(Recommended for your \(String(format: "%.0f", profile.totalMemoryGB))GB \(profile.chipName): \(rec.modelID))")
        print()

        // Load (which downloads) then immediately unload
        let loaded = try await manager.load(modelID, progressHandler: makeProgressHandler())
        print()
        print("Downloaded successfully.")
        print("  Model type: \(loaded.metadata.modelType)")
        print("  Layers: \(loaded.metadata.numLayers)")
        print("  Quantization: \(loaded.metadata.quantizationBits)-bit")
        print("  Est. weight size: \(String(format: "%.1f", Double(loaded.metadata.estimatedWeightBytes) / 1_073_741_824)) GB")

        if loaded.metadata.mtpVariant != .none {
            let mtp = MTPDetector.detect(metadata: loaded.metadata)
            print("  MTP: \(mtp.description)")
        }

        await manager.unload()
    }
}

struct ListCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List cached models"
    )

    func run() async throws {
        let profile = HardwareProfiler.profile()
        let manager = ModelManager(profile: profile)
        let cached = await manager.listCachedModels()

        if cached.isEmpty {
            print("No models cached. Use 'scry pull <model-id>' to download one.")
            print()
            let rec = await manager.recommendModel()
            print("Recommended for your hardware: \(rec.modelID)")
            print("  \(rec.reason)")
        } else {
            print("Cached models:")
            for model in cached {
                print("  \(model)")
            }
        }
    }
}
