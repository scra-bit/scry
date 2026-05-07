import ArgumentParser
import Foundation
import Scry

struct ChatCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "chat",
        abstract: "Interactive chat with KV cache persistence"
    )

    @OptionGroup var modelOpts: ModelOptions
    @OptionGroup var genOpts: GenerationOptions

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
            let memConfig = engine.currentMemoryConfig
            print("Model: \(meta.modelType) | \(meta.numLayers)L | \(meta.quantizationBits)-bit")
            if mtp.isSupported {
                print("MTP: \(mtp.description)")
            }
            if let config = memConfig {
                let kvDesc = config.kvBits.map { "\($0)-bit" } ?? "fp16"
                let ctxDesc = config.maxKVSize.map { "\($0)" } ?? "unlimited"
                print("Memory: KV=\(kvDesc), maxContext=\(ctxDesc), prefillStep=\(config.prefillStepSize)")
            }
        }

        print()
        print("Type /help for commands, /quit to exit")
        print()

        var temperature = genOpts.temperature
        var maxTokens = genOpts.maxTokens

        // REPL loop
        while true {
            print("> ", terminator: "")
            fflush(stdout)

            guard let input = readLine(strippingNewline: true) else {
                break  // EOF
            }

            let trimmed = input.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }

            // Handle slash commands
            if trimmed.hasPrefix("/") {
                let handled = await handleCommand(
                    trimmed, engine: engine,
                    temperature: &temperature, maxTokens: &maxTokens
                )
                if !handled { break }  // /quit
                continue
            }

            // Generate response
            print()
            var lastStats: GenerationStats?

            do {
                for try await event in engine.chat(
                    prompt: trimmed,
                    temperature: temperature,
                    maxTokens: maxTokens,
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
            } catch {
                print("\nError: \(error.localizedDescription)")
            }

            print()
            if let stats = lastStats {
                print(stats.summary)
            }
            print()
        }
    }

    /// Handle a slash command. Returns false if the REPL should exit.
    private func handleCommand(
        _ command: String,
        engine: GenerationEngine,
        temperature: inout Float,
        maxTokens: inout Int
    ) async -> Bool {
        let parts = command.split(separator: " ", maxSplits: 1)
        let cmd = String(parts[0]).lowercased()
        let arg = parts.count > 1 ? String(parts[1]) : nil

        switch cmd {
        case "/quit", "/exit", "/q":
            return false

        case "/help", "/h":
            print("""
            Commands:
              /help              Show this help
              /quit              Exit chat
              /reset             Clear conversation history and KV cache
              /stats             Show model and memory info
              /temperature <N>   Set temperature (current: \(temperature))
              /maxtokens <N>     Set max tokens (current: \(maxTokens))
            """)

        case "/reset":
            await engine.clearSession()
            print("Conversation cleared.")

        case "/stats":
            let profile = engine.profile
            print("Hardware: \(profile.chipName) | \(String(format: "%.0f", profile.totalMemoryGB)) GB | \(profile.tier.rawValue)")
            print("Bandwidth: \(String(format: "%.0f", profile.estimatedBandwidthGBps)) GB/s")
            if let wb = engine.weightBytes {
                let theoretical = profile.theoreticalTokPerSec(weightBytes: wb)
                print("Weights: \(String(format: "%.1f", Double(wb) / 1_073_741_824)) GB")
                print("Theoretical max: \(String(format: "%.1f", theoretical)) tok/s")
            }

        case "/temperature", "/temp":
            if let arg = arg, let val = Float(arg) {
                temperature = val
                print("Temperature set to \(val)")
            } else {
                print("Current temperature: \(temperature)")
            }

        case "/maxtokens":
            if let arg = arg, let val = Int(arg) {
                maxTokens = val
                print("Max tokens set to \(val)")
            } else {
                print("Current max tokens: \(maxTokens)")
            }

        default:
            print("Unknown command: \(cmd). Type /help for available commands.")
        }

        return true
    }
}
