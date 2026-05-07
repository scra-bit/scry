import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - Generation Event

/// Events emitted during generation.
public enum GenerationEvent: Sendable {
    case chunk(String)
    case info(GenerationStats)
}

// MARK: - Generation Engine

/// Central coordinator: ties model, memory, and output together.
/// Owns a ChatSession for multi-turn KV cache persistence.
///
/// Not an actor — relies on ModelContainer's internal serialization.
/// Must be used from a single task at a time (the CLI or HTTP handler ensures this).
public final class GenerationEngine: @unchecked Sendable {
    public let profile: HardwareProfile
    public let modelManager: ModelManager
    public let memoryController: MemoryController

    private var session: ChatSession?
    private var memoryConfig: MemoryConfiguration?
    private var loadedModelID: String?
    private var loadedMetadata: ModelMetadata?
    private var measuredWeightBytes: Int?

    public init(profile: HardwareProfile) {
        self.profile = profile
        self.modelManager = ModelManager(profile: profile)
        self.memoryController = MemoryController(profile: profile)
    }

    // MARK: - Setup

    /// Load a model and configure memory. Call once before generating.
    public func setup(
        modelID: String,
        systemPrompt: String? = nil,
        progressHandler: (@Sendable (Double) -> Void)? = nil
    ) async throws {
        let loaded = try await modelManager.load(modelID, progressHandler: progressHandler)

        // Measure actual memory usage via a synthetic pass
        let measurement = try await measureMemory(container: loaded.container)
        measuredWeightBytes = measurement.weightBytes

        // Configure memory strategy
        memoryConfig = memoryController.configure(
            measurement: measurement,
            metadata: loaded.metadata
        )

        // Create ChatSession with configured parameters
        var params = GenerateParameters()
        if let config = memoryConfig {
            if let kvBits = config.kvBits {
                params.kvBits = kvBits
            }
            params.kvGroupSize = config.kvGroupSize
            if let maxKV = config.maxKVSize {
                params.maxKVSize = maxKV
            }
            params.prefillStepSize = config.prefillStepSize
        }

        session = ChatSession(
            loaded.container,
            instructions: systemPrompt,
            generateParameters: params
        )

        loadedModelID = modelID
        loadedMetadata = loaded.metadata
    }

    /// Setup with minimal overhead (skip tune pass). Uses static estimation.
    public func setupFast(
        modelID: String,
        systemPrompt: String? = nil,
        progressHandler: (@Sendable (Double) -> Void)? = nil
    ) async throws {
        let loaded = try await modelManager.load(modelID, progressHandler: progressHandler)

        memoryConfig = memoryController.configureFromEstimate(metadata: loaded.metadata)
        measuredWeightBytes = loaded.metadata.estimatedWeightBytes

        var params = GenerateParameters()
        if let config = memoryConfig {
            if let kvBits = config.kvBits {
                params.kvBits = kvBits
            }
            params.kvGroupSize = config.kvGroupSize
            if let maxKV = config.maxKVSize {
                params.maxKVSize = maxKV
            }
            params.prefillStepSize = config.prefillStepSize
        }

        session = ChatSession(
            loaded.container,
            instructions: systemPrompt,
            generateParameters: params
        )

        loadedModelID = modelID
        loadedMetadata = loaded.metadata
    }

    // MARK: - Generation

    /// Generate a streaming response to a prompt. Preserves KV cache across calls.
    public func chat(
        prompt: String,
        temperature: Float = 0.6,
        maxTokens: Int = 4096,
        topP: Float = 0.9
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        guard let session = session else {
            return AsyncThrowingStream { $0.finish(throwing: ModelManagerError.modelNotLoaded) }
        }

        // Update generation parameters
        session.generateParameters.temperature = temperature
        session.generateParameters.topP = topP
        session.generateParameters.maxTokens = maxTokens

        return AsyncThrowingStream { continuation in
            let task = Task {
                var promptTokens = 0
                var genTokens = 0
                var promptTime: Double = 0
                var genTime: Double = 0

                do {
                    for try await event in session.streamDetails(to: prompt, images: [], videos: []) {
                        if Task.isCancelled { break }
                        switch event {
                        case .chunk(let text):
                            continuation.yield(.chunk(text))
                        case .info(let info):
                            promptTokens = info.promptTokenCount
                            genTokens = info.generationTokenCount
                            promptTime = info.promptTime ?? 0
                            genTime = info.generateTime ?? 0
                        case .toolCall:
                            break
                        @unknown default:
                            break
                        }
                    }

                    let stats = GenerationStats(
                        promptTokenCount: promptTokens,
                        generationTokenCount: genTokens,
                        promptTimeSeconds: promptTime,
                        generateTimeSeconds: genTime
                    )
                    continuation.yield(.info(stats))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    /// Single-shot generation (no KV cache persistence).
    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        temperature: Float = 0.6,
        maxTokens: Int = 4096,
        topP: Float = 0.9
    ) async throws -> (text: String, stats: GenerationStats) {
        guard let loaded = await modelManager.loaded else {
            throw ModelManagerError.modelNotLoaded
        }

        var params = GenerateParameters()
        params.temperature = temperature
        params.topP = topP
        params.maxTokens = maxTokens
        if let config = memoryConfig {
            if let kvBits = config.kvBits {
                params.kvBits = kvBits
            }
            params.kvGroupSize = config.kvGroupSize
            if let maxKV = config.maxKVSize {
                params.maxKVSize = maxKV
            }
            params.prefillStepSize = config.prefillStepSize
        }

        let singleSession = ChatSession(
            loaded.container,
            instructions: systemPrompt,
            generateParameters: params
        )

        var fullText = ""
        var promptTokens = 0
        var genTokens = 0
        var promptTime: Double = 0
        var genTime: Double = 0

        for try await event in singleSession.streamDetails(to: prompt, images: [], videos: []) {
            switch event {
            case .chunk(let text):
                fullText += text
            case .info(let info):
                promptTokens = info.promptTokenCount
                genTokens = info.generationTokenCount
                promptTime = info.promptTime ?? 0
                genTime = info.generateTime ?? 0
            case .toolCall:
                break
            @unknown default:
                break
            }
        }

        let stats = GenerationStats(
            promptTokenCount: promptTokens,
            generationTokenCount: genTokens,
            promptTimeSeconds: promptTime,
            generateTimeSeconds: genTime
        )

        return (fullText, stats)
    }

    // MARK: - Session Control

    /// Clear the chat session's KV cache and conversation history.
    public func clearSession() async {
        await session?.clear()
    }

    /// Get the current model ID.
    public var currentModelID: String? { loadedModelID }

    /// Get the current model metadata.
    public var currentMetadata: ModelMetadata? { loadedMetadata }

    /// Get the measured weight bytes (after setup).
    public var weightBytes: Int? { measuredWeightBytes }

    /// Get the current memory configuration.
    public var currentMemoryConfig: MemoryConfiguration? { memoryConfig }

    // MARK: - Memory Measurement

    private func measureMemory(container: ModelContainer) async throws -> MemoryMeasurement {
        // Use the model container to run a synthetic prefill pass and measure actual allocations
        let measureTokenCount = 128

        let result: MemoryMeasurement = try await container.perform { context in
            // Measure weight bytes from actual parameters
            let weightBytes = context.model.parameters()
                .flattenedValues()
                .reduce(0) { $0 + $1.nbytes }

            // Track memory before and after a synthetic forward pass
            let memBefore = MLX.GPU.activeMemory

            // Create a synthetic input
            let inputTokens = MLXArray(Array(repeating: Int32(1), count: measureTokenCount))
            let inputArray = inputTokens.reshaped([1, measureTokenCount])

            // Build LM input
            let lmInput = LMInput(text: .init(tokens: inputArray))

            // Run a forward pass (this also warms Metal shaders)
            let caches = context.model.newCache(parameters: .init())
            let result = context.model(lmInput.text, cache: caches, state: nil)
            eval(result.logits)

            let memAfter = MLX.GPU.activeMemory
            let totalAllocated = Int(memAfter) - Int(memBefore)
            let kvBytes = max(totalAllocated - weightBytes, 0)

            // Workspace is the peak transient allocation minus steady-state
            let workspace = Int(Double(weightBytes) * 0.12)  // ~12% overhead estimate

            return MemoryMeasurement(
                weightBytes: weightBytes,
                kvBytesPerToken: Double(kvBytes) / Double(measureTokenCount),
                workspaceBytes: workspace,
                measurementTokenCount: measureTokenCount
            )
        }

        return result
    }
}
