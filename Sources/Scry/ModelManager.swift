import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - Model Metadata

/// Parsed metadata from config.json that the factory doesn't expose.
public struct ModelMetadata: Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numLayers: Int
    public let numAttentionHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let quantizationBits: Int       // 16 = unquantized, 4/2 = quantized
    public let intermediateSize: Int
    public let mtpNumLayers: Int           // 0 = no MTP
    public let mtpVariant: MTPVariant

    /// Rough parameter count estimate from architecture dimensions.
    public var estimatedParameters: Int {
        let embedding = vocabSize * hiddenSize
        let perLayer = 4 * hiddenSize * hiddenSize       // Q, K, V, O
            + 2 * hiddenSize * intermediateSize           // FFN up + down
            + hiddenSize                                  // norms, biases
        return embedding + numLayers * perLayer
    }

    /// Estimated weight size in bytes.
    public var estimatedWeightBytes: Int {
        return estimatedParameters * quantizationBits / 8
    }
}

// MARK: - MTP Variant Detection

public enum MTPVariant: String, Sendable {
    case none
    case qwen       // mtp_num_hidden_layers > 0
    case gemma4     // assistant_config present
    case step       // num_nextn_predict_layers > 0
}

// MARK: - Loaded Model

/// A model that's been loaded into GPU memory with all associated metadata.
public struct LoadedModel: Sendable {
    public let container: ModelContainer
    public let metadata: ModelMetadata
    public let modelID: String
    public let loadTimeSeconds: Double
}

// MARK: - Model Recommendation

public struct ModelRecommendation: Sendable {
    public let modelID: String
    public let estimatedWeightGB: Double
    public let reason: String
}

// MARK: - Model Manager

/// Manages the full model lifecycle: resolve, download, load, parse metadata, swap.
/// Actor because model loading/unloading must be serialized.
public actor ModelManager {
    private let profile: HardwareProfile
    private var currentModel: LoadedModel?

    public init(profile: HardwareProfile) {
        self.profile = profile
    }

    /// The currently loaded model, if any.
    public var loaded: LoadedModel? { currentModel }

    // MARK: - Load

    /// Load a model by HuggingFace ID or local path.
    /// Downloads if not cached. Returns the loaded model with metadata.
    public func load(
        _ modelID: String,
        progressHandler: (@Sendable (Double) -> Void)? = nil
    ) async throws -> LoadedModel {
        // Unload previous model
        if currentModel != nil {
            unload()
        }

        let start = CFAbsoluteTimeGetCurrent()

        // Resolve configuration
        let configuration: ModelConfiguration
        if modelID.hasPrefix("/") || modelID.hasPrefix("file://") {
            let url = URL(fileURLWithPath: modelID)
            configuration = ModelConfiguration(directory: url)
        } else {
            configuration = ModelConfiguration(id: modelID)
        }

        // Load via factory (handles download, weight loading, tokenizer)
        let container: ModelContainer
        do {
            container = try await LLMModelFactory.shared.loadContainer(
                configuration: configuration
            ) { progress in
                progressHandler?(progress.fractionCompleted)
            }
        } catch {
            // Fallback: try VLM factory for vision-language models
            container = try await VLMModelFactory.shared.loadContainer(
                configuration: configuration
            ) { progress in
                progressHandler?(progress.fractionCompleted)
            }
        }

        // Parse config.json for metadata the factory doesn't expose
        let modelDirectory = configuration.modelDirectory(hub: HubApi())
        let metadata = try parseMetadata(from: modelDirectory)

        let loadTime = CFAbsoluteTimeGetCurrent() - start

        let loaded = LoadedModel(
            container: container,
            metadata: metadata,
            modelID: modelID,
            loadTimeSeconds: loadTime
        )
        currentModel = loaded
        return loaded
    }

    // MARK: - Unload

    /// Unload the current model, releasing GPU memory.
    public func unload() {
        currentModel = nil
        // Release MLX's internal cache to free pages promptly
        MLX.GPU.clearCache()
    }

    // MARK: - Pre-flight Check

    /// Check if a model will likely fit in memory without loading it.
    public func willFit(estimatedWeightBytes: Int) -> (fits: Bool, headroomBytes: Int) {
        let budget = Int(Double(profile.availableModelMemoryBytes) * 0.7)
        let headroom = budget - estimatedWeightBytes
        return (headroom > 0, headroom)
    }

    // MARK: - Recommendation

    /// Recommend a model based on available memory.
    public func recommendModel() -> ModelRecommendation {
        let budgetGB = profile.availableModelMemoryGB * 0.7

        // Prefer MTP-capable models when available
        switch budgetGB {
        case ..<2:
            return ModelRecommendation(
                modelID: "mlx-community/Llama-3.2-1B-Instruct-4bit",
                estimatedWeightGB: 0.7,
                reason: "Best fit for \(String(format: "%.0f", profile.totalMemoryGB))GB — lightweight 1B model"
            )
        case 2..<4:
            return ModelRecommendation(
                modelID: "mlx-community/Llama-3.2-3B-Instruct-4bit",
                estimatedWeightGB: 1.8,
                reason: "Good balance for \(String(format: "%.0f", profile.totalMemoryGB))GB"
            )
        case 4..<7:
            return ModelRecommendation(
                modelID: "mlx-community/Qwen3-8B-4bit",
                estimatedWeightGB: 4.3,
                reason: "MTP-capable 8B model, recommended for \(String(format: "%.0f", profile.totalMemoryGB))GB"
            )
        case 7..<12:
            return ModelRecommendation(
                modelID: "mlx-community/Qwen2.5-14B-Instruct-4bit",
                estimatedWeightGB: 7.5,
                reason: "Strong 14B model for \(String(format: "%.0f", profile.totalMemoryGB))GB"
            )
        case 12..<20:
            return ModelRecommendation(
                modelID: "mlx-community/Qwen2.5-32B-Instruct-4bit",
                estimatedWeightGB: 17,
                reason: "32B model fits comfortably in \(String(format: "%.0f", profile.totalMemoryGB))GB"
            )
        case 20..<40:
            return ModelRecommendation(
                modelID: "mlx-community/Qwen2.5-72B-Instruct-4bit",
                estimatedWeightGB: 35,
                reason: "Full 72B model for \(String(format: "%.0f", profile.totalMemoryGB))GB"
            )
        default:
            return ModelRecommendation(
                modelID: "mlx-community/Llama-3.1-405B-Instruct-4bit",
                estimatedWeightGB: 200,
                reason: "Server-class: 405B model"
            )
        }
    }

    // MARK: - Cache Management

    /// List models cached locally (in HuggingFace hub cache).
    public func listCachedModels() -> [String] {
        let cacheDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: cacheDir, includingPropertiesForKeys: nil
        ) else { return [] }

        return contents.compactMap { url -> String? in
            let configPath = url.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: configPath.path) else { return nil }
            // Convert directory name back to model ID (models--org--name → org/name)
            let name = url.lastPathComponent
            if name.hasPrefix("models--") {
                return name.dropFirst(8).replacingOccurrences(of: "--", with: "/")
            }
            return name
        }
    }

    /// Delete a cached model from disk.
    public func deleteCachedModel(_ modelID: String) throws {
        let safeName = "models--" + modelID.replacingOccurrences(of: "/", with: "--")
        let cacheDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent(safeName)
        try FileManager.default.removeItem(at: cacheDir)
    }

    // MARK: - Metadata Parsing

    /// Parse config.json for metadata the factory doesn't surface.
    private func parseMetadata(from modelDirectory: URL) throws -> ModelMetadata {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ModelManagerError.invalidConfig
        }
        return Self.extractMetadata(from: config)
    }

    /// Extract metadata from a parsed config dictionary.
    /// Public for testing.
    public static func extractMetadata(from config: [String: Any]) -> ModelMetadata {
        let modelType = config["model_type"] as? String ?? "unknown"
        let hiddenSize = config["hidden_size"] as? Int
            ?? config["d_model"] as? Int ?? 4096
        let numLayers = config["num_hidden_layers"] as? Int
            ?? config["n_layer"] as? Int ?? 32
        let numAttentionHeads = config["num_attention_heads"] as? Int
            ?? config["n_head"] as? Int ?? 32
        let numKVHeads = config["num_key_value_heads"] as? Int ?? numAttentionHeads
        let headDim = config["head_dim"] as? Int ?? (hiddenSize / numAttentionHeads)
        let vocabSize = config["vocab_size"] as? Int ?? 32000
        let maxPos = config["max_position_embeddings"] as? Int ?? 4096
        let intermediateSize = config["intermediate_size"] as? Int ?? (hiddenSize * 4)

        // Quantization
        var quantBits = 16
        if let quantConfig = config["quantization_config"] as? [String: Any] {
            quantBits = quantConfig["bits"] as? Int ?? 16
        }

        // MTP detection — three different config patterns
        let mtpVariant: MTPVariant
        let mtpLayers: Int
        if let mtpCount = config["mtp_num_hidden_layers"] as? Int, mtpCount > 0 {
            mtpVariant = .qwen
            mtpLayers = mtpCount
        } else if config["assistant_config"] != nil {
            mtpVariant = .gemma4
            mtpLayers = 1  // Gemma4 drafter is a single block
        } else if let stepCount = config["num_nextn_predict_layers"] as? Int, stepCount > 0 {
            mtpVariant = .step
            mtpLayers = stepCount
        } else {
            mtpVariant = .none
            mtpLayers = 0
        }

        return ModelMetadata(
            modelType: modelType,
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            numAttentionHeads: numAttentionHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            vocabSize: vocabSize,
            maxPositionEmbeddings: maxPos,
            quantizationBits: quantBits,
            intermediateSize: intermediateSize,
            mtpNumLayers: mtpLayers,
            mtpVariant: mtpVariant
        )
    }
}

// MARK: - Errors

public enum ModelManagerError: Error, LocalizedError {
    case invalidConfig
    case modelTooLarge(required: Int, available: Int)
    case modelNotLoaded

    public var errorDescription: String? {
        switch self {
        case .invalidConfig:
            return "Could not parse model config.json"
        case .modelTooLarge(let required, let available):
            let reqGB = String(format: "%.1f", Double(required) / 1_073_741_824)
            let avlGB = String(format: "%.1f", Double(available) / 1_073_741_824)
            return "Model requires ~\(reqGB) GB but only \(avlGB) GB available"
        case .modelNotLoaded:
            return "No model is currently loaded"
        }
    }
}
