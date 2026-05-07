import Foundation
import Hummingbird
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - OpenAI API Types

public struct ChatCompletionRequest: Decodable, Sendable {
    public let model: String?
    public let messages: [ChatMessage]
    public let stream: Bool?
    public let temperature: Float?
    public let maxTokens: Int?
    public let topP: Float?
    public let frequencyPenalty: Float?
    public let presencePenalty: Float?
    public let stop: [String]?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, topP = "top_p"
        case maxTokens = "max_tokens"
        case frequencyPenalty = "frequency_penalty"
        case presencePenalty = "presence_penalty"
        case stop
    }
}

public struct ChatMessage: Codable, Sendable {
    public let role: String
    public let content: String?

    public init(role: String, content: String?) {
        self.role = role
        self.content = content
    }
}

struct ChatCompletionResponse: Encodable, Sendable {
    let id: String
    let object: String = "chat.completion"
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: Usage?

    struct Choice: Encodable, Sendable {
        let index: Int
        let message: ChatMessage
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
    }

    struct Usage: Encodable, Sendable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }
}

struct ChatCompletionChunk: Encodable, Sendable {
    let id: String
    let object: String = "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [ChunkChoice]

    struct ChunkChoice: Encodable, Sendable {
        let index: Int
        let delta: Delta
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index, delta
            case finishReason = "finish_reason"
        }
    }

    struct Delta: Encodable, Sendable {
        let role: String?
        let content: String?
    }
}

struct ModelListResponse: Encodable, Sendable {
    let object: String = "list"
    let data: [ModelInfo]

    struct ModelInfo: Encodable, Sendable {
        let id: String
        let object: String = "model"
        let created: Int
        let ownedBy: String = "local"

        enum CodingKeys: String, CodingKey {
            case id, object, created
            case ownedBy = "owned_by"
        }
    }
}

// MARK: - Request Queue

/// Serializes access to the model — one generation at a time.
actor RequestQueue {
    private var isProcessing = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    var queueDepth: Int { waiting.count }
    var isBusy: Bool { isProcessing }

    func acquire() async {
        if !isProcessing {
            isProcessing = true
            return
        }
        await withCheckedContinuation { waiting.append($0) }
    }

    func release() {
        if let next = waiting.first {
            waiting.removeFirst()
            next.resume()
        } else {
            isProcessing = false
        }
    }
}

// MARK: - Scry Server

/// OpenAI-compatible HTTP server for local model inference.
public final class ScryServer: Sendable {
    let engine: GenerationEngine
    let telemetry: ServerTelemetry
    let queue: RequestQueue
    let modelID: String

    public init(engine: GenerationEngine, modelID: String) {
        self.engine = engine
        self.telemetry = ServerTelemetry()
        self.queue = RequestQueue()
        self.modelID = modelID
    }

    /// Start the server on the given port. Blocks until shutdown.
    public func start(port: Int = 8080) async throws {
        let router = Router()

        // CORS headers for browser clients
        router.middlewares.add(CORSMiddleware())

        // Routes
        router.post("/v1/chat/completions", use: handleChatCompletions)
        router.get("/v1/models", use: handleListModels)
        router.get("/health", use: handleHealth)

        let app = Application(router: router, configuration: .init(address: .hostname("0.0.0.0", port: port)))

        print("Serving \(modelID) on http://localhost:\(port)")
        print("Endpoints:")
        print("  POST /v1/chat/completions")
        print("  GET  /v1/models")
        print("  GET  /health")
        print("")

        try await app.run()
    }

    // MARK: - Handlers

    @Sendable
    func handleChatCompletions(request: Request, context: BasicRequestContext) async throws -> Response {
        let body = try await request.body.collect(upTo: 10 * 1024 * 1024)  // 10MB limit
        let chatRequest = try JSONDecoder().decode(ChatCompletionRequest.self, from: body)

        let requestID = "chatcmpl-\(UUID().uuidString.prefix(12))"
        let created = Int(Date().timeIntervalSince1970)
        let isStream = chatRequest.stream ?? false

        // Build prompt from messages
        let prompt = buildPrompt(from: chatRequest.messages)
        let temperature = chatRequest.temperature ?? 0.7
        let maxTokens = chatRequest.maxTokens ?? 4096
        let topP = chatRequest.topP ?? 0.9

        if isStream {
            return try await handleStreamingResponse(
                requestID: requestID,
                created: created,
                prompt: prompt,
                temperature: temperature,
                maxTokens: maxTokens,
                topP: topP
            )
        } else {
            return try await handleNonStreamingResponse(
                requestID: requestID,
                created: created,
                prompt: prompt,
                temperature: temperature,
                maxTokens: maxTokens,
                topP: topP
            )
        }
    }

    private func handleStreamingResponse(
        requestID: String,
        created: Int,
        prompt: String,
        temperature: Float,
        maxTokens: Int,
        topP: Float
    ) async throws -> Response {
        await telemetry.requestStarted()
        await queue.acquire()

        let stream = engine.chat(
            prompt: prompt,
            temperature: temperature,
            maxTokens: maxTokens,
            topP: topP
        )

        let sseStream = AsyncThrowingStream<String, Error> { continuation in
            Task { [weak self] in
                guard let self = self else { return }

                // Initial chunk with role
                let roleChunk = ChatCompletionChunk(
                    id: requestID,
                    created: created,
                    model: self.modelID,
                    choices: [.init(
                        index: 0,
                        delta: .init(role: "assistant", content: nil),
                        finishReason: nil
                    )]
                )
                if let data = try? JSONEncoder().encode(roleChunk),
                   let json = String(data: data, encoding: .utf8) {
                    continuation.yield("data: \(json)\n\n")
                }

                var lastStats: GenerationStats?

                do {
                    for try await event in stream {
                        switch event {
                        case .chunk(let text):
                            let chunk = ChatCompletionChunk(
                                id: requestID,
                                created: created,
                                model: self.modelID,
                                choices: [.init(
                                    index: 0,
                                    delta: .init(role: nil, content: text),
                                    finishReason: nil
                                )]
                            )
                            if let data = try? JSONEncoder().encode(chunk),
                               let json = String(data: data, encoding: .utf8) {
                                continuation.yield("data: \(json)\n\n")
                            }
                        case .info(let stats):
                            lastStats = stats
                        }
                    }

                    // Final chunk with finish_reason
                    let doneChunk = ChatCompletionChunk(
                        id: requestID,
                        created: created,
                        model: self.modelID,
                        choices: [.init(
                            index: 0,
                            delta: .init(role: nil, content: nil),
                            finishReason: "stop"
                        )]
                    )
                    if let data = try? JSONEncoder().encode(doneChunk),
                       let json = String(data: data, encoding: .utf8) {
                        continuation.yield("data: \(json)\n\n")
                    }

                    continuation.yield("data: [DONE]\n\n")
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }

                if let stats = lastStats {
                    await self.telemetry.requestCompleted(stats: stats)
                } else {
                    await self.telemetry.requestFailed()
                }
                await self.queue.release()
            }
        }

        let responseBody = ResponseBody(asyncSequence: sseStream.map { chunk in
            ByteBuffer(string: chunk)
        })

        return Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .cacheControl: "no-cache",
                .connection: "keep-alive",
            ],
            body: responseBody
        )
    }

    private func handleNonStreamingResponse(
        requestID: String,
        created: Int,
        prompt: String,
        temperature: Float,
        maxTokens: Int,
        topP: Float
    ) async throws -> Response {
        await telemetry.requestStarted()
        await queue.acquire()

        defer {
            Task { await queue.release() }
        }

        do {
            let (text, stats) = try await engine.generate(
                prompt: prompt,
                temperature: temperature,
                maxTokens: maxTokens,
                topP: topP
            )

            await telemetry.requestCompleted(stats: stats)

            let response = ChatCompletionResponse(
                id: requestID,
                created: created,
                model: modelID,
                choices: [.init(
                    index: 0,
                    message: ChatMessage(role: "assistant", content: text),
                    finishReason: "stop"
                )],
                usage: .init(
                    promptTokens: stats.promptTokenCount,
                    completionTokens: stats.generationTokenCount,
                    totalTokens: stats.promptTokenCount + stats.generationTokenCount
                )
            )

            let data = try JSONEncoder().encode(response)
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(data: data))
            )
        } catch {
            await telemetry.requestFailed()
            throw error
        }
    }

    @Sendable
    func handleListModels(request: Request, context: BasicRequestContext) async throws -> Response {
        let response = ModelListResponse(data: [
            .init(id: modelID, created: Int(Date().timeIntervalSince1970))
        ])
        let data = try JSONEncoder().encode(response)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    @Sendable
    func handleHealth(request: Request, context: BasicRequestContext) async throws -> Response {
        let health = await telemetry.healthSnapshot(
            modelID: modelID,
            memoryUsed: Int(MLX.GPU.Memory.activeMemory),
            memoryAvailable: engine.profile.maxRecommendedWorkingSetBytes,
            mtpEnabled: engine.currentMetadata.map { $0.mtpVariant != .none } ?? false
        )
        let data = try JSONEncoder().encode(health)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    // MARK: - Helpers

    private func buildPrompt(from messages: [ChatMessage]) -> String {
        // Extract the last user message as the prompt
        // System messages are handled via ChatSession's instructions
        // For stateless mode, we concatenate the conversation
        let lastUserMessage = messages.last { $0.role == "user" }
        return lastUserMessage?.content ?? ""
    }
}

// MARK: - CORS Middleware

struct CORSMiddleware: RouterMiddleware {
    func handle(
        _ request: Request,
        context: BasicRequestContext,
        next: (Request, BasicRequestContext) async throws -> Response
    ) async throws -> Response {
        var response = try await next(request, context)
        response.headers[.accessControlAllowOrigin] = "*"
        response.headers[.accessControlAllowMethods] = "GET, POST, OPTIONS"
        response.headers[.accessControlAllowHeaders] = "Content-Type, Authorization"
        return response
    }
}
