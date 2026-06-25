import Foundation
import CoreAILanguageModels
import FoundationModels

/// One streamed chunk from the engine: the cumulative generated text so far and
/// the prompt/output token counts the engine reports (zero until available).
/// A plain value type so consumers (the view model, the bench) never import the
/// underlying LLM framework.
public struct EngineChunk: Sendable {
    public let text: String
    public let inputTokens: Int
    public let outputTokens: Int

    public init(text: String, inputTokens: Int, outputTokens: Int) {
        self.text = text
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
    }
}

/// Abstraction over the on-device LLM runtime. This is the single seam that
/// isolates the Core AI framework: `ChatViewModel` and `LLMChatBench` depend on
/// this protocol, not on `CoreAILanguageModel` / `LanguageModelSession` directly.
public protocol LLMEngine: Sendable {
    /// Load the model bundle at `url`. Must complete before `stream`.
    func load(at url: URL) async throws
    /// Stream a response to `prompt`, yielding cumulative chunks.
    func stream(prompt: String, temperature: Double, maxTokens: Int) -> AsyncThrowingStream<EngineChunk, Error>
}

/// `LLMEngine` backed by Apple Core AI. This is the **only** file in the package
/// that imports `CoreAILanguageModels` / `FoundationModels`.
public final class CoreAIEngine: LLMEngine, @unchecked Sendable {
    private var model: CoreAILanguageModel?
    private var session: LanguageModelSession?

    public init() {}

    public func load(at url: URL) async throws {
        // These exports are compiled for S=1 steps; a larger chunk would trip an
        // MPS shape assertion during prefill. An engine-specific quirk, kept here.
        setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        let m = try await CoreAILanguageModel(resourcesAt: url)
        self.model = m
        self.session = LanguageModelSession(
            model: m,
            instructions: "You are a helpful AI assistant. Answer concisely."
        )
    }

    public func stream(prompt: String, temperature: Double, maxTokens: Int) -> AsyncThrowingStream<EngineChunk, Error> {
        guard let session else {
            return AsyncThrowingStream { $0.finish(throwing: NSError(domain: "LLMChat", code: 1, userInfo: [NSLocalizedDescriptionKey: "Engine not loaded"])) }
        }
        let options = GenerationOptions(temperature: temperature, maximumResponseTokens: maxTokens)
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    for try await partial in session.streamResponse(to: prompt, options: options) {
                        continuation.yield(EngineChunk(
                            text: partial.content,
                            inputTokens: partial.usage.input.totalTokenCount,
                            outputTokens: partial.usage.output.totalTokenCount
                        ))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
