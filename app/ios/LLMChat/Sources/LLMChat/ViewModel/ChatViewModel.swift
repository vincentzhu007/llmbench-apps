import Foundation
import SwiftUI
import CoreAILanguageModels
import FoundationModels

/// Per-model chat controller. One instance per `ChatScreen`. Loads a single
/// model lazily, streams responses, and measures throughput.
///
/// Only overall throughput is reliably measurable on the `LanguageModelSession`
/// path: CoreAI buffers streamed output (snapshots arrive in a burst rather
/// than one per generated token), so a prefill-vs-decode split is not exposed.
/// For these S=1 model exports prefill and decode run at the same per-token
/// rate, so overall throughput equals both. See the design doc for the
/// follow-up (pipelined engine) that would give a true split on the 0.8B.
@MainActor
final class ChatViewModel: ObservableObject {
    let model: ModelDescriptor

    @Published var messages: [ChatMessage] = []
    @Published var inputText = ""
    @Published var isGenerating = false
    @Published var modelLoadState: LoadState = .idle
    @Published var streamingText = ""
    @Published var liveTokPerSec: Double? = nil   // live read-out while streaming

    // Generation settings
    @Published var temperature: Double = 0.7
    @Published var maxTokens: Int = 512

    // Session-wide average throughput over messages that carry metrics
    @Published var avgThroughput: Double? = nil

    private var languageModel: CoreAILanguageModel?   // retained for the session's lifetime
    private var session: LanguageModelSession?

    enum LoadState: String {
        case idle = "Load Model"
        case loading = "Loading…"
        case ready = "Ready"
        case error = "Error"
    }

    init(model: ModelDescriptor) {
        self.model = model
    }

    var isAvailable: Bool { model.resolveBundleURL() != nil }

    func loadModel() async {
        guard languageModel == nil else { return }
        guard let bundleURL = model.resolveBundleURL() else {
            modelLoadState = .error
            return
        }
        modelLoadState = .loading
        // These model exports are compiled for S=1 (single-token) steps. A
        // larger chunk threshold makes prefill process many tokens at once and
        // trips an MPS shape assertion, so force single-token chunking.
        setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        do {
            let m = try await CoreAILanguageModel(resourcesAt: bundleURL)
            self.languageModel = m
            self.session = LanguageModelSession(
                model: m,
                instructions: "You are \(model.displayName), a helpful AI assistant. Answer concisely."
            )
            modelLoadState = .ready
        } catch {
            print("Load error: \(error)")
            modelLoadState = .error
        }
    }

    func send() async {
        guard let session,
              !inputText.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        let prompt = inputText
        inputText = ""
        messages.append(ChatMessage(role: .user, content: prompt))
        isGenerating = true
        streamingText = ""
        liveTokPerSec = nil

        let options = GenerationOptions(temperature: temperature, maximumResponseTokens: maxTokens)
        let t0 = ContinuousClock.now
        var tTTF: Duration?
        var lastContent = ""
        var usageInputTokens = 0
        var usageOutputTokens = 0

        do {
            let stream = session.streamResponse(to: prompt, options: options)
            for try await partial in stream {
                if tTTF == nil { tTTF = ContinuousClock.now - t0 }
                lastContent = partial.content
                streamingText = lastContent
                usageInputTokens = partial.usage.input.totalTokenCount
                usageOutputTokens = partial.usage.output.totalTokenCount
                let liveTokens = usageInputTokens + usageOutputTokens
                liveTokPerSec = liveRate(tokens: liveTokens, t0: t0)
            }

            let totalDur = ContinuousClock.now - t0
            let promptTokens = usageInputTokens > 0 ? usageInputTokens : estimateTokens(prompt)
            let outputTokens = max(usageOutputTokens > 0 ? usageOutputTokens : estimateTokens(lastContent), 1)
            let throughput = rate(promptTokens + outputTokens, over: totalDur)

            messages.append(ChatMessage(
                role: .assistant,
                content: lastContent,
                metrics: ChatMessage.Metrics(
                    promptTokens: promptTokens,
                    outputTokens: outputTokens,
                    throughput: throughput,
                    ttftMs: Int((durationSeconds(tTTF ?? totalDur) * 1000).rounded())
                )
            ))
            updateAverage(with: throughput)
        } catch {
            // Streaming failed — fall back to a one-shot respond with no metrics.
            do {
                let resp = try await session.respond(to: prompt, options: options)
                messages.append(ChatMessage(role: .assistant, content: resp.content))
            } catch let fallbackError {
                messages.append(ChatMessage(role: .assistant, content: "Error: \(fallbackError.localizedDescription)"))
            }
        }

        isGenerating = false
        streamingText = ""
        liveTokPerSec = nil
    }

    func clearChat() {
        messages.removeAll()
        avgThroughput = nil
    }

    // MARK: - Timing helpers

    private func rate(_ tokens: Int, over duration: Duration) -> Double {
        let seconds = durationSeconds(duration)
        return seconds > 0 ? Double(tokens) / seconds : 0
    }

    private func liveRate(tokens: Int, t0: ContinuousClock.Instant) -> Double? {
        let elapsed = durationSeconds(ContinuousClock.now - t0)
        guard elapsed > 0.05 else { return nil }
        return Double(tokens) / elapsed
    }

    private func updateAverage(with throughput: Double) {
        let measured = messages.compactMap(\.metrics)
        guard !measured.isEmpty else { return }
        avgThroughput = measured.map(\.throughput).reduce(0, +) / Double(measured.count)
    }

    /// Rough token estimate (~3.5 chars/token) used when the engine doesn't
    /// report `Usage`, and for the live counter before usage arrives.
    private func estimateTokens(_ text: String) -> Int {
        max(Int((Double(text.count) / 3.5).rounded()), text.isEmpty ? 0 : 1)
    }

    private func durationSeconds(_ d: Duration) -> Double {
        let (s, atto) = d.components
        return Double(s) + Double(atto) / 1e18
    }
}
