import Foundation
import SwiftUI

/// Per-model chat controller. One instance per `ChatScreen`. Loads a model via
/// the injected `LLMEngine`, streams responses, and measures throughput.
///
/// Only overall throughput is reliably measurable on this path: CoreAI buffers
/// streamed output, so a prefill-vs-decode split is not exposed. With the S=1
/// model exports prefill and decode run at the same per-token rate, so overall
/// throughput equals both. See the design doc for the follow-up (pipelined
/// engine) that would give a true split on the 0.8B.
@MainActor
final class ChatViewModel: ObservableObject {
    let model: ModelDescriptor
    private let engine: LLMEngine
    private var isLoaded = false

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

    enum LoadState: String {
        case idle = "Load Model"
        case loading = "Loading…"
        case ready = "Ready"
        case error = "Error"
    }

    init(model: ModelDescriptor, engine: LLMEngine = CoreAIEngine()) {
        self.model = model
        self.engine = engine
    }

    var isAvailable: Bool { model.resolveBundleURL() != nil }

    func loadModel() async {
        guard !isLoaded else { return }
        guard let bundleURL = model.resolveBundleURL() else {
            modelLoadState = .error
            return
        }
        modelLoadState = .loading
        do {
            try await engine.load(at: bundleURL)
            isLoaded = true
            modelLoadState = .ready
        } catch {
            print("Load error: \(error)")
            modelLoadState = .error
        }
    }

    func send() async {
        guard isLoaded,
              !inputText.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        let prompt = inputText
        inputText = ""
        messages.append(ChatMessage(role: .user, content: prompt))
        isGenerating = true
        streamingText = ""
        liveTokPerSec = nil

        let t0 = ContinuousClock.now
        var tTTF: Duration?
        var lastContent = ""
        var inputTokens = 0
        var outputTokens = 0
        var reasoningTokens = 0

        do {
            for try await chunk in engine.stream(prompt: prompt, temperature: temperature, maxTokens: maxTokens) {
                if tTTF == nil { tTTF = ContinuousClock.now - t0 }
                lastContent = chunk.text
                streamingText = lastContent
                inputTokens = chunk.inputTokens
                outputTokens = chunk.outputTokens
                reasoningTokens = chunk.reasoningTokens
                liveTokPerSec = liveRate(tokens: inputTokens + outputTokens, t0: t0)
            }

            let totalDur = ContinuousClock.now - t0
            let promptTokens = inputTokens > 0 ? inputTokens : estimateTokens(prompt)
            let outputTokens = max(outputTokens > 0 ? outputTokens : estimateTokens(lastContent), 1)
            let throughput = rate(promptTokens + outputTokens, over: totalDur)

            messages.append(ChatMessage(
                role: .assistant,
                content: lastContent,
                metrics: ChatMessage.Metrics(
                    promptTokens: promptTokens,
                    outputTokens: outputTokens,
                    throughput: throughput,
                    ttftMs: Int((durationSeconds(tTTF ?? totalDur) * 1000).rounded()),
                    reasoningTokens: reasoningTokens
                )
            ))
            updateAverage(with: throughput)
        } catch {
            messages.append(ChatMessage(role: .assistant, content: "Error: \(error.localizedDescription)"))
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
    /// report token counts, and for the live counter before they arrive.
    private func estimateTokens(_ text: String) -> Int {
        max(Int((Double(text.count) / 3.5).rounded()), text.isEmpty ? 0 : 1)
    }

    private func durationSeconds(_ d: Duration) -> Double {
        let (s, atto) = d.components
        return Double(s) + Double(atto) / 1e18
    }
}
