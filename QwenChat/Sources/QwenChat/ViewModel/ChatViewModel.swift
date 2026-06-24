import Foundation
import SwiftUI
import CoreAILanguageModels
import FoundationModels

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var inputText = ""
    @Published var isGenerating = false
    @Published var modelLoadState: LoadState = .idle
    @Published var streamingText = ""

    // Model info
    @Published var modelName = "Qwen3-0.6B"
    @Published var modelSize = "331 MB"
    @Published var modelFormat = "4bit · dynamic"
    @Published var modelSpeed = "~200 tok/s"
    @Published var modelVocab = "151,936"

    // Settings
    @Published var temperature: Double = 0.7
    @Published var maxTokens: Int = 512

    private var model: (any LanguageModel)?
    private var session: LanguageModelSession?

    enum LoadState: String {
        case idle = "Load Model"
        case loading = "Loading..."
        case ready = "Ready"
        case error = "Error"
    }

    func loadModel() async {
        guard let bundleURL = findBundle() else {
            modelLoadState = .error
            return
        }
        modelLoadState = .loading
        setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        do {
            let m = try await CoreAILanguageModel(resourcesAt: bundleURL)
            self.model = m
            self.session = LanguageModelSession(
                model: m,
                instructions: "You are Qwen3, a helpful AI assistant. Answer concisely."
            )
            modelLoadState = .ready
        } catch {
            print("Load error: \(error)")
            modelLoadState = .error
        }
    }

    private func findBundle() -> URL? {
        let name = "qwen3_0_6b_4bit_dynamic"
        let candidates: [URL?] = [
            Bundle.main.bundleURL.appendingPathComponent("Models/\(name)"),
            FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
                .first?.appendingPathComponent("Models/\(name)"),
            // Dev fallback
            URL(fileURLWithPath: "/Users/zgd/Code/llm/core-ai/exports/qwen3_0.6b/\(name)"),
        ]
        for url in candidates {
            if let url, FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }

    func send() async {
        guard let session, !inputText.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        let prompt = inputText
        inputText = ""
        messages.append(ChatMessage(role: .user, content: prompt))
        isGenerating = true
        streamingText = ""
        do {
            let response = try await session.respond(to: prompt)
            messages.append(ChatMessage(role: .assistant, content: response.content))
        } catch {
            messages.append(ChatMessage(role: .assistant, content: "Error: \(error.localizedDescription)"))
        }
        isGenerating = false
        streamingText = ""
    }

    func clearChat() { messages.removeAll() }
}
