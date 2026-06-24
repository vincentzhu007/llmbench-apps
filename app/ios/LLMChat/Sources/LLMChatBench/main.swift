import Foundation
import CoreAILanguageModels
import FoundationModels
import LLMChat

/// Headless benchmark that mirrors the app's metric: overall throughput from
/// total wall-clock time + real token counts from `Usage`. A clean run here
/// means the per-reply read-out in the app is sound.
@main
struct LLMChatBench {
    static func main() async {
        // These model exports are compiled for S=1 steps; a larger chunk would
        // trip an MPS shape assertion during prefill.
        setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        let prompts = [
            "The capital of France is",
            "List three primary colors.",
        ]
        for desc in ModelRegistry.all {
            print("=== \(desc.displayName)  (\(desc.quant)) ===")
            guard let url = desc.resolveBundleURL() else {
                print("  bundle not found: \(desc.bundleName)\n")
                continue
            }
            let model: CoreAILanguageModel
            do {
                model = try await CoreAILanguageModel(resourcesAt: url)
            } catch {
                print("  load failed: \(error)\n")
                continue
            }
            let session = LanguageModelSession(
                model: model,
                instructions: "You are a helpful assistant. Be concise."
            )
            for prompt in prompts {
                await run(session: session, prompt: prompt)
            }
            print("")
        }
    }

    static func run(session: LanguageModelSession, prompt: String) async {
        let t0 = ContinuousClock.now
        var firstToken: Duration?
        var content = ""
        var inputTokens = 0
        var outputTokens = 0

        do {
            for try await partial in session.streamResponse(to: prompt) {
                if firstToken == nil { firstToken = ContinuousClock.now - t0 }
                content = partial.content
                inputTokens = partial.usage.input.totalTokenCount
                outputTokens = partial.usage.output.totalTokenCount
            }
        } catch {
            print("  [\(prompt)] stream failed: \(error)")
            return
        }

        let total = durationSeconds(ContinuousClock.now - t0)
        let ttft = durationSeconds(firstToken ?? .zero)
        let inTok = inputTokens > 0 ? inputTokens : estimateTokens(prompt)
        let outTok = max(outputTokens, 1)
        let throughput = Double(inTok + outTok) / total

        print(String(format: "  Q: %@", prompt))
        print(String(format: "     A: %@", String(content.prefix(60)).replacingOccurrences(of: "\n", with: " ")))
        print(String(format: "     tok %d→%d  total %.2fs  TTFT %.2fs  throughput %.0f tok/s",
                     inTok, outTok, total, ttft, throughput))
    }

    static func estimateTokens(_ text: String) -> Int {
        max(Int((Double(text.count) / 3.5).rounded()), 1)
    }

    static func durationSeconds(_ d: Duration) -> Double {
        let (s, a) = d.components
        return Double(s) + Double(a) / 1e18
    }
}
