import Foundation
import LLMChat

/// Headless benchmark that mirrors the app's metric: overall throughput from
/// total wall-clock time + real token counts. A clean run here means the
/// per-reply read-out in the app is sound. Uses the same `LLMEngine` the app
/// uses, so it exercises the exact same Core AI path.
@main
struct LLMChatBench {
    static func main() async {
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
            let engine = CoreAIEngine()
            do {
                try await engine.load(at: url)
            } catch {
                print("  load failed: \(error)\n")
                continue
            }
            for prompt in prompts {
                await run(engine: engine, prompt: prompt)
            }
            print("")
        }
    }

    static func run(engine: LLMEngine, prompt: String) async {
        let t0 = ContinuousClock.now
        var firstToken: Duration?
        var content = ""
        var inputTokens = 0
        var outputTokens = 0

        do {
            for try await chunk in engine.stream(prompt: prompt, temperature: 0.7, maxTokens: 512) {
                if firstToken == nil { firstToken = ContinuousClock.now - t0 }
                content = chunk.text
                inputTokens = chunk.inputTokens
                outputTokens = chunk.outputTokens
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
