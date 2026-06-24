import CoreAILanguageModels
import FoundationModels
import Foundation

func sizeStr(_ url: URL) -> String {
    let fm = FileManager.default
    guard let e = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return "0 B" }
    var total: Int64 = 0
    for case let file as URL in e {
        total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }
    if total >= 1_073_741_824 { return String(format: "%.1f GB", Double(total) / 1_073_741_824) }
    if total >= 1_048_576 { return String(format: "%.0f MB", Double(total) / 1_048_576) }
    return "\(total) B"
}

func main() async {
    let bundleDir = "../exports/qwen3_0.6b/qwen3_0_6b_4bit_dynamic"
    let url = URL(fileURLWithPath: bundleDir)

    print("=== Qwen3-0.6B via Foundation Models ===\n")

    // Qwen3 is standard architecture (2-state KV cache) — no extra states needed.
    // No COREAI_CHUNK_THRESHOLD required for dynamic models.

    do {
        // ---- Load ----
        let loadStart = Date()
        let model = try await CoreAILanguageModel(resourcesAt: url)
        let loadMs = -loadStart.timeIntervalSinceNow * 1000

        let aimodelURL = url.appendingPathComponent("\(url.lastPathComponent).aimodel")

        let session = LanguageModelSession(
            model: model,
            instructions: "You are a helpful assistant. Answer concisely."
        )

        // Short prompt test
        let shortPrompt = "请介绍5种上海的美食"
        print("Prompt: \(shortPrompt)")
        let t0 = Date()
        let response = try await session.respond(to: shortPrompt)
        let shortMs = -t0.timeIntervalSinceNow * 1000
        let shortUsage = session.usage

        print("Response: \(response.content.prefix(600))")
        if response.content.count > 600 { print("... (\(response.content.count) total chars)") }

        // Long prompt test (~1k chars)
        let longPrompt = """
        请从历史背景、技术原理、应用场景和未来趋势四个方面，详细介绍人工智能的发展历程。\
        人工智能（Artificial Intelligence）是计算机科学的重要分支。1956年达特茅斯会议正式提出AI概念。\
        早期符号主义AI试图通过逻辑推理模拟人类思维。20世纪80年代专家系统曾一度兴起。\
        进入21世纪，深度学习取得突破。2012年AlexNet在ImageNet比赛中大幅领先。\
        2017年Google提出Transformer架构，改变了NLP领域。GPT、BERT等预训练模型展现了强大能力。\
        当前AI正向多模态、具身智能等方向演进。AI安全治理也日益受到关注。\
        请综合以上内容进行全面分析。
        """

        print("\n--- Long prompt test ---")
        print("Prompt length: \(longPrompt.count) chars")

        let t1 = Date()
        let longResponse = try await session.respond(to: longPrompt)
        let longMs = -t1.timeIntervalSinceNow * 1000
        let longUsage = session.usage

        print("Response (first 400 chars): \(longResponse.content.prefix(400))")
        if longResponse.content.count > 400 { print("... (\(longResponse.content.count) total chars)") }

        // ---- Stats ----
        print("")
        print(String(repeating: "─", count: 62))
        print("Backend    Core AI Pipelined Engine (GPU)")
        print("Model      Qwen3-0.6B · 4bit · dynamic")
        print("Bundle     qwen3_0_6b_4bit_dynamic")
        print(String(repeating: "─", count: 62))
        print("Size       bundle \(sizeStr(url))  .aimodel \(sizeStr(aimodelURL))")
        print(String(format: "Load        %.0f ms", loadMs))
        print(String(repeating: "─", count: 62))
        let shortTokens = shortUsage.totalTokenCount
        let longTokens = longUsage.totalTokenCount - shortTokens

        print("Short (\(shortPrompt.count) chars):")
        print(String(format: "  Inference  %.0f ms", shortMs))
        print("  Tokens      \(shortTokens)")
        if shortMs > 0 && shortTokens > 0 {
            print(String(format: "  Throughput  %.1f tok/s", Double(shortTokens) / (shortMs / 1000.0)))
        }
        print("Long (\(longPrompt.count) chars):")
        print(String(format: "  Inference  %.0f ms", longMs))
        print("  Tokens      \(longTokens)")
        if longMs > 0 && longTokens > 0 {
            print(String(format: "  Throughput  %.1f tok/s", Double(longTokens) / (longMs / 1000.0)))
        }
        print(String(repeating: "─", count: 62))

    } catch {
        print("Error: \(error)")
    }
}

await main()
