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
    let bundleDir = "../qwen3.5-0.8B-CoreAI/gpu-pipelined/qwen3_5_0_8b_decode_int8hu_perchan_sym"
    let url = URL(fileURLWithPath: bundleDir)

    print("=== Qwen3.5-0.8B via Foundation Models ===\n")

    setenv("COREAI_CHUNK_THRESHOLD", "1", 1)

    do {
        // ---- Load ----
        let loadStart = Date()
        let model = try await CoreAILanguageModel(resourcesAt: url)
        let loadMs = -loadStart.timeIntervalSinceNow * 1000

        let session = LanguageModelSession(
            model: model,
            instructions: "You are a helpful assistant. Answer concisely."
        )

        // ---- Model size ----
        let aimodelURL = url.appendingPathComponent("\(url.lastPathComponent).aimodel")
        let bundleSize = sizeStr(url)
        let aimodelFileSize = sizeStr(aimodelURL)

        // ---- Long prompt (~1k tokens) ----
        // Build a long prompt by repeating paragraphs
        let baseText = """
        请详细分析以下内容的各个方面，包括历史背景、技术原理、应用场景和未来发展趋势。\
        人工智能（Artificial Intelligence）是计算机科学的一个重要分支，旨在创建能够模拟人类智能的系统。\
        从1956年达特茅斯会议正式提出AI概念以来，这一领域经历了多次起伏。\
        早期的符号主义AI试图通过逻辑推理来模拟人类思维，但受限于计算能力和知识表示的困难，进展缓慢。\
        20世纪80年代，专家系统曾一度兴起，但很快因维护困难和知识获取瓶颈而衰落。\
        进入21世纪，随着大数据、云计算和GPU并行计算的发展，深度学习技术取得了突破性进展。\
        2012年，AlexNet在ImageNet图像识别比赛中大幅领先传统方法，标志着深度学习时代的到来。\
        此后，循环神经网络（RNN）、长短期记忆网络（LSTM）、生成对抗网络（GAN）等技术相继涌现。\
        2017年，Google提出了Transformer架构，彻底改变了自然语言处理领域的面貌。\
        基于Transformer的预训练语言模型如BERT、GPT系列展现出了强大的语义理解和生成能力。\
        2022年底，ChatGPT的发布引发了全球性的AI热潮，展示了大型语言模型在对话、写作、编程等方面的惊人能力。\
        当前，AI技术正在向多模态、具身智能、通用人工智能（AGI）等方向演进。\
        同时，AI的安全治理、伦理规范、隐私保护等问题也日益受到关注。\
        在产业应用方面，AI已渗透到医疗诊断、金融风控、自动驾驶、智能制造、智慧农业等各个领域。\
        未来，AI预计将在科学发现、药物研发、气候变化应对等方面发挥更大作用。\
        然而，如何确保AI系统的可靠性、可解释性和公平性，仍然是亟待解决的关键挑战。\
        请从以上多个维度对这一主题进行全面深入的分析，并给出你的见解。
        """

        // Build ~1k token prompt by repeating
        let shortPrompt = "请介绍5种上海的美食"
        var longPrompt = baseText
        // Add question-like content to reach ~1k tokens
        longPrompt += "\n\n" + String(repeating: baseText + "\n", count: 3)
        // Trim to reasonable length
        if longPrompt.count > 8000 { longPrompt = String(longPrompt.prefix(8000)) }

        print("Prompt length: \(longPrompt.count) chars")

        print("Prompt tokens: (estimated from inference)")

        // ---- Inference ----
        let inferenceStart = Date()
        let response = try await session.respond(to: longPrompt)
        let inferenceMs = -inferenceStart.timeIntervalSinceNow * 1000

        // ---- Output (truncated) ----
        let previewLen = min(response.content.count, 500)
        print("\nResponse (first \(previewLen) chars):")
        print(String(response.content.prefix(previewLen)))
        if response.content.count > previewLen {
            print("... (\(response.content.count) total chars)")
        }

        // ---- Timing breakdown ----
        let usage = session.usage
        let totalTokens = usage.totalTokenCount
        // Estimate prompt/output split from char ratio (~2 chars/token for Chinese)
        let estPromptTokens = max(1, totalTokens * longPrompt.count / (longPrompt.count + response.content.count))
        let estOutputTokens = totalTokens - estPromptTokens

        let avgMsPerToken = inferenceMs / Double(max(totalTokens, 1))
        let prefillMs = Double(estPromptTokens) * avgMsPerToken
        let decodeMs = Double(estOutputTokens) * avgMsPerToken
        let prefillTokPerSec = estPromptTokens > 0 ? Double(estPromptTokens) / (prefillMs / 1000.0) : 0
        let decodeTokPerSec = estOutputTokens > 0 ? Double(estOutputTokens) / (decodeMs / 1000.0) : 0
        let overallTokPerSec = totalTokens > 0 ? Double(totalTokens) / (inferenceMs / 1000.0) : 0

        print("")
        print(String(repeating: "─", count: 62))
        print("Backend    Core AI Pipelined Engine (GPU)")
        print("Model      Qwen3.5-0.8B · int8 · decode-only S=1")
        print(String(repeating: "─", count: 62))
        print("Size       bundle \(bundleSize)  .aimodel \(aimodelFileSize)")
        print(String(format: "Load        %.0f ms", loadMs))
        print(String(repeating: "─", count: 62))
        print("Prompt      ~\(estPromptTokens) tokens  (\(longPrompt.count) chars)")
        print("Output      ~\(estOutputTokens) tokens  (\(response.content.count) chars)")
        print("Total       \(totalTokens) tokens")
        print(String(repeating: "─", count: 62))
        print(String(format: "Prefill     %.0f ms  (%.1f tok/s)", prefillMs, prefillTokPerSec))
        print(String(format: "Decode      %.0f ms  (%.1f tok/s)", decodeMs, decodeTokPerSec))
        print(String(format: "Total       %.0f ms  (%.1f tok/s)", inferenceMs, overallTokPerSec))
        print(String(repeating: "─", count: 62))

    } catch {
        print("Error: \(error)")
    }
}

await main()
