import CoreAI
import Foundation

// MARK: - NDArray helpers

func ndDesc(_ d: InferenceValue.Descriptor?) -> NDArrayDescriptor? {
    guard case .ndArray(let nd)? = d else { return nil }
    return nd
}

func fillNDArray<T, S: Sequence>(_ array: inout NDArray, as: T.Type, with values: S)
where S.Element == T, T: BitwiseCopyable {
    var view = array.mutableView(as: T.self)
    view.withUnsafeMutablePointer { ptr, _, _ in
        var i = 0
        for v in values { ptr[i] = v; i += 1 }
    }
}

func readNDArray<T: BitwiseCopyable>(_ array: NDArray, as: T.Type, count: Int) -> [T] {
    var out = [T]()
    out.reserveCapacity(count)
    array.view(as: T.self).withUnsafePointer { ptr, _, _ in
        for i in 0..<count { out.append(ptr[i]) }
    }
    return out
}

// MARK: - BPE Tokenizer

final class BPETokenizer {
    var vocab: [String: Int] = [:]
    var invVocab: [Int: String] = [:]
    var mergeRank: [String: Int] = [:]

    init?(dir: URL) {
        guard let vd = try? Data(contentsOf: dir.appendingPathComponent("vocab.json")),
              let vj = try? JSONSerialization.jsonObject(with: vd) as? [String: Any],
              let v = vj as? [String: Int] else { return nil }
        self.vocab = v
        for (t, i) in v { invVocab[i] = t }

        if let s = try? String(contentsOf: dir.appendingPathComponent("merges.txt"), encoding: .utf8) {
            for (i, line) in s.components(separatedBy: "\n").dropFirst()
                .filter({ !$0.isEmpty && !$0.hasPrefix("#") }).enumerated() {
                let p = line.components(separatedBy: " ")
                if p.count >= 2 { mergeRank["\(p[0]) \(p[1])"] = i }
            }
        }
    }

    func encode(_ text: String) -> [Int] {
        var ids = [Int]()
        for word in text.components(separatedBy: " ") {
            guard !word.isEmpty else { continue }
            var tokens = word.map { String($0) }
            while tokens.count > 1 {
                var best = Int.max, idx = -1
                for i in 0..<(tokens.count - 1) {
                    if let r = mergeRank["\(tokens[i]) \(tokens[i+1])"], r < best { best = r; idx = i }
                }
                if idx < 0 { break }
                tokens[idx] += tokens[idx + 1]; tokens.remove(at: idx + 1)
            }
            for t in tokens { ids.append(vocab[t] ?? vocab["<unk>"] ?? 0) }
        }
        return ids
    }

    func decode(_ ids: [Int]) -> String {
        var text = ""
        for id in ids {
            if let t = invVocab[id] {
                if t.hasPrefix("Ġ") || t.hasPrefix("▁") { text += " " + String(t.dropFirst()) }
                else { text += t }
            }
        }
        return text.replacingOccurrences(of: "Ġ", with: " ").replacingOccurrences(of: "▁", with: " ")
    }

    var eosTokenId: Int { vocab["<|im_end|>"] ?? vocab["<|endoftext|>"] ?? 248046 }
}

// MARK: - Simple Forward Engine

struct ForwardEngine {
    let fn: InferenceFunction
    let seqLen: Int
    let vocabSize: Int

    let inputDesc: NDArrayDescriptor
    let outputDesc: NDArrayDescriptor

    init(modelURL: URL) async throws {
        let model = try await AIModel(contentsOf: modelURL)
        guard let d = model.functionDescriptor(for: "main"),
              let f = try model.loadFunction(named: "main") else {
            throw NSError(domain: "Engine", code: 1)
        }
        self.fn = f
        guard let iDesc = ndDesc(d.inputDescriptor(of: "input_ids")),
              let oDesc = ndDesc(d.outputDescriptor(of: "logits")) else {
            throw NSError(domain: "Engine", code: 2)
        }
        self.inputDesc = iDesc
        self.outputDesc = oDesc
        self.seqLen = iDesc.shape[1]
        self.vocabSize = oDesc.shape.last ?? 248320
        print("Engine: seqLen=\(seqLen), vocab=\(vocabSize)")
    }

    func forward(_ tokenIds: [Int32]) async throws -> [Int32] {
        let n = min(tokenIds.count, seqLen)
        var padded = [Int32](repeating: 0, count: seqLen)
        let start = seqLen - n
        for i in 0..<n { padded[start + i] = tokenIds[tokenIds.count - n + i] }

        let shape = [1, seqLen]
        var ids = NDArray(descriptor: inputDesc.resolvingDynamicDimensions(shape))
        fillNDArray(&ids, as: Int32.self, with: padded)

        var result = try await fn.run(inputs: ["input_ids": ids])

        guard let logitsArr = result.remove("logits")?.ndArray else {
            throw NSError(domain: "Engine", code: 3)
        }
        let logits = readNDArray(logitsArr, as: Float16.self, count: seqLen * vocabSize)

        // Last token's logits + argmax
        let offset = (seqLen - 1) * vocabSize
        var best = Int32(0), bestVal = logits[offset]
        for i in 1..<vocabSize where logits[offset + i] > bestVal { bestVal = logits[offset + i]; best = Int32(i) }
        return [best]
    }
}

// MARK: - Main

func main() async {
    let args = CommandLine.arguments
    guard args.count >= 3 else {
        print("Usage: qwen-run <model.aimodel> <tokenizer-dir> [prompt]")
        return
    }

    let modelPath = args[1]; let tokPath = args[2]
    let prompt = args.count > 3 ? args[3] : "The capital of France is"

    guard let tok = BPETokenizer(dir: URL(fileURLWithPath: tokPath)) else {
        print("Error: tokenizer"); return
    }
    print("Qwen3.5-4B int4 stateless forward")
    print("Model: \(modelPath)")

    do {
        let engine = try await ForwardEngine(modelURL: URL(fileURLWithPath: modelPath))

        var history = tok.encode(prompt)
        print("Prompt (\(history.count) tokens): \(prompt)")
        let t0 = Date()

        let maxNew = 32; let eos = Int32(tok.eosTokenId)
        for _ in 0..<maxNew {
            let next = try await engine.forward(history.map { Int32($0) })
            if next[0] == eos { break }
            history.append(Int(next[0]))
        }

        let elapsed = -t0.timeIntervalSinceNow
        let generated = history.dropFirst(tok.encode(prompt).count)
        print("Generated: \(tok.decode(generated.map { Int($0) }))")
        print(String(format: "\(generated.count) tokens in %.1fs (%.1f tok/s)", elapsed, Double(generated.count)/elapsed))
    } catch {
        print("Error: \(error)")
    }
}

await main()
