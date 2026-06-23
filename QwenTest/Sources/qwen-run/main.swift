import CoreAI
import Foundation

// MARK: - NDArray helpers

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

func ndDesc(_ d: InferenceValue.Descriptor?) -> NDArrayDescriptor? {
    guard case .ndArray(let nd)? = d else { return nil }
    return nd
}

// MARK: - BPE Tokenizer

final class BPETokenizer {
    var vocab: [String: Int] = [:]
    var invVocab: [Int: String] = [:]
    var mergeRank: [String: Int] = [:]

    init?(dir: URL) {
        let vocabURL = dir.appendingPathComponent("vocab.json")
        let mergesURL = dir.appendingPathComponent("merges.txt")

        guard let vocabData = try? Data(contentsOf: vocabURL),
              let vocabJSON = try? JSONSerialization.jsonObject(with: vocabData) as? [String: Any],
              let vocabDict = vocabJSON as? [String: Int] else { return nil }
        self.vocab = vocabDict
        for (t, id) in vocabDict { invVocab[id] = t }

        if let s = try? String(contentsOf: mergesURL, encoding: .utf8) {
            let lines = s.components(separatedBy: "\n").dropFirst()
                .filter { !$0.isEmpty && !$0.hasPrefix("#") }
            for (i, line) in lines.enumerated() {
                let p = line.components(separatedBy: " ")
                if p.count >= 2 { mergeRank["\(p[0]) \(p[1])"] = i }
            }
        }
    }

    func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        for word in text.components(separatedBy: " ") {
            guard !word.isEmpty else { continue }
            var tokens = word.map { String($0) }
            while tokens.count > 1 {
                var best = Int.max, idx = -1
                for i in 0..<(tokens.count - 1) {
                    if let r = mergeRank["\(tokens[i]) \(tokens[i+1])"], r < best { best = r; idx = i }
                }
                if idx < 0 { break }
                tokens[idx] = tokens[idx] + tokens[idx + 1]
                tokens.remove(at: idx + 1)
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

// MARK: - Inference Engine (host-cache: states as I/O tensors)

final class InferenceEngine {
    let fn: InferenceFunction
    let desc: InferenceFunctionDescriptor
    let ctx: Int
    let vocabSize: Int

    let inputIdsDesc, posIdsDesc, maskDesc: NDArrayDescriptor
    let pastKDesc, pastVDesc, convDesc, recDesc: NDArrayDescriptor

    var pastK, pastV, convState, recState: NDArray

    init?(modelURL: URL, maxCtx: Int) async throws {
        let model = try await AIModel(contentsOf: modelURL)
        guard let d = model.functionDescriptor(for: "main"),
              let f = try model.loadFunction(named: "main") else { return nil }
        self.fn = f; self.desc = d; self.ctx = maxCtx

        guard let iid = ndDesc(d.inputDescriptor(of: "input_ids")),
              let pid = ndDesc(d.inputDescriptor(of: "position_ids")),
              let mid = ndDesc(d.inputDescriptor(of: "causal_mask")),
              let pkd = ndDesc(d.inputDescriptor(of: "past_k")),
              let pvd = ndDesc(d.inputDescriptor(of: "past_v")),
              let cd = ndDesc(d.inputDescriptor(of: "conv_state")),
              let rd = ndDesc(d.inputDescriptor(of: "rec_state")),
              let ld = ndDesc(d.outputDescriptor(of: "logits")) else { return nil }

        self.inputIdsDesc = iid; self.posIdsDesc = pid; self.maskDesc = mid
        self.pastKDesc = pkd; self.pastVDesc = pvd; self.convDesc = cd; self.recDesc = rd
        self.vocabSize = ld.shape.last ?? 248320

        self.pastK = Self.makeZero(pkd, maxCtx)
        self.pastV = Self.makeZero(pvd, maxCtx)
        self.convState = Self.makeZero(cd, maxCtx)
        self.recState = Self.makeZero(rd, maxCtx)

        print("Engine: vocab=\(vocabSize), ctx=\(maxCtx)")
    }

    private static func makeZero(_ desc: NDArrayDescriptor, _ ctx: Int) -> NDArray {
        let shape = desc.shape.map { $0 < 0 ? ctx : $0 }
        var arr = NDArray(descriptor: desc.resolvingDynamicDimensions(shape))
        fillNDArray(&arr, as: Float16.self, with: [Float16](repeating: 0, count: shape.reduce(1, *)))
        return arr
    }

    func step(tokenId: Int32, position: Int) async throws -> [Float16] {
        var inputIds = NDArray(descriptor: inputIdsDesc.resolvingDynamicDimensions([1, 1]))
        fillNDArray(&inputIds, as: Int32.self, with: [tokenId])

        var posIds = NDArray(descriptor: posIdsDesc.resolvingDynamicDimensions([1, 1]))
        fillNDArray(&posIds, as: Int32.self, with: [Int32(position)])

        let maskShape = maskDesc.shape.map { $0 < 0 ? ctx + 1 : $0 }
        var mask = NDArray(descriptor: maskDesc.resolvingDynamicDimensions(maskShape))
        var maskVals = [Float16](repeating: 0, count: maskShape.reduce(1, *))
        for i in 0...position { maskVals[i] = 1.0 }
        fillNDArray(&mask, as: Float16.self, with: maskVals)

        var result = try await fn.run(inputs: [
            "input_ids": inputIds, "position_ids": posIds, "causal_mask": mask,
            "past_k": pastK, "past_v": pastV,
            "conv_state": convState, "rec_state": recState,
        ])

        guard let logitsArr = result.remove("logits")?.ndArray else {
            throw NSError(domain: "Engine", code: 1)
        }
        let logits = readNDArray(logitsArr, as: Float16.self, count: vocabSize)

        if let kCur = result.remove("k_cur")?.ndArray {
            pastK = writeKVColumn(into: pastK, desc: pastKDesc, from: kCur, position: position)
        }
        if let vCur = result.remove("v_cur")?.ndArray {
            pastV = writeKVColumn(into: pastV, desc: pastVDesc, from: vCur, position: position)
        }
        if let c = result.remove("conv_cur")?.ndArray {
            convState = rebuildState(from: c, desc: convDesc)
        }
        if let r = result.remove("rec_cur")?.ndArray {
            recState = rebuildState(from: r, desc: recDesc)
        }

        return logits
    }

    private func writeKVColumn(into cache: NDArray, desc: NDArrayDescriptor,
                                from col: NDArray, position: Int) -> NDArray {
        let nL = desc.shape[0], nH = desc.shape[2], hd = desc.shape.last ?? 256
        var data = readNDArray(cache, as: Float16.self, count: nL * nH * ctx * hd)
        let colData = readNDArray(col, as: Float16.self, count: nL * nH * hd)

        for l in 0..<nL {
            for h in 0..<nH {
                let co = l * nH * hd + h * hd
                let do_ = l * nH * ctx * hd + h * ctx * hd + position * hd
                for d in 0..<hd { data[do_ + d] = colData[co + d] }
            }
        }

        let shape = desc.shape.map { $0 < 0 ? ctx : $0 }
        var newArr = NDArray(descriptor: desc.resolvingDynamicDimensions(shape))
        fillNDArray(&newArr, as: Float16.self, with: data)
        return newArr
    }

    private func rebuildState(from src: NDArray, desc: NDArrayDescriptor) -> NDArray {
        let shape = desc.shape.map { $0 < 0 ? ctx : $0 }
        let total = shape.reduce(1, *)
        var newArr = NDArray(descriptor: desc.resolvingDynamicDimensions(shape))
        let data = readNDArray(src, as: Float16.self, count: total)
        fillNDArray(&newArr, as: Float16.self, with: data)
        return newArr
    }

    func argmax(_ logits: [Float16]) -> Int32 {
        var best = 0, bestVal = logits[0]
        for i in 1..<logits.count where logits[i] > bestVal { bestVal = logits[i]; best = i }
        return Int32(best)
    }
}

// MARK: - Main

func main() async {
    let args = CommandLine.arguments
    guard args.count >= 3 else {
        print("Usage: qwen-run <model.aimodel> <tokenizer-dir> [prompt]")
        return
    }

    let modelURL = URL(fileURLWithPath: args[1])
    let tokDir = URL(fileURLWithPath: args[2])
    let prompt = args.count > 3 ? args[3] : "The capital of France is"

    print("=== Qwen3.5-0.8B Core AI Inference (Swift) ===")

    guard let tok = BPETokenizer(dir: tokDir) else {
        print("Error: failed to load tokenizer"); return
    }

    do {
        guard let engine = try await InferenceEngine(modelURL: modelURL, maxCtx: 2048) else {
            print("Error: failed to create engine"); return
        }

        let promptIds = tok.encode(prompt)
        print("Prompt: '\(prompt)' (\(promptIds.count) tokens)")

        var logits: [Float16] = []
        let t0 = Date()
        for (i, tid) in promptIds.enumerated() {
            logits = try await engine.step(tokenId: Int32(tid), position: i)
        }
        let prefillMs = -t0.timeIntervalSinceNow * 1000
        print(String(format: "Prefill: %d tokens in %.0fms", promptIds.count, prefillMs))

        let eosId = Int32(tok.eosTokenId)
        var generated: [Int32] = []
        var nextId = engine.argmax(logits)
        let decStart = Date()

        for _ in 0..<64 {
            if nextId == eosId { break }
            generated.append(nextId)
            logits = try await engine.step(tokenId: nextId, position: promptIds.count + generated.count - 1)
            nextId = engine.argmax(logits)
        }

        let decElapsed = -decStart.timeIntervalSinceNow
        print("\n" + String(repeating: "=", count: 50))
        print("Generated: \(tok.decode(generated.map { Int($0) }))")
        print(String(repeating: "=", count: 50))
        if !generated.isEmpty {
            print(String(format: "%d tokens in %.2fs (%.1f tok/s)",
                         generated.count, decElapsed, Double(generated.count) / decElapsed))
        }
    } catch {
        print("Error: \(error)")
    }
}

await main()
