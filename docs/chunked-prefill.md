# Chunked Prefill 支持状态

## Apple 官方方案

Apple 的 `InferenceEngine` 协议**原生支持 chunked prefill**：

```swift
// InferenceEngine.swift (apple/coreai-models)
public protocol InferenceEngine {
    var prefillChunkSize: Int { get }  // 默认 512 tokens
    var chunkThreshold: Int { get }    // 默认 1024 tokens
}
```

流程：prompt 超过 `chunkThreshold` 时自动拆分为 `prefillChunkSize` 的块，每块一次 GPU encode。无需应用层代码改动。

### 前提条件

模型必须是**动态 shape 导出**——通过 `coreai_models` 的 `export_to_coreai` 管道，使用 Apple 复合算子（`RMSNorm`、`RoPE`、`SDPA`、`GatedDeltaUpdate` 等）重写的模型。

```
动态 shape:  input_ids [1, -1] → logits [1, -1, vocab]   ← 支持 chunked prefill
静态 shape:  input_ids [1,  1] → logits [1,  1, vocab]   ← 仅 S=1
```

## Qwen3.5 现状

### 当前可用的模型

| 变体 | input_ids shape | COREAI_CHUNK_THRESHOLD | Prefill | Decode |
|------|:---:|:---:|---|---|
| gpu-pipelined (int8hu) | `[1, 1]` 静态 | 只能 =1 | S=1 逐步 ~70 tok/s | S=1 ~70 tok/s |
| ios-gpu (host-cache) | `[1, 1]` 静态 | 只能 =1 | S=1 逐步 | S=1 |
| macos (dynamic int8) | `[1, 1]` 静态 | 只能 =1 | S=1 逐步 | S=1 |

所有已发布的 Qwen3.5 Core AI bundle 都是 **decode-only loop-free** 导出——`input_ids` 静态 `[1, 1]`。chunked prefill 不可用。

### 尝试 chunk=64 报错

```
CoreAIRuntime/NDArrayDescriptor.swift:139: Fatal error:
  Shape at dimension 1 of 64 is not a valid substitution for source shape 1
```

根因: bundle 导出时 `input_ids` 被固定为 `[1, 1]`，引擎尝试创建 `[1, 64, vocab]` 的 logits 输出时 shape 冲突。

### 为什么是 loop-free

Qwen3.5 的 GatedDeltaNet 层使用 `scf.while` loop 做 SSM 时序扫描。在 **macOS 27 beta** 的 MPSGraph GPU delegate 上，`scf.while` **无法 lower**（"region type mismatch"）。

Zoo 的解决方案: 导出时每条 SSM 层走 `use_loopfree_step=True`——在 S=1 时与 while_loop 数值完全等价。代价是 input_ids 必须静态 `[1, 1]`。

## 要启用 chunked prefill

### 必要条件

1. **Apple 修复 MPSGraph 的 `scf.while` lowering bug** (FB23024751 / `apple/coreai-models#5`)
2. 或者用 `coreai_models` 覆盖层（`qwen3_5.py`）重写模型，导出动态 shape

### Zoo 的 chunked prefill 实测数据

Model card 中记录的分块 prefill（通过独立 prefill companion graph）：

| 方案 | 185-token prompt 耗时 | Prefill 速度 |
|------|:---:|:---:|
| S=1 逐步 prefill | 4.2 s | ~44 tok/s |
| **q16 chunked prefill** | **1.26 s** | **147 tok/s** |

q16 prefill companion graph: SSM scan 在图中展开，fp32 累积；KV/conv/rec states 传递给 decode graph。仅处理**完整块**——余数走 q=1 decode。

## 相关文件

- `CoreAIPipelinedEngine.swift` — `processChunkedInput()`, `chunkThreshold`, `prefillChunkSize`
- `InferenceEngine.swift` — 协议定义
- `coreai-model-zoo/zoo/qwen3.5.md` — model card
- `coreai-model-zoo/knowledge/coreai-beta-mpsgraph-kvwrite-bug.md` — beta bug 细节
