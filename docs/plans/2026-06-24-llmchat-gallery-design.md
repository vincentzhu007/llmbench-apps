# LLMChat — Gallery 风格 LLM Chat 应用设计

**日期**: 2026-06-24
**目标**: 把现有 LLMChat 重构为 Google AI Gallery 风格的 UI 应用，支持 LLM 对话，并展示推理速度。同时支持 macOS 和 iOS 双平台运行。

---

## 实施补遗（2026-06-25）

设计落地时发现一个关键事实，**metrics 口径相应调整**：

- **设计原定** prefill/decode 拆分（第 3/4 段）基于流式 TTFT 近似。
- **实测发现**：CoreAI 的 `streamResponse` 会**缓冲输出、批量吐 snapshot**（11 个 snapshot 挤在 38ms 内，首个 snapshot 时已生成 77 token），所以 streaming 切不出 prefill/decode；`PerformanceMetrics.shared` 在 `LanguageModelSession` 路径也不填充（仅 pipelined 引擎手动喂）。
- **且模型为 S=1 导出**（`COREAI_CHUNK_THRESHOLD=1`，否则 prefill 多 token 触发 `NDArrayDescriptor` 形状断言崩溃），S=1 下 prefill 与 decode 本就是同一种单 token 操作、速率相等。
- **决定（用户已选 A）**：每条回复显示**可靠的总吞吐** `throughput = (prompt+output) tokens / 总墙钟时间` + 真实 token 数 + TTFT，诚实标注 `(S=1: prefill≈decode)`。验证：0.6B ~175 tok/s、0.8B ~70 tok/s（与 model card 吻合）。
- **后续 B（未做）**：接入 `CoreAIPipelinedEngine` + 0.8B 的 `prefill_b2048` 模型，拿真实的 prefill/decode 拆分。仅 0.8B 可行（0.6B 无独立 prefill 导出）。

另：设计第 2/6 段假设 0.8B 按 `ios-gpu/macos` 平台分支加载 —— 实测那两个目录是裸 aimodel（无 tokenizer/metadata），不是 `LanguageModelSession` 可加载的 bundle。已简化为**每模型单 bundle、两端共用**（0.8B 用 `gpu-pipelined/..._perchan_sym` bundle）。


---

## 1. 整体架构与导航

Gallery 优先的两层导航结构：

```
NavigationStack
 └─ GalleryScreen (根，模型卡片网格)
      ├─ [Qwen3-0.6B 卡片]  → push
      └─ [Qwen3.5-0.8B 卡片] → push
                                  ↓
                          ChatScreen (每模型独立)
```

**重构职责分离**：

- **`ModelRegistry`**（静态）— 持有 `[ModelDescriptor]`，描述每个模型。Gallery 的数据源，加模型只改这一处。
- **`ChatViewModel`** — 按模型实例化（`ChatViewModel(model:)`），只管"加载这一个模型 + 这一轮对话 + 这一轮计时"。不再硬编码 Qwen3-0.6B。
- **`ChatSession` 计时结果** — 一次 `send()` 产出的 metrics，挂在对应 `ChatMessage` 上。

**数据流**: `GalleryScreen` 点卡片 → 用该 `ModelDescriptor` 构造 `ChatViewModel` → `ChatScreen(vm:)` → `.task` 里 `vm.loadModel()` → 用户输入 → `vm.send()` 流式生成 + 计时 → 每条回复带 `Metrics`。

**平台**: iOS 27 + macOS 27 双平台（见第 6 段）。Gallery 用自适应 `LazyVGrid`（iPhone 单列、iPad/Mac 多列）。

---

## 2. 模型注册表与 Gallery 卡片

**`ModelDescriptor`** 数据结构：

```swift
struct ModelDescriptor: Identifiable {
    let id: String                 // "qwen3-0.6b"
    let displayName: String        // "Qwen3-0.6B"
    let bundleName: String         // "qwen3_0_6b_4bit_dynamic"
    let quant: String              // "4bit · dynamic"
    let params: String             // "0.6B"
    let vocab: String              // "151,936"
    let romSize: String?           // "331 MB"
    let systemIcon: String         // "brain.head.profile"
    let accentColor: Color         // .purple
    let tagline: String            // "Fast · on-device"

    /// 按平台分支：iOS 加载 ios-gpu 变体，macOS 加载 macos 变体
    var pathCandidates: [String] {
        #if os(iOS)
        return ["Resources/Models/\(id)/ios-gpu", /* dev fallback */]
        #else
        return ["Resources/Models/\(id)/macos", /* dev fallback */]
        #endif
    }
}

enum ModelRegistry {
    static let all: [ModelDescriptor] = [ .qwen3_0_6b, .qwen3_5_0_8b ]
}
```

两个模型用不同主色 + 图标区分：0.6B 紫色、0.8B 蓝色。

**可用模型**:

| 模型 | 路径 | 特点 |
|------|------|------|
| Qwen3-0.6B (4bit dynamic) | `exports/qwen3_0.6b/` | 走 `LanguageModelSession` 流式路径 |
| Qwen3.5-0.8B (int8) | `qwen3.5-0.8B-CoreAI/{ios-gpu,macos}/` | 按平台选 ios-gpu / macos 变体 |

**`GalleryCard`** 卡片视觉（参考 Google AI Gallery）:

```
┌─────────────────────────────┐
│  ┌────┐                     │
│  │ 🧠 │  Qwen3-0.6B         │
│  └────┘  4bit · dynamic     │
│          0.6B · 151,936 vocab│
│                             │
│  ● Ready   331 MB  ~200 tok/s│
└─────────────────────────────┘
```

- `RoundedRectangle(cornerRadius: 20)` + `.ultraThinMaterial`
- 状态点：idle(灰) / loading(转圈) / ready(绿) / error(红)
- 标题行 + 量化/参数副标题 + 底部状态/ROM/预估速度

**`GalleryScreen`**: `LazyVGrid` 自适应列，顶部大标题 "Models"，`NavigationLink` 推入 `ChatScreen`。

---

## 3. 聊天界面与流式计时（核心）

**计时原理**（基于 `session.streamResponse`）:

```
send() 开始
  t0 = 时刻0
  promptTokens = model.tokenCount(prompt)        ← prefill 输入量
streamResponse 迭代:
  第1个 token 到达 → tTTF (time to first token)  ← prefill 结束
  第2..N个 token   → 累加 decode 计时
  tEnd = 最后一个 token 时刻
outputTokens = 累计 token 数

prefillMs  = tTTF - t0
decodeMs   = tEnd - tTTF
prefill tok/s = promptTokens / (prefillMs/1000)   ← 分子是输入 token
decode  tok/s = outputTokens / (decodeMs/1000)    ← 分子是输出 token
```

口径说明：`prefill tok/s` 分子是**输入** token 数，`decode tok/s` 分子是**输出** token 数 —— LLM 推理 benchmark 标准口径（与 model card ~70 tok/s 对得上）。

**`ChatViewModel` 改造**:

```swift
init(model: ModelDescriptor) { self.model = model }

func send() async {
    let t0 = ContinuousClock.now
    let promptTokens = (try? await model.tokenCount(text)) ?? estimated
    var outputTokens = 0
    var tTTF: Duration?
    let stream = session.streamResponse(to: prompt)
    for try await chunk in stream {
        if tTTF == nil { tTTF = elapsed(since: t0) }   // 首 token
        streamingText += chunk.content; outputTokens += 1
    }
    let total = elapsed(since: t0)
    // → 算出 prefill/decode tok/s，写进 ChatMessage.metrics
}
```

**UI 行为**: 流式期间显示带光标气泡（复用 `isStreaming`），临时小标签显示实时 decode tok/s。生成结束正式 `Metrics` 落到消息。

**降级**:
- `tokenCount` 不可用 → `字符数/3.5` 估算，标签旁标 `~`。
- `streamResponse` 报错 → 回退 `respond(to:)`，不显示速度（避免崩）。

---

## 4. 气泡下方的指标标签

**`ChatMessage` 增加可选 `Metrics`**:

```swift
struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    var content: String
    var metrics: Metrics?        // 仅 assistant 消息有
    let timestamp = Date()

    struct Metrics {
        let promptTokens: Int
        let outputTokens: Int
        let prefillTokPerSec: Double
        let decodeTokPerSec: Double
        let ttftMs: Int           // 首 token 延迟
    }
    enum Role { case user, assistant }
}
```

**`MetricsLabel`** 视图（紧凑、等宽、低对比度）:

```
Qwen3-0.6B
┌────────────────────────────┐
│ The capital of France is   │  ← 气泡本体
│ Paris.                     │
└────────────────────────────┘
 ▲ Prefill 12 tok · 69/s   ▼ Decode 8 tok · 71/s   ⚡184ms
```

- `font(.system(size: 10, weight: .medium, design: .monospaced))`
- `.foregroundStyle(.secondary)`，气泡下方 `padding(.top, 4)`
- **▲** 蓝色 prefill，**▼** 绿色 decode，**⚡** 橙色 TTFT
- token >1000 缩写 `1.5k`；tok/s 取整数；估算值带 `~`

**流式中**: 临时 `~ Decode N tok · XX/s`（实时跳动），首 token 前显示 `prefill…` 微动画。结束替换为正式 `MetricsLabel`。

**会话均值**: `ChatScreen` 顶部 / ModelCard 显示本会话最近 N 条消息的平均 prefill/decode tok/s。

---

## 5. 文件清单与改动

当前 6 文件 → 重构为（`Models/` 目录新增）:

| 文件 | 操作 | 内容 |
|------|------|------|
| `LLMChatApp.swift` | 改 | `@main` 下放，库内改为 `RootView` → `GalleryScreen` |
| **`Models/ModelRegistry.swift`** | 新增 | `ModelDescriptor` + `ModelRegistry.all`（2 模型，按平台分支路径） |
| `ViewModel/ChatViewModel.swift` | 改 | `init(model:)`、流式计时、`metrics`、会话均值 |
| **`Views/GalleryScreen.swift`** | 新增 | `LazyVGrid` + `NavigationLink` → `ChatScreen` |
| **`Views/GalleryCard.swift`** | 新增 | 单个模型卡片 |
| `Views/ContentView.swift` → **`ChatScreen.swift`** | 改名+改 | 聊天主界面，绑定 `ChatViewModel` |
| **`Views/MetricsLabel.swift`** | 新增 | 气泡下方指标标签 |
| `Views/ChatBubble.swift` | 改 | 挂载 `MetricsLabel` + 流中实时标签 |
| `Views/ModelCard.swift` | 改 | 聊天页顶部模型头，改用 `ModelDescriptor` 驱动 |
| `Views/SettingsSheet.swift` | 改 | 字段从 `vm.model`（descriptor）读取 |

**保留**: `CoreAILanguageModel` 加载、`LanguageModelSession`、`streamResponse`、`.dark` 主题、`Material`、跨平台条件编译。

---

## 6. 双平台工程结构

SwiftPM `executableTarget` 只能出 macOS 可执行文件，两个 Xcode App target 不能共用 executable。所以把共享代码抽成**库**，两个 App target 各自带壳。

**改造 `Package.swift`**（executable → library）:

```swift
products: [
    .library(name: "LLMChat", targets: ["LLMChat"]),  // .executable → .library
],
targets: [
    .target(name: "LLMChat", dependencies: [...])     // .executableTarget → .target
]
```

`LLMChatApp.swift` 的 `@main` 去掉，改成库内 `RootView`（指向 `GalleryScreen`）。`@main` 下放到各 App target。

**新增 Xcode 工程** `LLMChatApps.xcodeproj`（放 `app/ios/LLMChat/App/`）:

```
LLMChatApps.xcodeproj
├── LLMChat-macOS  (target)  ← imports 库 LLMChat
│     · @main App → RootView()
│     · Embed: macos/ 模型 bundle 进 Resources/Models
└── LLMChat-iOS    (target)  ← imports 库 LLMChat
      · @main App → RootView()
      · Embed: ios-gpu/ 模型 bundle 进 Resources/Models
```

每个 App target 入口极简:

```swift
@main struct LLMChatApp: App {
    var body: some Scene {
        WindowGroup { RootView().preferredColorScheme(.dark) }
    }
}
```

模型路径按平台分支见第 2 段 `pathCandidates`。两个 App 都先查 `Bundle.main`（各自嵌入的模型），再退化到 dev fallback。

**构建验证**:
- macOS: `xcodebuild -scheme LLMChat-macOS` 或 Xcode 直接跑
- iOS: `xcodebuild -scheme LLMChat-iOS -destination 'platform=iOS Simulator,...'`

**已知独立问题**: `swift build` 卡在 git `safe.bareRepository` 报错（拉 xgrammar 依赖时），与平台无关，实现阶段先修（`git config --global --add safe.directory` 或清缓存）。

---

## 一句话总结

**Gallery 网格选模型 → 进入按模型实例化的聊天页 → 流式生成 + TTFT 计时 → 每条回复气泡下方显示真实的 Prefill/Decode tok/s。macOS / iOS 双平台，共享库 + 两个 Xcode App target。**
