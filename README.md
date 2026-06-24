# llmbench-apps

> 在 Apple 设备上本地运行大语言模型（LLM）的工作区：基于 Apple **Core AI**（`.aimodel`，iOS 27 / macOS 27）的端侧推理应用、模型转换管道，以及相关的分析与文档。

本仓库是一个「端侧 LLM 实验台」，把 HuggingFace 上的开源模型转换成 Apple Core AI 的 `.aimodel` 包，并在 SwiftUI 应用里通过 Apple 的 `FoundationModels` / `CoreAI` 框架加载、推理。所有计算在本地完成（GPU / Apple Neural Engine），不依赖云端。

---

## 目录

- [它解决什么问题](#它解决什么问题)
- [仓库结构](#仓库结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [核心组件](#核心组件)
  - [LLMChat — SwiftUI 聊天应用](#llmchat--swiftui-聊天应用)
  - [模型转换管道（Python）](#模型转换管道python)
  - [coreai-model-zoo 子模块](#coreai-model-zoo-子模块)
- [让应用加载你自己的模型](#让应用加载你自己的模型)
- [已知限制：Chunked Prefill](#已知限制chunked-prefill)
- [相关链接](#相关链接)

---

## 它解决什么问题

Apple 在 iOS 27 / macOS 27 引入了全新的 **Core AI** 运行时与 `.aimodel` 包格式，用于在端侧高效运行生成式模型。但要把一个 HuggingFace 模型真正跑起来，需要打通三件事：

1. **转换**：把 PyTorch / HF 模型用 Apple 复合算子（`RMSNorm`、`RoPE`、`SDPA`、`GatedDeltaUpdate` 等）重写，导出为动态 shape 的 `.aimodel`。
2. **运行时**：在 Swift 里通过 `CoreAILM` 加载包，并接入 Apple 的 `LanguageModelSession` 做对话。
3. **应用**：一个能加载模型、管理对话、调节生成参数的 UI。

本仓库把这三件事放在一个可复现的工作区里，并以 **Qwen3.5**（混合 SSM + 全注意力架构）作为主线示例。

---

## 仓库结构

```
llmbench-apps/
├── LLMChat/                       # Gallery 风格 SwiftUI 聊天应用（iOS 27 / macOS 27）
│   ├── Package.swift              # SwiftPM：LLMChat 库 + LLMChatRunner(macOS) + LLMChatBench
│   ├── Sources/LLMChat/           # 共享 UI 库
│   │   ├── RootView.swift         # 库入口 → GalleryScreen
│   │   ├── Models/ModelRegistry.swift   # 模型描述 + 路径解析（按平台）
│   │   ├── ViewModel/ChatViewModel.swift# 按模型实例化 + 流式 + throughput 计时
│   │   └── Views/                 # GalleryScreen/Card · ChatScreen · ChatBubble · MetricsLabel · ModelCard · SettingsSheet
│   ├── Sources/LLMChatRunner/     # macOS @main App（swift run）
│   ├── Sources/LLMChatBench/      # headless bench，打印每个模型 throughput
│   └── App/project.yml            # xcodegen：macOS + iOS 双 App target 工程描述
│
├── export_qwen35_4b_stateful.py   # 导出 Qwen3.5-4B → 有状态 .aimodel（fp16 / int8）
├── qwen3_5_overlay.py             # Qwen3.5 模型覆写层（hybrid SSM + full-attention）
│
├── docs/
│   ├── chunked-prefill.md          # Chunked prefill 支持状态与限制分析
│   └── plans/                      # 设计文档
│
├── coreai-model-zoo/              # git submodule：参考生态（模型卡 / 转换脚本 / 知识库）
├── .gitmodules
└── README.md
```

> 说明：仓库历史上的 `QwenChat/`、`QwenTest/`、`FMTest/` 以及若干早期 Python 脚本已从工作区移除，统一整合为现在的 `LLMChat` 应用 + 转换脚本结构。

---

## 环境要求

**运行 SwiftUI 应用**

- **Xcode 27**（Swift 6.4 + iOS 27 / macOS 27 SDK，含 `CoreAI` 框架）
- 目标设备：iOS 27 或 macOS 27
- 兄弟依赖：需在**同级目录**有 `coreai-models` 运行时库（应用通过 `Package.swift` 中的 `.package(path: "../coreai-models")` 引用它）

**运行模型转换管道**

- Python 3.10+
- PyTorch（2.9 / 2.11，脚本内含 `torch._check` 兼容补丁）
- `coreai_models`、`coreai_torch`（提供覆写原语与 `export_to_coreai`）
- `transformers`、`huggingface_hub`

---

## 快速开始

### 1. 构建并运行 LLMChat 应用

```bash
# 0) 先准备同级目录的 coreai-models 运行时（见「环境要求」）
git clone --recurse-submodules <本仓库> && cd llmbench-apps/LLMChat

# 1a) macOS：直接用 SwiftPM 运行（推荐开发用法）
swift run LLMChatRunner

# 1b) macOS / iOS：用 xcodegen 生成 Xcode 工程
cd App && xcodegen generate && open LLMChatApps.xcodeproj
#   在 Xcode 里选 LLMChat-macOS 或 LLMChat-iOS scheme 运行

# 1c) headless 验证每个模型的 throughput
swift run LLMChatBench
```

应用首页是模型卡片网格（Gallery），点一张卡片进入该模型的聊天页；每条回复气泡下方显示 `⚡ 吞吐 tok/s · 📥 prompt→output token · TTFT`。

### 2. 导出一个 Qwen3.5-4B 模型包

```bash
# fp16（默认）
python export_qwen35_4b_stateful.py \
    --hf-path /path/to/Qwen/Qwen3.5-4B \
    --mode fp16 \
    --out-dir exports

# int8 线性量化
python export_qwen35_4b_stateful.py \
    --hf-path /path/to/Qwen/Qwen3.5-4B \
    --mode int8lin \
    --full          # 导出全部 32 层（默认可截断用于调试）
```

产物：`exports/qwen3_5_4b_stateful_{mode}/`，包含 `*.aimodel`、`metadata.json` 和 `tokenizer/`。

---

## 核心组件

### LLMChat — Gallery 风格 SwiftUI 聊天应用

`LLMChat/` 是一个 **Gallery 优先**的 SwiftUI 应用：首页是模型卡片网格，点卡片进入该模型的独立聊天页。架构为**共享 UI 库**（`Sources/LLMChat`）+ 多个 App 宿主（macOS `swift run` 宿主、xcodegen 生成的 macOS/iOS App target、headless bench）。

- **模型注册表**：`Models/ModelRegistry.swift` 用 `ModelDescriptor` 描述每个模型（id / bundle 路径 / 量化 / 参数 / 词表 / 主色），`resolveBundleURL()` 按平台查找 bundle（App 内嵌 → Documents → 开发回退路径）。加模型只改这一处。
- **按模型实例化**：`ChatViewModel(model:)` 只负责"加载这一个模型 + 当前对话 + 计时"，不再硬编码模型信息。
- **流式 + 计时**：`send()` 用 `session.streamResponse` 边生成边显示；从**总墙钟时间 + `Usage` 真实 token 数**算出可靠 throughput（`prompt+output tok / 总时间`）。
- **指标展示**：每条 assistant 气泡下挂 `MetricsLabel`（throughput / token 数 / TTFT），流中显示实时 `LiveMetricsLabel`，顶部 `SessionStatsBar` 显示会话均值。
- **生成参数**：`SettingsSheet` 调 Temperature / Max Tokens，读自当前 descriptor。

> **速度口径说明**：CoreAI 的 `streamResponse` 会缓冲输出（snapshot 批量到达），`LanguageModelSession` 路径拿不到 prefill/decode 拆分；且已发布的 bundle 都是 **S=1** 导出（`COREAI_CHUNK_THRESHOLD=1`，否则 prefill 多 token 触发形状断言崩溃），S=1 下 prefill≈decode≈总吞吐。因此 App 显示**可靠的总吞吐**（实测 0.6B ~175 tok/s、0.8B ~70 tok/s，与 model card 吻合）。要拿真实的 prefill/decode 拆分需接 `CoreAIPipelinedEngine`（仅 0.8B 有 `prefill_b2048` 导出），作为后续增强（见设计文档补遗）。

### 模型转换管道（Python）

两个脚本配合，把 HuggingFace 上的 Qwen3.5-4B 转成有状态、动态 shape 的 Core AI 包：

| 脚本 | 作用 |
|------|------|
| `qwen3_5_overlay.py` | Qwen3.5 的**模型覆写层**。用 `coreai_models` 原语重写网络：每隔 4 层为带 GQA + QK-norm + RoPE + KV cache 的**全注意力层**，其余为基于 `coreai_torch.GatedDeltaUpdate` 复合算子的 **SSM（线性注意力）层**。有状态前向，4 个状态：`k_cache`、`v_cache`、`conv_state`、`rec_state`。 |
| `export_qwen35_4b_stateful.py` | **导出脚本**。加载覆写层 →（可选）`int8lin` 块量化 → `export_to_coreai` 导出动态 shape IR → `optimize()` → 保存为 `.aimodel` 并生成 `metadata.json` + tokenizer。仅 `k_cache`/`v_cache` 作为真正的 CoreAI 状态（带 slice_update），`conv_state`/`rec_state` 作为常规输入输出。 |

主要参数：

- `--mode fp16|int8lin`：精度
- `--hf-path`：本地 HF 模型目录或仓库 id
- `--max-ctx`：最大上下文长度（默认 4096）
- `--num-layers N`：截断到 N 层（调试）；`--full` 导出全部 32 层

### coreai-model-zoo 子模块

`coreai-model-zoo/` 是一个 git submodule（`john-rocky/coreai-model-zoo`），作为**参考生态**：

- `conversion/` — 各类模型（Qwen3.x、Gemma 4、GLM-4.7、LFM2、MiniCPM-V、RF-DETR 等）的导出脚本
- `knowledge/` — Core AI 相关的深度文档（pipelined engine、有状态 KV cache、量化、自定义 Metal 内核、compute units 等）
- `apps/` — 示例应用与相关 patch
- `zoo/` — 各模型的 model card（含实测吞吐数据）

克隆时记得带上子模块：

```bash
git clone --recurse-submodules <本仓库>
# 若已克隆，补齐子模块：
git submodule update --init --recursive
```

---

## 让应用加载你自己的模型

应用通过 `ModelDescriptor.resolveBundleURL()` 按以下顺序查找 bundle：

1. `Bundle.main/Models/<bundleName>`（App 内嵌）
2. App 的 `Documents/Models/<bundleName>`
3. 开发回退路径（`devPath` 绝对路径）

加模型只需在 `LLMChat/Sources/LLMChat/Models/ModelRegistry.swift` 的 `ModelRegistry.all` 里追加一个 `ModelDescriptor(...)`（填 `bundleName`、量化、词表、主色、`devPath` 指向你的 `exports/<你的模型>/`），Gallery 网格就会多出一张卡片。

---

## 已知限制：Chunked Prefill

目前所有已发布的 Qwen3.5 Core AI bundle 都是 **decode-only、loop-free** 导出——`input_ids` 被固定为静态 `[1, 1]`，因此 Apple `InferenceEngine` 原生支持的 **chunked prefill 不可用**。

根因：Qwen3.5 的 GatedDeltaNet 层用 `scf.while` 循环做 SSM 时序扫描，而在 macOS 27 beta 的 MPSGraph GPU delegate 上 `scf.while` 无法 lower（"region type mismatch"）。导出时改用 `use_loopfree_step=True`（S=1 时与 while_loop 数值等价）可绕过，但代价就是 `input_ids` 必须静态 `[1, 1]`。

详见 [docs/chunked-prefill.md](docs/chunked-prefill.md)（含 Apple 官方方案、报错、实测吞吐对比、启用条件）。

---

## 相关链接

- Apple **Core AI** / `FoundationModels` 框架（iOS 27 / macOS 27）
- 子模块生态：[coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)
- 模型来源：[Hugging Face](https://huggingface.co/)（如 Qwen3.5 系列）

---

*本仓库面向 iOS 27 / macOS 27 beta；API 与导出管道可能随 beta 演进。*
