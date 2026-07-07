# Qwen3-4B 适配记录（macOS / iOS）

把 HuggingFace 的 **Qwen3-4B**（dense，36 层，hidden 2560）经 Apple Core AI 导出为 `.aimodel`，并跑进 LLMChat app。macOS 顺利；**iOS 上踩了一串坑**（ANE 编译 bug、内存爆、上下文长度下限、entitlements），本文记录全过程与最终可用配置，便于复现。

> 主线模型仍是仓库早期的 Qwen3.5（hybrid SSM，见 `qwen3_5_overlay.py`）。Qwen3-4B 走的是 **coreai 自带的 dense qwen3 overlay**，和 hybrid 那条无关。

---

## 0. 关键事实（结论先行）

| 项 | 值 |
|----|----|
| 模型 | Qwen3-4B（dense，标准全注意力，非 hybrid）|
| 导出工具 | `coreai.llm.export`（coreai_models 自带 CLI），**不是** `export_qwen35_4b_stateful.py`（那是 hybrid 3.5 的）|
| macOS 量化 | per-block INT4 对称（`--compression 4bit`，torch 预导出），非 palettized |
| iOS 量化 | **同 macOS 的 4bit build**（见下文为什么 iOS 专属量化不可用）|
| 运行后端 | **GPU**（`CoreAIPipelinedEngine`，动态 shape 模型自动选）|
| macOS context | 40960（默认）|
| iOS context | **4096**（40960 在 8GB 手机上 OOM，4096 是下限之上的最小可用值）|
| iOS 内存 | 加了 `increased-memory-limit` + `extended-virtual-addressing` entitlement |
| 实测 | macOS 4B ~28-34 tok/s；iPhone 17 Pro 可加载并对话 |

---

## 1. 架构选择：dense，不是 hybrid

下载的是 `Qwen/Qwen3-4B`（`config.json`: `model_type: qwen3`，36 层 dense 全注意力）。仓库自带的 `qwen3_5_overlay.py` 是给 **Qwen3.5（hybrid SSM + 全注意力）**重写的，**套不到 dense Qwen3 上**（Qwen3 没有 SSM 层 / conv_state / rec_state，权重对不上）。

正解：用 coreai_models 内置的 dense overlay
[`llm-engines/apple-core-ai/coreai-models/python/src/coreai_models/models/macos/qwen3.py`](../llm-engines/apple-core-ai/coreai-models/python/src/coreai_models/models/macos/qwen3.py)
里的 `Qwen3ForCausalLM`，通过 CLI `coreai.llm.export` 调用（registry 里已有 `qwen3-4b` 预设）。

---

## 2. 导出管道（macOS，一次成功）

### 2.1 装核心库
`coreai_models` 以 editable 方式装，`.pth` 要指向本地源码：
```
/Users/zgd/Tool/Miniconda3/lib/python3.12/site-packages/_editable_impl_coreai_models.pth
→ llm-engines/apple-core-ai/coreai-models/python/src
```
（移动 coreai-models 后要同步改这个 `.pth`，否则 `import coreai_models` 断。）

### 2.2 本地权重导出包装脚本
`coreai.llm.export` 把本地目录当 HF repo id 传给 `snapshot_download` 会校验失败。包装脚本
[`llm-engines/apple-core-ai/export_local_coreai.py`](../llm-engines/apple-core-ai/export_local_coreai.py)
做两件事再透传所有 CLI 参数：
- patch `huggingface_hub.snapshot_download / hf_hub_download`：本地目录直接返回
- patch `torch._check`：兼容 torch 2.9 vs 2.11 签名

### 2.3 macOS 导出命令
```bash
python3 llm-engines/apple-core-ai/export_local_coreai.py /Users/zgd/Code/llm/model/Qwen/Qwen3-4B \
  --platform macOS --compute-precision float16 --experimental \
  --output-dir llm-engines/apple-core-ai/exports \
  --output-name qwen3_4b_macos_4bit --overwrite
```
- 裸路径（非 registry 短名）需 `--experimental --compute-precision float16`
- 产物 `qwen3_4b_macos_4bit/`（2.1 GB，`kind: llm`，含 `.aimodel` + `metadata.json` + `tokenizer/`）
- 非 palettized per-block INT4 → 走 GPU（`EngineFactory` 对动态 shape 模型选 Pipelined），bench 实测 ~28-34 tok/s

---

## 3. iOS：一串坑与解法

iOS 4B 是最难的部分。依次遇到并解决：

### 3.1 iOS palettized 导出 → kmeans 爆内存（Mac 导出阶段）
iOS 默认量化是 **kmeans palettization**（`4bit_weight_palettized_group32`）。
[`compression.py`](../llm-engines/apple-core-ai/coreai-models/python/src/coreai_models/export/compression.py)
里 **`num_workers=32` 硬编码**，每个 worker（spawn）各自加载整个 8GB 模型 → 32×8=256GB，24GB Mac 直接 OOM 被杀。

**修法**：改成环境变量可调（默认 2）：
```python
num_workers = int(os.environ.get("COREAI_KMEANS_WORKERS", "2"))
```
导出时 `COREAI_KMEANS_WORKERS=1`（串行，~12 分钟，峰值 ~4GB）。> 此补丁在 coreai-models 包内（gitignored），重 clone 后要重打。

### 3.2 iOS palettized 模型 → 设备加载时 ANE 编译崩溃
导出成功的 palettized 包，在 iPhone 加载时 **SIGABRT**，崩在：
```
ANECompiler mlir::anecir::fillPalettizedKernelInfo
← MPSToANECValidation ← ANERegionFormationPass
← MPSGraphExecutable specializedModuleWithDevice
```
**根因**：MPSGraph 给设备编译时，把 palettized 算子往 ANE（神经网络引擎）上放，**iOS 27 beta 的 ANE 编译器对 palettized LLM 权重有 bug**，`fillPalettizedKernelInfo` 抛 C++ 异常 → abort。**不是 OOM、不是我们代码。**

### 3.3 非 palettized iOS 导出 → overlay 签名不兼容
想绕开 palettization，用 macOS 那个 per-block INT4（`--compression 4bit`）+ `--platform iOS`：
- coreai 默认禁止：`macOS quantization preset provided, but platform is iOS`（patch 掉这个检查后…）
- 真正卡点：iOS overlay `Qwen3ForCausalLMForiOS.forward()` 要求 `key_cache`/`value_cache`（有状态签名），标准 CLI 的 example_inputs 给不了 → `TypeError`。

即 **iOS overlay 只能走有状态/pipelined 导出流程**，标准 `coreai.llm.export` 导不出非 palettized iOS 包。

### 3.4 解法：iOS 复用 macOS 的非 palettized 4bit build
关键洞察：**非 palettized 权重不会被 MPSGraph 往 ANE 上塞**（无 `fillPalettizedKernelInfo` 路径）→ 在 iOS GPU 上正常加载。所以 iOS 直接用 macOS 那个 4bit build（`qwen3_4b_macos_4bit`），绕开 ANE bug + overlay 导出问题。

`ModelDescriptor` 改成平台分支（`macOSBundleName` + `iOSBundleName`），iOS 指向同一个非 palettized build。

### 3.5 40960 context → 手机加载 OOM
非 palettized build 在 iPhone 加载又崩，这次是 **`operator new` 抛 `std::bad_alloc`**：
```
MPSGraph BumpMmapResourceAllocator::allocateResource ← allocateBufferTensorBlob
← MPSGraphPackage readResources ← MPSGraphDelegate.resolveCoreAI
```
**根因**：4B 默认 `max_context_length=40960`，KV cache 预分配好几 GB，加上 2.1GB 权重，**8GB iPhone 分配不出**。Mac 24GB 没事。

### 3.6 缩 context：512/2048 都不行，4096 可用
模型 context 维度 **min=2048**（且动态 Dim 要求 `max > min`）：
- `--max-context-length 512` → `AssertionError: Dim inconsistent min=2048, max=512`（512 < 下限）
- `--max-context-length 2048` → `min=2048, max=2048`（min==max 不是合法动态 Dim）
- `--max-context-length 4096` → ✓（KV cache 相比 40960 缩 ~10×，到 ~0.5-0.6GB）

```bash
python3 llm-engines/apple-core-ai/export_local_coreai.py /Users/zgd/Code/llm/model/Qwen/Qwen3-4B \
  --platform macOS --compute-precision float16 --experimental \
  --max-context-length 4096 \
  --output-dir llm-engines/apple-core-ai/exports \
  --output-name qwen3_4b_macos_4bit_ctx4096 --overwrite
```

### 3.7 内存 entitlement：抬高 app 上限
即便 4096 ctx（峰值 ~3GB），仍贴着 iPhone 普通 app ~4-5GB 上限。给 iOS target 加 entitlement
[`apps/ios/LLMChat/App/LLMChat-iOS.entitlements`](../apps/ios/LLMChat/App/LLMChat-iOS.entitlements)：
- `com.apple.developer.kernel.increased-memory-limit`：jetsam 上限抬到 ~5-6GB
- `com.apple.developer.kernel.extended-virtual-addressing`：扩大虚拟地址空间，缓解大块连续内存分配失败（即上面的 bad_alloc）

`project.yml` 里 `CODE_SIGN_ENTITLEMENTS: LLMChat-iOS.entitlements`，自动签名时一并签入。

> iPhone 单 app 实际内存上限不公开，8GB 机型普通 app ~4-5GB，加 entitlement 可到 ~5-6GB+；且 `bad_alloc` 可能在 jetsam 阈值之下就发生（连续大块拿不到），extended-virtual-addressing 正好治这个。

### 3.8 iOS 文件共享侧载
iPhone 看不到 Mac 上的模型，需 Finder 侧载：
- iOS Info.plist 要有 `UIFileSharingEnabled`（用真 `Info-iOS.plist`，不是 `GENERATE_INFOPLIST_FILE` —— 这个 key 没 `INFOPLIST_KEY_` 映射）
- `ModelDescriptor.resolveBundleURL()` 查 `Documents/<bundleName>`（侧载目录）
- USB 连 iPhone → Finder → 文件 → 把 bundle 文件夹拖进 LLMChat

---

## 4. 最终可用配置

**macOS**：`qwen3_4b_macos_4bit`（40960 ctx，4bit，GPU，dev 路径回退加载）

**iOS**：`qwen3_4b_macos_4bit_ctx4096`（4096 ctx，4bit，非 palettized，GPU）+ 内存 entitlements，侧载进 Documents。

`ModelRegistry` 里 4B 条目：
```swift
macOSBundleName: "qwen3_4b_macos_4bit",
iOSBundleName:   "qwen3_4b_macos_4bit_ctx4096",
```

---

## 5. 踩坑速查表

| 现象 | 根因 | 解法 |
|------|------|------|
| `import coreai_models` 失败 | editable `.pth` 指向移动前旧路径 | 改 `.pth` 指到 `llm-engines/apple-core-ai/coreai-models/python/src` |
| `snapshot_download` 校验本地路径失败 | coreai.llm.export 不认本地目录 | 用 `export_local_coreai.py` 包装（patch snapshot_download）|
| 导出阶段 Mac OOM | kmeans `num_workers=32` 各加载 8GB | `COREAI_KMEANS_WORKERS=1`（改 compression.py）|
| iOS 加载 SIGABRT（ANECompiler.fillPalettizedKernelInfo）| iOS 27 beta ANE 对 palettized 权重编译 bug | 用**非 palettized** 的 per-block INT4 build（走 GPU）|
| iOS 非 palettized 导出 TypeError（key_cache/value_cache）| iOS overlay 有状态签名 | 别走 iOS overlay；直接用 macOS 的非 palettized build |
| iOS 加载 bad_alloc（MPSGraph allocateResource）| 40960 ctx KV cache 太大，8GB 不够 | `--max-context-length 4096`（下限 2048 之上最小）|
| 4096 仍贴内存上限 | iPhone app 内存上限低 | 加 `increased-memory-limit` + `extended-virtual-addressing` entitlement |
| Finder 看不到 app 文件 | `UIFileSharingEnabled` 没进 Info.plist | 用真 `Info-iOS.plist`（该 key 无 INFOPLIST_KEY 映射）|

---

## 6. 未做 / 后续

- **2bit/3bit 更小量化**：coreai-opt 的 `FakeQuantize` 支持显式 `n_bits`（可写自定义 `--compression-config` 做 non-palettized 3bit，~1.6GB），但 sub-byte 权重在 GPU 上会 unpack 成 int8 算（省的是存储/加载内存，算力不省），且 2bit 质量明显下降。4096 ctx + entitlement 已够用，暂未做。
- **iOS palettized 真正可用**：等 Apple 修 iOS 27 ANE 的 palettized 编译 bug 后，palettized group32/8 才能在手机上跑（更省、可能上 ANE）。
- **pipelined 引擎 / 真 prefill-decode 拆分**：见 `docs/plans/2026-06-24-llmchat-gallery-design.md` 补遗（B 路线）。

---

## 7. 相关文件 / 提交

- 导出包装：[`llm-engines/apple-core-ai/export_local_coreai.py`](../llm-engines/apple-core-ai/export_local_coreai.py)
- 平台分支注册表：[`apps/ios/LLMChat/Sources/LLMChat/Models/ModelRegistry.swift`](../apps/ios/LLMChat/Sources/LLMChat/Models/ModelRegistry.swift)
- iOS 工程签名 + entitlements：[`apps/ios/LLMChat/App/project.yml`](../apps/ios/LLMChat/App/project.yml)、[`LLMChat-iOS.entitlements`](../apps/ios/LLMChat/App/LLMChat-iOS.entitlements)、[`Info-iOS.plist`](../apps/ios/LLMChat/App/Info-iOS.plist)
- 本地补丁（coreai-models 内，gitignored）：
  - `compression.py`：`COREAI_KMEANS_WORKERS`（kmeans 并行度）
  - `llm/export.py`：允许 macOS 4bit preset 用在 iOS（探索性，最终没用上）
- 主要 commit：`b9082c4`（4B macOS + wrapper）、`8725682`（per-platform bundles）、`c875e06`（iOS 用非 palettized）、`c1087c3`（4096 ctx）、`223e66d`（内存 entitlements）
