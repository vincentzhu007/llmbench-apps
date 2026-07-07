import Foundation
import SwiftUI

/// Describes one on-device model the Gallery can load and chat with.
///
/// Each descriptor maps to a Core AI model bundle (a directory containing
/// `metadata.json` + `<bundleName>.aimodel` + `tokenizer/`), loadable via
/// `CoreAILanguageModel(resourcesAt:)`. A model can have separate macOS and
/// iOS bundle builds (different compression/quantization per platform);
/// `bundleName` resolves to the right one for the current platform. Bundle
/// resolution checks the app bundle first, then side-loaded Documents, then
/// an absolute dev-path fallback (macOS development only).
public struct ModelDescriptor: Identifiable, Hashable, Sendable {
    public let id: String
    public let displayName: String
    public let macOSBundleName: String
    public let iOSBundleName: String?       // nil = no iOS build (macOS-only model)
    public let quant: String                // e.g. "4bit"
    public let params: String               // e.g. "0.6B"
    public let vocab: String                // e.g. "151,936"
    public let romSize: String?             // e.g. "331 MB"
    public let systemIcon: String           // SF Symbol
    public let accentColor: Color
    public let tagline: String
    public let estimatedDecodeTokPerSec: Int   // model-card hint shown before first run
    public let devPath: String             // macOS dev-machine fallback (ignored on iOS device)

    /// The bundle name to look for on the current platform.
    public var bundleName: String {
        #if os(iOS)
        return iOSBundleName ?? macOSBundleName
        #else
        return macOSBundleName
        #endif
    }

    public init(
        id: String,
        displayName: String,
        macOSBundleName: String,
        iOSBundleName: String? = nil,
        quant: String,
        params: String,
        vocab: String,
        romSize: String?,
        systemIcon: String,
        accentColor: Color,
        tagline: String,
        estimatedDecodeTokPerSec: Int,
        devPath: String
    ) {
        self.id = id
        self.displayName = displayName
        self.macOSBundleName = macOSBundleName
        self.iOSBundleName = iOSBundleName
        self.quant = quant
        self.params = params
        self.vocab = vocab
        self.romSize = romSize
        self.systemIcon = systemIcon
        self.accentColor = accentColor
        self.tagline = tagline
        self.estimatedDecodeTokPerSec = estimatedDecodeTokPerSec
        self.devPath = devPath
    }

    /// Resolve the bundle directory for this model on the current platform.
    /// Returns the first existing candidate, or nil if none are present
    /// (the Gallery card then shows an "unavailable" state instead of crashing).
    public func resolveBundleURL() -> URL? {
        let documents = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)
            .first
        let candidates: [URL] = [
            // 1. Shipped inside the app bundle (real distribution, both platforms)
            Bundle.main.bundleURL.appendingPathComponent("Models/\(bundleName)"),
            // 2. Side-loaded into the app's Documents via Finder file sharing:
            //    either flat (Documents/<bundle>) or under a Models/ subfolder.
            documents?.appendingPathComponent(bundleName) ?? URL(fileURLWithPath: "/dev/null"),
            documents?.appendingPathComponent("Models/\(bundleName)") ?? URL(fileURLWithPath: "/dev/null"),
            // 3. Dev fallback: absolute path on the dev machine (macOS only)
            URL(fileURLWithPath: devPath),
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }
}

public enum ModelRegistry {
    /// The real, loadable on-device models. Add a new `.init(...)` here to
    /// grow the Gallery. Models with an iOS build set `iOSBundleName`; others
    /// are macOS-only (their card shows "Not found" on iOS until exported).
    public static let all: [ModelDescriptor] = [
        ModelDescriptor(
            id: "qwen3-0.6b",
            displayName: "Qwen3-0.6B",
            macOSBundleName: "qwen3_0_6b_4bit_dynamic",
            quant: "4bit · dynamic",
            params: "0.6B",
            vocab: "151,936",
            romSize: "331 MB",
            systemIcon: "brain.head.profile",
            accentColor: .purple,
            tagline: "Fast · on-device",
            estimatedDecodeTokPerSec: 200,
            devPath: "/Users/zgd/Code/llm/llmbench-apps/llm-engines/apple-core-ai/exports/qwen3_0.6b/qwen3_0_6b_4bit_dynamic"
        ),
        ModelDescriptor(
            id: "qwen3.5-0.8b",
            displayName: "Qwen3.5-0.8B",
            macOSBundleName: "qwen3_5_0_8b_decode_int8hu_perchan_sym",
            quant: "int8 · per-channel",
            params: "0.8B",
            vocab: "248,320",
            romSize: "1.2 GB",
            systemIcon: "cpu",
            accentColor: .blue,
            tagline: "Higher quality · int8",
            estimatedDecodeTokPerSec: 70,
            devPath: "/Users/zgd/Code/llm/llmbench-apps/llm-engines/apple-core-ai/qwen3.5-0.8B-CoreAI/gpu-pipelined/qwen3_5_0_8b_decode_int8hu_perchan_sym"
        ),
        ModelDescriptor(
            id: "qwen3-4b",
            displayName: "Qwen3-4B",
            macOSBundleName: "qwen3_4b_macos_4bit",
            // iOS uses a reduced-context (4096) variant of the non-palettized
            // 4bit build. The 40960-context build OOMs on an 8GB iPhone (the KV
            // cache alone is several GB); 4096 shrinks the KV cache ~10× so the
            // 4B fits. Non-palettized per-block INT4 → runs on GPU, avoids the
            // iOS 27 beta ANE palettization compiler crash.
            iOSBundleName: "qwen3_4b_macos_4bit_ctx4096",
            quant: "4bit",
            params: "4B",
            vocab: "151,936",
            romSize: "2.1 GB",
            systemIcon: "sparkles",
            accentColor: .teal,
            tagline: "Larger · higher quality",
            estimatedDecodeTokPerSec: 35,
            devPath: "/Users/zgd/Code/llm/llmbench-apps/llm-engines/apple-core-ai/exports/qwen3_4b_macos_4bit"
        ),
    ]
}
