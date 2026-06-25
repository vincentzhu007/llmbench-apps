// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LLMChat",
    platforms: [.iOS("27.0"), .macOS("27.0")],
    products: [
        .library(name: "LLMChat", targets: ["LLMChat"]),
        .executable(name: "LLMChatRunner", targets: ["LLMChatRunner"]),
        .executable(name: "LLMChatBench", targets: ["LLMChatBench"]),
    ],
    dependencies: [
        .package(path: "../../../apple-core-ai/coreai-models")
    ],
    targets: [
        // Shared UI library: Gallery, chat screen, streaming timing, metrics.
        // Imported by the Xcode app targets (see App/project.yml) and the
        // SwiftPM runners below.
        .target(
            name: "LLMChat",
            dependencies: [
                .product(name: "CoreAILM", package: "coreai-models"),
            ],
            path: "Sources/LLMChat",
            linkerSettings: [
                .linkedFramework("CoreAI"),
                .linkedFramework("FoundationModels"),
                .linkedFramework("SwiftUI"),
            ]
        ),

        // macOS SwiftUI app host, for `swift run LLMChatRunner` during dev.
        .executableTarget(
            name: "LLMChatRunner",
            dependencies: ["LLMChat"],
            path: "Sources/LLMChatRunner",
            linkerSettings: [
                .linkedFramework("CoreAI"),
                .linkedFramework("FoundationModels"),
                .linkedFramework("SwiftUI"),
            ]
        ),

        // Headless bench: loads each model, runs one stream, prints prefill /
        // decode tok/s. Verifies the timing pipeline without a GUI.
        .executableTarget(
            name: "LLMChatBench",
            dependencies: ["LLMChat"],
            path: "Sources/LLMChatBench",
            linkerSettings: [
                .linkedFramework("CoreAI"),
                .linkedFramework("FoundationModels"),
            ]
        ),
    ]
)
