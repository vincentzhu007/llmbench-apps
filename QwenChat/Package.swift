// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "QwenChat",
    platforms: [.iOS("27.0"), .macOS("27.0")],
    products: [
        .executable(name: "QwenChat", targets: ["QwenChat"]),
    ],
    dependencies: [
        .package(path: "../coreai-models")
    ],
    targets: [
        .executableTarget(
            name: "QwenChat",
            dependencies: [
                .product(name: "CoreAILM", package: "coreai-models"),
            ],
            linkerSettings: [
                .linkedFramework("CoreAI"),
                .linkedFramework("FoundationModels"),
                .linkedFramework("SwiftUI"),
            ]
        ),
    ]
)
