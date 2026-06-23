// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "QwenTest",
    platforms: [.macOS("27.0")],
    products: [
        .executable(name: "qwen-run", targets: ["qwen-run"]),
    ],
    targets: [
        .executableTarget(
            name: "qwen-run",
            linkerSettings: [
                .linkedFramework("CoreAI"),
                .linkedFramework("Foundation"),
            ]
        ),
    ]
)
