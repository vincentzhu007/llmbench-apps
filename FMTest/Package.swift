// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "FMTest",
    platforms: [.macOS("27.0")],
    products: [
        .executable(name: "fm-test", targets: ["fm-test"]),
    ],
    dependencies: [
        .package(path: "../coreai-models")
    ],
    targets: [
        .executableTarget(
            name: "fm-test",
            dependencies: [
                .product(name: "CoreAILM", package: "coreai-models"),
            ],
            linkerSettings: [
                .linkedFramework("CoreAI"),
                .linkedFramework("FoundationModels"),
                .linkedFramework("Foundation"),
            ]
        ),
    ]
)
