// swift-tools-version: 6.0
// Qwen3-TTS for MLX-Swift — TTS core (Talker + CodePredictor + SpeechTokenizer codec),
// audio utilities, and ICL voice cloning. Derived from soniqo/speech-swift (Apache-2.0).

import PackageDescription

let package = Package(
    name: "Qwen3TTS",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "Qwen3TTS", targets: ["Qwen3TTS"]),
        .library(name: "AudioCommon", targets: ["AudioCommon"]),
        .library(name: "Qwen3TTSCloning", targets: ["Qwen3TTSCloning"]),
    ],
    dependencies: [
        // Updated from 0.21.0 to 0.30.0 for stability improvements, race condition fixes,
        // and synchronization with Python MLX implementation
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        // Note: Source files originate from soniqo/speech-swift (swift-tools-version 5.9).
        // Swift 5 language mode avoids strict concurrency errors on upstream code.
        .target(
            name: "AudioCommon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/AudioCommon"
        ),
        .target(
            name: "Qwen3TTS",
            dependencies: [
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/Qwen3TTS"
        ),
        .target(
            name: "Qwen3TTSCloning",
            dependencies: [
                "Qwen3TTS",
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/Qwen3TTSCloning"
        ),
        .testTarget(
            name: "Qwen3TTSTests",
            dependencies: ["Qwen3TTS", "AudioCommon"],
            path: "Tests/Qwen3TTSTests"
        ),
        .testTarget(
            name: "Qwen3TTSCloningTests",
            dependencies: ["Qwen3TTSCloning"],
            path: "Tests/Qwen3TTSCloningTests"
        ),
    ],
    swiftLanguageModes: [.v5]
)
