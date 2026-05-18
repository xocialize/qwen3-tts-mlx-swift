// swift-tools-version: 6.0
// Qwen3-TTS — Internalized from qwen3-asr-swift (Qwen3TTS + AudioCommon targets only)

import PackageDescription

let package = Package(
    name: "Qwen3TTS",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "Qwen3TTS", targets: ["Qwen3TTS"]),
        .library(name: "AudioCommon", targets: ["AudioCommon"]),
    ],
    dependencies: [
        // Updated from 0.21.0 to 0.30.0 for stability improvements, race condition fixes,
        // and synchronization with Python MLX implementation
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        // Note: Source files originate from qwen3-asr-swift (swift-tools-version 5.9).
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
        .testTarget(
            name: "Qwen3TTSTests",
            dependencies: ["Qwen3TTS", "AudioCommon"],
            path: "Tests/Qwen3TTSTests"
        ),
    ],
    swiftLanguageModes: [.v5]
)
