// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "scry",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "Scry", targets: ["Scry"]),
        .executable(name: "scry", targets: ["scry-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.5.0"),
    ],
    targets: [
        .target(
            name: "Scry",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Hummingbird", package: "hummingbird"),
            ]
        ),
        .executableTarget(
            name: "scry-cli",
            dependencies: [
                "Scry",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
    ]
)
