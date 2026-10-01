// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "PhononTTS",
    // iOS 18 / macOS 15: the models are Core ML 8 programs. Apple silicon only; the Neural
    // Engine is what makes this fast.
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "PhononTTS", targets: ["PhononTTS"]),
    ],
    targets: [
        // The compiled Rust core, from `ios/build-xcframework.sh`. A client given a release
        // zip instead would use `.binaryTarget(name:url:checksum:)` here.
        .binaryTarget(name: "PhononCore", path: "PhononCore.xcframework"),
        .target(
            name: "PhononTTS",
            dependencies: ["PhononCore"],
            linkerSettings: [
                .linkedFramework("CoreML"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("Accelerate"),
                .linkedFramework("AVFoundation"),
            ]
        ),
    ]
)
