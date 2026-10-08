// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "ptts",
    // iOS 18 / macOS 15: the models are Core ML 8 programs. Apple silicon only; the Neural
    // Engine is what makes this fast.
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "ptts", targets: ["PhononTTS"]),
    ],
    targets: [
        // Source builds use the local compiled core. Release packaging replaces this target
        // with its versioned download URL and the checksum of that exact archive.
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
