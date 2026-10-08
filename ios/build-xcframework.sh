#!/bin/bash
# Build PhononCore.xcframework, the compiled Rust core the ptts Swift package wraps:
# iPhone, the iOS simulator and Apple-silicon Macs, each a static library with the C header.
#
# Output: ios/PhononTTS/PhononCore.xcframework
set -euo pipefail
cd "$(dirname "$0")/.."

TARGETS=(aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin)
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
PROFILE=release

rustup target add "${TARGETS[@]}" >/dev/null
for t in "${TARGETS[@]}"; do
  echo "building $t"
  cargo build --profile "$PROFILE" -p ptts-ffi --target "$t"
done

HEADERS="$(mktemp -d)"
cp ptts-ffi/include/ptts.h "$HEADERS/"
cat > "$HEADERS/module.modulemap" <<'MAP'
module PhononCore {
    header "ptts.h"
    export *
}
MAP

OUT=ios/PhononTTS/PhononCore.xcframework
rm -rf "$OUT"
args=()
for t in "${TARGETS[@]}"; do
  args+=(-library "$TARGET_DIR/$t/$PROFILE/libptts_ffi.a" -headers "$HEADERS")
done
xcodebuild -create-xcframework "${args[@]}" -output "$OUT" >/dev/null
rm -rf "$HEADERS"
du -sh "$OUT"/*/libptts_ffi.a

# The same framework zipped, with the checksum a client's Package.swift needs to fetch it by
# URL instead of by path: `.binaryTarget(name: "PhononCore", url: ..., checksum: ...)`.
ZIP=ios/PhononTTS/PhononCore.xcframework.zip
rm -f "$ZIP"
(cd ios/PhononTTS && zip -qry PhononCore.xcframework.zip PhononCore.xcframework)
echo "zip: $ZIP ($(du -h "$ZIP" | cut -f1)), checksum $(swift package compute-checksum "$ZIP")"
