# iOS and macOS

For contributors to the Swift package. Apps that want to use it should start with
[`PhononTTS/README.md`](PhononTTS/README.md).

| path | what |
|---|---|
| [`PhononTTS/`](PhononTTS/) | The Swift package: `Phonon`, `PhononPlayer`, `PhononModels`. |
| `build-xcframework.sh` | Builds `PhononTTS/PhononCore.xcframework`, the Rust core the package wraps, for iPhone, the simulator and Apple-silicon Macs, and a zip of it for releases. |
| [`../ptts-coreml-ffi/`](../ptts-coreml-ffi/) | The C interface the framework exports, and its header. |
| [`../ptts-coreml/`](../ptts-coreml/) | The Core ML graphs and the driver that runs them. |
| [`../ptts/examples/export_coreml.rs`](../ptts/examples/export_coreml.rs) | The exporter that turns a checkpoint into a model bundle. |

## Building

```bash
./ios/build-xcframework.sh                 # needs Xcode and a Rust toolchain
swift build --package-path ios/PhononTTS   # the package against the local framework
```

`ios/PhononTTS/Package.swift` points at the framework by path, so the package works from a
checked-out folder with the framework placed inside it.

## Preparing a delivery

Apps receive three things: the `ios/PhononTTS` folder, the framework and the model directory.

1. Build the framework with `./ios/build-xcframework.sh`, and put
   `ios/PhononTTS/PhononCore.xcframework` inside the delivered `PhononTTS` folder. The script
   also writes a zip and its checksum, for a later release that SwiftPM fetches by URL.
2. Export the models with
   `cargo run --release -p ptts --example export_coreml -- --dir <checkpoint> --voices <voices> <out-dir>`.
   Each export stamps `bundle.json`, so an app holding older models replaces them the next time
   it installs or downloads.

## Measuring on a device

The simulator has no Neural Engine, so speed can only be measured on a device, with an app on
the package. Two things to know when doing it:

- Measure with pauses between utterances, as a person tapping does. Back-to-back runs keep the
  CPU clocked up and read about 20% faster than an app feels.
- To force a fresh compile, change the models' `bundle.json`, by re-exporting, rather than
  uninstalling the app: uninstalling the last app signed by a free developer team also removes
  the phone's trust in that team.

## Checking a new checkpoint

The graphs are sized from the checkpoint's config, so any Pocket TTS checkpoint with a single
flow step should export. Compare the result against `ptts`'s own CPU path before relying on it:
with the same voice, text and seed, the first latent should agree to within a percent on the
CPU and a few percent on the Neural Engine, and the first frame of audio to better than 20 dB.
Later frames drift apart, because generation feeds its own output back in.
