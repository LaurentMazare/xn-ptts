# iOS and macOS

For contributors. Apps start with [`PhononTTS/README.md`](PhononTTS/README.md).

| path | what |
|---|---|
| [`PhononTTS/`](PhononTTS/) | The Swift package: `Phonon`, `PhononPlayer`, `PhononModels`. |
| `build-xcframework.sh` | Builds `PhononTTS/PhononCore.xcframework` from `ptts-coreml-ffi`, for iPhone, the simulator and Apple-silicon Macs, plus a zip and its checksum for a future URL-based release. |
| [`../ptts-coreml-ffi/`](../ptts-coreml-ffi/) | The C interface the framework exports. `include/ptts.h` is written by hand and must match `src/lib.rs`. |
| [`../ptts-coreml/`](../ptts-coreml/) | The Core ML graphs and the driver that runs them. |
| [`../ptts/examples/export_coreml.rs`](../ptts/examples/export_coreml.rs) | Converts a checkpoint into a model bundle. Without `--dir` it downloads `kyutai/pocket-tts`. |

```bash
./ios/build-xcframework.sh
swift build --package-path ios/PhononTTS
```

Speed can only be measured on a device, with an app on the package: the simulator has no Neural
Engine. Measure with pauses between utterances, since back-to-back runs keep the CPU clocked up
and read about 20% faster than an app feels. To force a fresh compile, re-export the models
rather than uninstalling the app: uninstalling the last app signed by a free developer team also
removes the phone's trust in that team.

A new checkpoint should be compared against `ptts`'s CPU path before it is relied on: with the
same voice, text and seed, the first latent should agree to within a percent on the CPU and a few
percent on the Neural Engine. Later frames drift apart, because generation feeds its output back.
