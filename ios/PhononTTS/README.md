# PhononTTS

Add Phonon text to speech to your own iOS or macOS app. The model runs on the device, with its
transformer on the Apple Neural Engine, and audio streams while it generates: speech starts
within a few tens of milliseconds and is produced 12 to 24 times faster than it plays. Nothing
leaves the device, and nothing needs a network once the models are installed.

```swift
import PhononTTS

let models = try PhononModels.install(bundled: Bundle.main.url(forResource: "Models", withExtension: nil)!)
let tts = try await Phonon.load(models: models, language: .english)
let player = try PhononPlayer()
try await player.play(tts.stream("Hello! This is running entirely on the phone."))
```

## What you need besides this package

Two things that are not in the source, both built from a checkout of this repository with Rust
installed, or used as supplied if you were given them:

- **`PhononCore.xcframework`**, the compiled core the package wraps, for iPhone, the iOS
  simulator and Apple-silicon Macs. From the repository root:

  ```bash
  ./ios/build-xcframework.sh
  ```

  It lands in `ios/PhononTTS/PhononCore.xcframework`, where the package expects it.
- **The model, converted to Core ML**: about 430 MB of Core ML packages, voices, tokenizer and a
  `bundle.json` describing them. Convert the model folder you were given, the one with
  `config.json` and `model.q8.gguf`:

  ```bash
  cargo run --release -p ptts --example export_coreml -- --dir "$MODEL_DIR" phonon-coreml
  ```

## Requirements

- iOS 18 or macOS 15, on Apple silicon. The models are Core ML 8 programs.
- Xcode 16.
- About 430 MB of disk for the models, and about 95 MB of memory while speaking.

The iOS simulator works for development, but it has no Neural Engine, so speech there is several
times slower than on a device.

## 1. Add the package

With `PhononCore.xcframework` inside the `PhononTTS` folder, next to `Package.swift`, in
Xcode choose *File, Add Package Dependencies, Add Local*, select the `PhononTTS` folder, and add
the `PhononTTS` library to your app target. Or from another `Package.swift`:

```swift
.package(path: "path/to/PhononTTS"),
```

The package links Core ML, Core Video, Accelerate and AVFoundation for you.

## 2. Install the models

Core ML compiles each model for the device beside the model itself, so the models must be in a
writable directory, not inside the read-only app bundle. `PhononModels` puts them in
`Application Support/PhononTTS/models`, excluded from iCloud backup, in one of two ways.

**Ship them in the app.** Add the model directory to your target as a folder reference (a blue
folder, not a group) named `Models`, and copy it out once:

```swift
let models = try PhononModels.install(bundled: Bundle.main.url(forResource: "Models", withExtension: nil)!)
```

This copies only when the bundled models differ from the installed ones, so it is cheap to call
at every launch.

**Or download them on first run** from a static file server you host, with the directory's
layout kept intact:

```swift
let models = try await PhononModels.download(from: URL(string: "https://your-host.example/phonon/")!) { fraction in
    // Called as bytes arrive, on a background thread.
}
```

Every file is checked against the size and SHA-256 listed in `bundle.json`. An interrupted
download resumes without fetching finished files again, and new models replace the installed
ones only once they are complete. When the same models are already installed, the call returns
at once. Downloading keeps the app small, which matters: the App Store asks before downloading
an app over 200 MB on a cellular connection.

## 3. Speak

```swift
let tts = try await Phonon.load(models: models, language: .english)
```

Load once, when your app starts or before speech is first needed, and keep the instance.
**The first load after an install or a model update takes longer**, about 10 s on an iPhone 16 Pro
and 20 s on an M5 Mac, while Core ML compiles the models for the device. It happens once; later
loads take about 0.6 s.
`Phonon.needsCompiling(models)` tells you in advance, so the app can say why it is waiting.

| | |
|---|---|
| `tts.speak(text) { samples in ... }` | Calls the closure with each 80 ms chunk of audio as it is generated, then returns `SpeechStats`. |
| `tts.stream(text)` | The same audio as an `AsyncThrowingStream<[Float], Error>`. |
| `tts.synthesize(text)` | All the samples at once. |
| `tts.voices`, `tts.voice`, `tts.setVoice(_:)` | The voices in the release, the one speaking, and switching. |
| `PhononPlayer` | Plays chunks as they arrive. `enqueue(_:)` and `stop()` work from any thread. |

Audio is mono Float32 at 24 kHz, `Phonon.sampleRate`.

- **Stopping.** Cancel the task that called `speak`, or stop iterating a `stream`, and generation
  stops at the next chunk. `player.stop()` silences what was already queued.
- **Threading.** Every method can be called from any thread or task. An instance speaks one
  utterance at a time; overlapping calls wait their turn. The `speak` closure runs on a
  background thread, in order.
- **Long text** is split at sentence ends and spoken sentence by sentence, with no gap.
- **Language is required.** Numbers, symbols and abbreviations are written out before the text
  is spoken, and how depends on the language: German normalized as English would read `@` as
  "at". Pass `.none` to speak the text exactly as written.
- **Voices.** Switching voice takes up to about 0.6 s, after which the next utterance starts at
  full speed.
- **Compute unit.** `Phonon.load(models:language:computeUnit:)` takes `.neuralEngine`, the
  default, or `.cpu`, a slower fallback whose audio differs slightly because the Neural Engine
  computes in 16-bit floating point.

## Performance

Measured on an iPhone 16 Pro, speaking a 2.3-second sentence:

| | Realtime factor | First audio |
|---|---|---|
| Neural Engine, a few seconds after the previous call | 12x | 37 ms |
| Neural Engine, calls back to back | 14.5x | 16 ms |
| CPU, a few seconds after the previous call | 10x | 55 ms |

Memory stays at about 93 MB while speaking, 96 MB at its peak.

The first two rows differ because iOS lowers the CPU's clock while an app is idle, so the first
second of an utterance runs slower. Other devices are not measured yet.

## Troubleshooting

- **The first load is slow.** That is the one-time compile after an install or a model update.
- **Loading fails with a file error.** The models must be in a writable directory. Use
  `PhononModels.download` or `PhononModels.install` rather than a path inside the app bundle.
- **Much slower than the numbers above.** Check that you are on a device, not the simulator, and
  that it is not hot: `ProcessInfo.processInfo.thermalState` above `.nominal` means it is
  throttling.

## Other checkpoints

The package runs any Pocket TTS checkpoint with a single flow step, once exported. To export the
published [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts) checkpoint, or your own,
from a checkout of this repository:

```bash
cargo run --release -p ptts --example export_coreml -- out/pocket-tts-coreml
```

`--dir` exports a local checkpoint instead of downloading one, and `--voices` takes a directory
of voice embeddings, such as the `create_voice` example writes. Host the output directory and
pass its URL to `PhononModels.download(from:)`.

Building the framework from source, for development on the package itself, is
`./ios/build-xcframework.sh`; [`ios/README.md`](../README.md) has the details. Apps written in C,
C++ or for Unity can use the C interface the framework exports,
[`ptts-coreml-ffi`](../../ptts-coreml-ffi/include/ptts.h), directly.

## Licensing

The code is MIT or Apache-2.0, like the rest of the repository. The Core ML schema compiled into
the framework is Apple's, under the BSD 3-clause license. The Phonon model weights are not part of
this repository and are covered by the terms they were provided under.
