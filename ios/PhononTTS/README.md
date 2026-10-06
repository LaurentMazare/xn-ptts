# PhononTTS

Phonon text to speech for your own iOS or macOS app. The model runs on the device, with its
transformer on the Apple Neural Engine, and audio streams as it is generated: on an iPhone 16 Pro
speech starts in under 40 ms and is produced 12 times faster than it plays.

```swift
import PhononTTS

let models = try PhononModels.install(bundled: Bundle.main.url(forResource: "Models", withExtension: nil)!)
let tts = try await Phonon.load(models: models, language: .english)
try await PhononPlayer().play(tts.stream("Hello! This is running entirely on the phone."))
```

It needs iOS 18 or macOS 15 on Apple silicon, Xcode 16, about 430 MB of disk for the models and
about 95 MB of memory while speaking. The simulator works, but without a Neural Engine it is
several times slower than a device.

## 1. Build the two pieces that are not in the source

From the root of a checkout, with Rust installed (or use the copies you were given):

```bash
./ios/build-xcframework.sh                                                          # PhononCore.xcframework
cargo run --release -p ptts --example export_coreml -- --dir "$MODEL_DIR" phonon-coreml   # the model, as Core ML
```

The first is the compiled core the package wraps, written into `ios/PhononTTS` where the package
expects it. The second converts the model folder you were given, the one with `config.json` and
`model.q8.gguf`. If that model takes conditions, such as `padding_bonus`, set them here with
`--condition padding_bonus=0.5` (repeatable): they are fixed in the exported folder, and those not
given take their defaults.

## 2. Add the package and the models

In Xcode, choose *File, Add Package Dependencies, Add Local*, select `ios/PhononTTS`, and add the
`PhononTTS` library to your app target. Then add the `phonon-coreml` folder to the target as a
folder reference (a blue folder, not a group) named `Models`.

Core ML compiles each model beside itself, so the models have to be copied out of the read-only
app bundle once. `PhononModels.install(bundled:)` does that, into Application Support and
excluded from backup, and does nothing when the installed copy is already current.

To keep the app small instead (the App Store asks before downloading an app over 200 MB on a
cellular connection), host the folder on any static server and download it on first run with
`PhononModels.download(from:progress:)`. Every file is checked against `bundle.json`, and an
interrupted download resumes.

## 3. Speak

Load once and keep the instance. The first load after an install or a model update compiles the
models for the device, about 10 s on an iPhone 16 Pro; later loads take about 0.6 s.
`Phonon.needsCompiling(_:)` tells you in advance which it will be.

| | |
|---|---|
| `tts.speak(text) { samples in ... }` | Calls the closure with each 80 ms chunk as it is generated, then returns `SpeechStats`. |
| `tts.stream(text)` | The same audio as an `AsyncThrowingStream<[Float], Error>`. |
| `tts.synthesize(text)` | All the samples at once. |
| `tts.voices`, `tts.setVoice(_:)` | The voices in the model folder, and switching between them (about 0.6 s). |
| `PhononPlayer` | Plays chunks as they arrive. `enqueue(_:)` and `stop()` work from any thread. |

Audio is mono Float32 at 24 kHz. Every method can be called from any thread or task, and an
instance speaks one utterance at a time. Cancelling the task that called `speak`, or ending a
`stream`, stops generation at the next chunk. Long text is split into sentences and spoken with
no gap.

`language` is required: numbers and symbols are spelled out before speaking, and how depends on
the language. `.none` speaks the text as written. `computeUnit: .cpu` is a slower fallback
whose audio differs slightly from the Neural Engine's, which computes in 16-bit floating point.

## Performance

On an iPhone 16 Pro, speaking a 2.3-second sentence:

| | Realtime factor | First audio |
|---|---|---|
| Neural Engine, a few seconds after the previous call | 12x | 37 ms |
| Neural Engine, calls back to back | 14.5x | 16 ms |
| CPU, a few seconds after the previous call | 10x | 55 ms |

The first two rows differ because iOS lowers the CPU's clock while an app is idle.

## Licensing

The code is MIT or Apache-2.0, like the rest of the repository. The Phonon model weights are not
part of this repository and are covered by the terms they were provided under.
