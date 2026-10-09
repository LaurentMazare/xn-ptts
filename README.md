# Phonon

**Streaming speech, on your device.**

Phonon brings natural text-to-speech to phones, laptops, browsers, and local services. It is built for offline assistants, accessibility tools, and interactive experiences that need speech without a cloud round trip. The runtime is written in Rust and needs no PyTorch.

[![Rust CI](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml/badge.svg?branch=main)](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml)
[![Python wheels](https://github.com/gradium-ai/xn-ptts/actions/workflows/maturin-pub.yml/badge.svg?branch=main)](https://github.com/gradium-ai/xn-ptts/actions/workflows/maturin-pub.yml)
[![Browser package](https://github.com/gradium-ai/xn-ptts/actions/workflows/npm-publish.yml/badge.svg?branch=main)](https://github.com/gradium-ai/xn-ptts/actions/workflows/npm-publish.yml)
[![Native packages](https://github.com/gradium-ai/xn-ptts/actions/workflows/cli-release.yml/badge.svg?branch=main)](https://github.com/gradium-ai/xn-ptts/actions/workflows/cli-release.yml)
[![Docker builds](https://github.com/gradium-ai/xn-ptts/actions/workflows/docker.yml/badge.svg?branch=main)](https://github.com/gradium-ai/xn-ptts/actions/workflows/docker.yml)
[![Code license: MIT OR Apache-2.0](https://img.shields.io/badge/code-MIT%20OR%20Apache--2.0-blue)](#license)
[![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white)](https://discord.gg/T85yt8kD33)

[![Published Rust version](https://img.shields.io/crates/v/ptts?label=crates.io)](https://crates.io/crates/ptts)
[![Published Python version](https://img.shields.io/pypi/v/ptts?label=PyPI)](https://pypi.org/project/ptts/)
[![Published npm version](https://img.shields.io/npm/v/phonon-tts?label=npm)](https://www.npmjs.com/package/phonon-tts)
[![Native downloads](https://img.shields.io/github/v/release/gradium-ai/xn-ptts?include_prereleases&label=native%20downloads)](https://github.com/gradium-ai/xn-ptts/releases)
[![Swift package](https://img.shields.io/badge/Swift-download-orange)](#swift)
[![Android QNN package](https://img.shields.io/badge/Android-QNN%20source%20preview-blue)](#android)

[What you get](#what-you-get) · [Choose an integration](#choose-an-integration) · [Packages and downloads](#packages-and-downloads) · [Quickstart](#quickstart) · [Guides](#guides)

> **Runtime preview available:** Python, Rust, browser packages, and native downloads are published. See the [runtime preview release](https://github.com/gradium-ai/xn-ptts/releases/tag/v0.4.0-rc.1) for installation instructions. The unversioned registry commands below target the upcoming stable release and currently select older packages. Phonon model files are available separately from the runtime packages. Docker public access is being finalized.

The browser runs synthesis on your device; it downloads model files on first use.

## What you get

- **Small enough to ship:** Choose a 40M or 90M parameter model. Both can ship inside a mobile app or load in a browser tab.
- **Five languages in one model:** Speak English, French, German, Spanish, and Portuguese, with more languages planned.
- **Fast across devices:** A custom inference stack runs in browsers and on phones, laptops, and embedded hardware, with Core ML on Apple devices, a QNN preview for supported Snapdragon NPUs, and GPU support. Multiple backends deliver faster inference than Kokoro and llama.cpp-based runtimes on the same hardware.
- **Voices ready to use or make your own:** Choose from ready-to-use voices, or explore voice design and cloning in [Gradium Studio](https://studio.gradium.ai/).
- **Control the delivery:** Adjust speaking speed and start playing streamed audio before the full utterance is ready.

## Performance

<!-- Performance measurements and methodology will go here. -->

## Choose an integration

| You want to… | Start here | What you need |
|---|---|---|
| Generate a speech file | [Command line](#command-line) | A desktop binary and a checkpoint. No Python or Rust required for a published binary. |
| Add speech to a script or backend | [Python](#python) | A Python wheel and a checkpoint. |
| Run speech in a web app | [Browser](#browser) | `phonon-tts` and model files served with your app. |
| Add speech to an iPhone or Mac app | [Swift](#swift) | The `ptts` Swift package and a prepared Core ML bundle. |
| Add speech to an Android app | [Android](#android) | A Kotlin AAR preview for supported Snapdragon NPUs, or the CPU native library. Source build and model files required. |
| Connect an existing app or self-host | [Docker and OpenAI-compatible API](#docker-and-openai-compatible-api) | Docker and a local checkpoint or HF repo. |
| Embed the runtime in Rust | [Rust](#rust) | The `ptts` crate and a checkpoint folder. |

Phonon is the product and model identity. The Rust, Python, Swift, and planned Android Maven packages are named **`ptts`**; the browser npm package is **`phonon-tts`**. This repository remains `xn-ptts`. Swift apps use `import PhononTTS`.

## Packages and downloads

Packages and native downloads are available as a runtime preview. Follow the [runtime preview release](https://github.com/gradium-ai/xn-ptts/releases/tag/v0.4.0-rc.1) to install the preview. The unversioned commands below are the installation paths for the upcoming stable release.

| Package | Get it | Details |
|---|---|---|
| Desktop `ptts` command | [GitHub Releases](https://github.com/gradium-ai/xn-ptts/releases) or [Homebrew tap](https://github.com/gradium-ai/homebrew-tap) | Linux x64/ARM64, Mac Apple silicon/Intel, and Windows x64 archives, with checksums. [Download guide and Homebrew](docs/cli.md). |
| Python `ptts` | [PyPI](https://pypi.org/project/ptts/) | Wheels for supported platforms; CPython 3.9+. [Python guide](ptts-pyo3/README.md). |
| Browser `phonon-tts` | [npm](https://www.npmjs.com/package/phonon-tts) | Worker, single-thread and threaded Wasm builds, and TypeScript declarations. [Browser guide](ptts-wasm/js/README.md). |
| Swift `ptts` | [GitHub Releases](https://github.com/gradium-ai/xn-ptts/releases) | `ptts-swift-<version>.zip` and its matching compiled framework. [Swift guide](ios/PhononTTS/README.md). |
| Android `ptts` (QNN preview) | [QNN Android guide](qnn/android/README.md) | Kotlin API and AAR for supported Snapdragon NPUs. Not published to Maven Central yet. |
| Android CPU library | [Build guide](android/README.md#1-build-the-library) | `libptts_ffi.so` and a Kotlin wrapper, built from source with Rust and the Android NDK. |
| `ptts-openai-server` | `ghcr.io/gradium-ai/ptts-openai-server:<version>` | CPU image for amd64 and arm64; model weights downloaded or mounted separately. [Server guide](ptts-openai-server/README.md). |
| Rust `ptts` | [crates.io](https://crates.io/crates/ptts) | Library API; add the `cli` feature to install the command. [Rust guide](ptts/README.md). |

Desktop downloads and Python wheels do not need a Rust compiler. Older x86 CPUs may need a source build: x86 wheels and desktop downloads target x86-64-v3. See the platform requirements in the [CLI guide](docs/cli.md) and [Python guide](ptts-pyo3/README.md).

## Model setup

Every integration needs an explicitly selected checkpoint. No model is chosen automatically.

The local examples below use a q8 checkpoint folder supplied to you, for example:

```text
model/
  config.json
  tokenizer.json
  model.q8.gguf
  default-voice.safetensors
```

Voice files may instead live in `voices/` or `embeddings/`. Use the files and voice names your checkpoint supplies; do not substitute another model's tokenizer or config. The examples use q8 weights. For f32 weights, omit `--quant q8` or `quant="q8"`, use `Quant::F32` in Rust, and set `quant: 'f32'` with `weights: { f32: '/model/model.safetensors' }` in the browser.

**Already received a model from us?** Extract it locally and use the matching integration below. You do not need a public Hugging Face repo or an HF token for local files. Runtime packages contain no model weights.

| Files you received | Use them with |
|---|---|
| A checkpoint with `config.json`, `tokenizer.json`, weights and voices | Command line, Python, Rust, browser, servers or Android CPU. Use a runtime version compatible with your checkpoint. |
| A prepared Core ML bundle | Swift. Add it to your app as `Models`, or host it for the Swift download API. A raw checkpoint needs [exporting first](ios/PhononTTS/README.md#1-build-the-two-pieces-that-are-not-in-the-source). |
| A QNN bundle with `metadata.json` and compiled context binaries | Android NPU. It must match the phone's SoC and the package's QNN runtime. Older bundles may need their target metadata updated; see the [preview guide](qnn/android/README.md). A raw checkpoint needs exporting and compiling first. |

For the shell examples below, set the folder once:

```sh
export MODEL_DIR=/absolute/path/to/model
```

The command line and Python can also acquire a checkpoint from Hugging Face. Use its repo ID and a fixed revision when you want a repeatable model version. Private repos require `HF_TOKEN` or a saved HF login. Local folders require no Hub access.

`lang` is required and selects **text normalization**, such as how numbers and symbols are spoken. Choose `en`, `fr`, `de`, `es`, `pt`, or `none` to pass text through unchanged. This setting does not establish which languages a checkpoint can speak.

## Quickstart

These examples use the published runtime preview and a supplied checkpoint. Install the preview using the [release instructions](https://github.com/gradium-ai/xn-ptts/releases/tag/v0.4.0-rc.1), or follow the [source setup](docs/development.md). The unversioned `uvx` and registry commands target the upcoming stable release.

### Command line

<details>
<summary>Show command line quickstart</summary>

After the stable release, [uv](https://docs.astral.sh/uv/) can run the Python command in an isolated environment. For the current preview, use the command in the [release instructions](https://github.com/gradium-ai/xn-ptts/releases/tag/v0.4.0-rc.1):

```sh
uvx ptts --model "$MODEL_DIR" --lang en --quant q8 "Hello from Phonon." -o speech.wav
```

`uvx` manages the Python environment and dependencies.

On Mac or Linux, install the prebuilt desktop command from Gradium's Homebrew tap. It currently installs the `0.4.0-rc.1` runtime preview:

```sh
brew install gradium-ai/tap/ptts
```

You can also extract a [desktop archive](docs/cli.md) and put `ptts` on your `PATH`. Neither path needs Python or Rust. Generate a WAV with a supplied checkpoint:

```sh
ptts --dir "$MODEL_DIR" --lang en --quant q8 "Hello from Phonon." -o speech.wav
```

The output is a mono 24 kHz WAV. Open it in your audio player.

If you prefer Cargo, install the command with `cargo install ptts --locked --features cli` after the matching stable release is published.

To download a checkpoint instead, replace the example repo and revision with your own:

```sh
ptts --repo OWNER/MODEL --revision COMMIT_SHA --lang en --quant q8 \
  "Hello from Phonon." -o speech.wav
```

Use `--voice NAME` to choose a voice; omitting it uses the checkpoint's default selection. Run `ptts --help` for the remaining options. [CLI guide →](docs/cli.md)

</details>

### Python

<details>
<summary>Show Python quickstart</summary>

For the stable release, install the Python package into your environment. Current preview users should follow the [release instructions](https://github.com/gradium-ai/xn-ptts/releases/tag/v0.4.0-rc.1):

```sh
python -m pip install ptts
```

Then save speech in four lines:

```python
import os
import ptts

tts = ptts.TTS(config=os.environ["MODEL_DIR"], lang="en", quant="q8")
tts.save("speech.wav", "Hello from Phonon.")
```

Reuse `tts`. `tts.synth(text)` returns a float32 NumPy array; `tts.stream(text)` yields audio chunks:

```python
with tts.stream("Speech can play while the rest is being generated.") as audio:
    for pcm in audio:
        # Send each chunk to your player at audio.sample_rate.
        print(pcm.shape)
```

Leaving the `with` block stops generation and releases its workers. For Hub loading, use `config="OWNER/MODEL"` and `revision="COMMIT_SHA"`.

The wheel also provides a command:

```sh
python -m ptts --model "$MODEL_DIR" --lang en --quant q8 \
  "Hello from Phonon." -o speech.wav
```

Rust and Python both install a command named `ptts`, with different flags. `python -m ptts` explicitly selects Python. [Python guide →](ptts-pyo3/README.md)

</details>

### Browser

<details>
<summary>Show browser quickstart</summary>

For the stable release, install the browser package into your web app. Current preview users should follow the [release instructions](https://github.com/gradium-ai/xn-ptts/releases/tag/v0.4.0-rc.1):

```sh
npm install phonon-tts
```

Serve the checkpoint's files under `/model/`, then generate a WAV:

```js
import { PhononTTS } from 'phonon-tts';

const tts = await PhononTTS.load({
  lang: 'en',
  model: {
    config: '/model/config.json',
    tokenizer: '/model/tokenizer.json',
    weights: { q8: '/model/model.q8.gguf' },
    voices: { default: '/model/default-voice.safetensors' },
  },
});

const wav = await tts.synthWav('Hello from Phonon.'); // WAV Blob
```

Use the returned `Blob` for a download link or your audio player. `tts.stream(text)` yields mono `Float32Array` chunks for streaming playback; the [browser guide](ptts-wasm/js/README.md#streaming) includes a complete Web Audio example. Call `tts.dispose()` when finished with the model.

The first load downloads and caches the files. `onProgress` reports download progress. Serve over HTTPS or localhost for caching, and use a browser with WebAssembly Relaxed SIMD. CPU is the default. Threading and WebGPU setup are covered in the [browser guide →](ptts-wasm/js/README.md).

</details>

### Swift

<details>
<summary>Show Swift quickstart</summary>

Extract `ptts-swift-<version>.zip`, add the folder to Xcode as a local package, and select the **`ptts`** library product. Swift Package Manager downloads the matching compiled framework. This path needs no Rust build.

Add the prepared Core ML model bundle to your app as a folder reference named `Models`:

```swift
import Foundation
import PhononTTS

guard let bundled = Bundle.main.url(forResource: "Models", withExtension: nil) else {
    throw PhononError(description: "Add the model folder to your app as Models.")
}
let models = try PhononModels.install(bundled: bundled)
let tts = try await Phonon.load(models: models, language: .english)
let player = try PhononPlayer()
try await player.play(tts.stream("Hello from Phonon."))
```

Keep `tts` and `player` in your app's state while audio plays. You can also download models on first run. Requires iOS 18+ or macOS 15+ on Apple silicon. The public Core ML model bundle is still being prepared. [Swift guide and Stop example →](ios/PhononTTS/README.md)

</details>

### Android

<details>
<summary>Show Android integration options</summary>

**Snapdragon NPU:** The [QNN Android preview](qnn/android/README.md) provides a Kotlin API and an AAR around the optimized QNN engine. It requires Android 12+, ARM64, working QNN HTP support and a model bundle compiled for the phone's SoC. Build the AAR from this repository using its build guide. The planned Maven package is `ai.gradium:ptts`; it is not published yet.

Load a supplied compiled model folder on a worker thread and stream audio to your player:

```kotlin
import ai.gradium.phonon.PhononTTS

val tts = PhononTTS.load(context, modelDirectory, lang = "en")
val sampleRate = tts.sampleRate
tts.speak("Hello from Phonon.") { pcm ->
    audioSink.write(pcm) // Mono float PCM at sampleRate.
    true // Return false to stop; tts.stop() also works from another thread.
}
// Reuse tts. Stop and wait for speech to finish before calling tts.close().
```

The preview guide covers native library extraction, model bundles and the runnable Speak/Stop example.

**CPU:** The existing [Android guide](android/README.md) provides a Kotlin wrapper and C API. From the repository root:

```sh
cargo install cargo-ndk
export ANDROID_NDK_HOME=/path/to/ndk
./android/build.sh
```

Copy `android/jniLibs` into your app's `src/main/`, then add the Kotlin wrapper and JNA dependency as described in the guide. Supply a checkpoint with its own config, tokenizer, weights and voices. Load `PhononTTS(modelDir, "en")` on a worker thread and reuse it. The guide includes AudioTrack playback, CPU requirements and callback cancellation.

</details>

### Docker and OpenAI-compatible API

<details>
<summary>Show Docker and API quickstart</summary>

Choose an image version from [GitHub Releases](https://github.com/gradium-ai/xn-ptts/releases) and replace `<release-version>` below, without the leading `v`. Public GHCR access is being finalized; until then use an image you can access or the [source setup](docs/development.md#docker). The Docker commands below use Bash or another POSIX shell. Mount your model folder and start the speech server:

```sh
PTTS_IMAGE="ghcr.io/gradium-ai/ptts-openai-server:<release-version>"
docker run --rm -p 127.0.0.1:8880:8880 -v "$MODEL_DIR:/models:ro" \
  -e PTTS_CONFIG=/models -e PTTS_LANG=en -e PTTS_QUANT=q8 \
  "$PTTS_IMAGE"
```

In a second terminal:

```sh
curl http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from Phonon.","voice":"default","response_format":"wav"}' \
  -o speech.wav
```

For applications with an OpenAI-compatible TTS setting, use `http://localhost:8880/v1` as the base URL and a voice your checkpoint supports. The [server guide](ptts-openai-server/README.md) includes client setup recipes, supported formats, and limits.

For HF acquisition and a cache that survives container replacement, use the [Hub Compose file](ptts-openai-server/compose.hub.yaml). Supply the repo, revision, and normalization language explicitly. The image contains no model weights. Deployment and access controls are in the [server guide →](ptts-openai-server/README.md).

</details>

### Rust

<details>
<summary>Show Rust quickstart</summary>

Add `ptts` with tokenizer support to your Rust project. This command targets the upcoming stable release; preview installation is in the [release instructions](https://github.com/gradium-ai/xn-ptts/releases/tag/v0.4.0-rc.1), and source development can use a [path dependency](docs/development.md#rust):

```sh
cargo add ptts --features hf
```

```rust
use ptts::checkpoint::{Checkpoint, ResolveOptions};
use ptts::preprocess::Lang;
use ptts::synth::{DeviceKind, Quant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = Checkpoint::resolve(
        std::env::var("MODEL_DIR")?,
        ResolveOptions { quant: Quant::Q80, weights: None },
    )?;
    let mut tts = checkpoint.builder(Lang::En).device(DeviceKind::Cpu).build()?;
    checkpoint.register_voices(&mut tts);
    let pcm = tts.say("Hello from Phonon.")?;
    ptts::wav::write_wav_file("speech.wav", &pcm, tts.sample_rate())?;
    Ok(())
}
```

Build with `--release` and reuse the model. `tts.stream(text)?` yields audio chunks for your own playback. The library reads local files; Hub acquisition belongs to the caller. [Rust API and small example →](ptts/README.md)

</details>

## Guides

| Topic | Guide |
|---|---|
| Desktop downloads, Homebrew, CPU requirements, CLI flags | [Command line](docs/cli.md) |
| Python installation, voices, streaming, and wheels | [Python](ptts-pyo3/README.md) |
| Browser playback, download progress, caching, threads, and WebGPU | [Browser](ptts-wasm/js/README.md) |
| Apple installation, model bundles, playback, and cancellation | [Swift](ios/PhononTTS/README.md) |
| Android CPU library, Kotlin, NDK, and playback | [Android](android/README.md) |
| Android Snapdragon NPU AAR, Kotlin API, and compiled bundles | [QNN Android](qnn/android/README.md) |
| Docker, OpenAI-compatible clients, and deployment | [HTTP server](ptts-openai-server/README.md) |
| Streaming text and audio over one connection | [WebSocket server](ptts-ws-server/README.md) |
| Rust library and the `say` example | [Rust](ptts/README.md) |
| Build packages, run the local demo, and develop from source | [Development](docs/development.md) |

### Platform notes

Native desktop packages cover the targets listed in the [CLI guide](docs/cli.md). Apple apps use Core ML. Android apps can use the CPU library or the QNN AAR preview on supported Snapdragon NPUs. QNN requires a matching compiled bundle and has no automatic CPU fallback. Browser apps use Wasm on CPU by default. Browser WebGPU is opt in. The npm package targets browsers, not native Node.js inference.

Performance and memory use depend on the checkpoint and device.

### Questions and feedback

Report bugs or request integrations through [GitHub Issues](https://github.com/gradium-ai/xn-ptts/issues). Include your runtime version, device/OS, model revision, weight format, normalization language, and a small reproduction.

## Acknowledgements

Phonon builds on [Pocket TTS](https://github.com/kyutai-labs/pocket-tts), developed by Kyutai, and uses the [xn](https://github.com/LaurentMazare/xn) Rust tensor runtime. Gradium's Phonon checkpoints are the primary target. Compatible Pocket TTS checkpoints use the same explicit loading interface when they supply their own config, tokenizer JSON, weights, and voices.

## License

The code is **MIT OR Apache-2.0**, at your option: [MIT](LICENSE-MIT) · [Apache-2.0](LICENSE-APACHE).

Model weights and voices are distributed separately and have their own licenses. Check the model license before redistribution.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this repository is dual licensed as above, without additional terms or conditions.
