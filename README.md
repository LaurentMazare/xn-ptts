# Phonon

Phonon is Gradium's on-device text-to-speech runtime, written in Rust, with Python bindings. It builds on [Pocket TTS](https://github.com/kyutai-labs/pocket-tts), developed by Kyutai. This preview pairs the code in this repository with a model package supplied by Gradium; the model is not in this repository.

Gradium's Phonon checkpoints are the primary integration target and use their own model config, weights, tokenizer, and voices. Pocket TTS checkpoints can be used when they supply a compatible `config.json`, `tokenizer.json`, and weights. The release will make the selected Phonon checkpoint the default.

[![Rust CI](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml)

## 1. Set up

For the source installation below, you need [Rust](https://rustup.rs), and [uv](https://docs.astral.sh/uv/) for Python.

Point `MODEL_DIR` at the model folder, the one holding `config.json`, `model.q8.gguf`, `tokenizer.json` and its voice assets:

```bash
export MODEL_DIR=/path/to/model
```

The CLI, Rust examples, Python, and both servers share the [checkpoint resolver](ptts/src/checkpoint.rs). Supply a local model directory or an HF repo explicitly. Each model supplies its own `config.json`, `tokenizer.json`, weights, and voice assets. No model or model config is selected automatically.

## 2. Run it

Install the `ptts` command from the repository root (the first build takes a few minutes):

```bash
cargo install --path ptts --locked --features cli
ptts --lang en --dir "$MODEL_DIR" --quant q8 "Hello world" -o out.wav
```

Both Rust and Python install a command named `ptts`, with different flags. `PATH` order selects which one runs. To select the Python CLI explicitly in its environment, use `python -m ptts`; the `uv run --project` command below selects the project's Python command.

With Python, from the repository root (the first run builds the package, a few minutes):

```bash
uv run --project ptts-pyo3 --locked ptts --lang en \
  --model "$MODEL_DIR" --quant q8 "Hello world" -o out.wav
```

`--quant q8` runs the model in q8, the format `model.q8.gguf` is stored in. The Rust and Python examples below set q8 too. Loading q8 weights as f32 expands them, which is slower and uses more memory. `--lang` is required. It picks how numbers, symbols and abbreviations are spelled out before synthesis: `en`, `fr`, `de`, `es` or `pt`, or `none` to use the text as written.

The Rust CLI also accepts `--repo <owner/model>` and an optional `--revision <commit>` instead of `--dir`. For a private repo, set `HF_TOKEN` or log in with the Hugging Face CLI. Run `ptts --help` for voice, device, and generation options. The [CLI guide](docs/cli.md) covers desktop downloads and platform requirements.

When no voice is specified, native integrations use the checkpoint's configured default, then `default`, then the first registered voice by name. Swift uses its exported bundle's voice selection. For a fixed choice, pass `--voice Freya` to either CLI, `voice="Freya"` to Python, or call `tts.setVoice("Freya")` in Swift.

## 3. Use it from Rust

Add the crate from your checkout as a path dependency:

```toml
[dependencies]
ptts = { path = "/path/to/xn-ptts/ptts", features = ["hf"] }
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

    let pcm = tts.say("Hello world")?;
    ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate())?;
    Ok(())
}
```

Load the model once and reuse it. `tts.say` returns the whole waveform as mono `f32` samples at `tts.sample_rate()`. `tts.stream(text)?` is an iterator of `Result<Vec<f32>>` chunks, yielded as they are generated, for playback that starts before the sentence is finished. Select a registered voice through `SpeechOptions::voice`; the checkpoint resolver discovers its file. Build with `--release`: a debug build is far too slow for realtime.

Long text is grouped at sentence boundaries, usually aiming for 50 text tokens per chunk. A single sentence may exceed that target. Sentences over 200 tokens are split at a sentence or clause mark when possible, otherwise at a word boundary. An indivisible piece over 200 tokens returns an input error before the KV cache is allocated. Each spoken piece starts again from the selected voice prompt.

For repeated requests in one voice, `tts.session_default(&SpeechOptions::default())?` keeps a primed session with a budget calculated from that voice and the configured chunk target. Use `tts.session(&options, max_seq_len)?` when you need to set the KV budget yourself.

## 4. Use it from Python

Install the package from your checkout into your project. This compiles the Rust code, so it needs Rust installed:

```bash
uv add /path/to/xn-ptts/ptts-pyo3      # or: pip install /path/to/xn-ptts/ptts-pyo3
```

```python
import os
import ptts

model = os.environ["MODEL_DIR"]
tts = ptts.TTS(lang="en", config=model, quant="q8")

tts.save("out.wav", "Hello world")    # write a 16-bit WAV
pcm = tts.synth("Hello world")        # float32 NumPy array at tts.sample_rate
with tts.stream("A longer sentence, played as it is generated.") as audio:
    for chunk in audio:
        ...                           # each chunk is a float32 NumPy array
```

Load the model once and reuse it. The [Python README](ptts-pyo3/README.md) covers voices and the remaining options.

## 5. Use it in an iOS or macOS app

The `ptts` Swift package exposes the `PhononTTS` module and runs the model on the device through Core ML, with its transformer on the Apple Neural Engine: about 12 times faster than realtime on an iPhone 16 Pro, with first audio in under 40 ms. It needs iOS 18 or macOS 15, and Xcode. The [Swift package guide](ios/PhononTTS/README.md) describes the prepared release package and source installation.

Build the package's compiled core and convert the model to Core ML, both from the repository root:

```bash
./ios/build-xcframework.sh
cargo run --release -p ptts --example export_coreml -- --dir "$MODEL_DIR" phonon-coreml
```

Then add `ios/PhononTTS` to your Xcode project as a local package, add the `phonon-coreml` folder to your app as a folder reference named `Models`, and speak:

```swift
import Foundation
import PhononTTS

guard let bundled = Bundle.main.url(forResource: "Models", withExtension: nil) else {
    throw PhononError(description: "Add the exported model folder to your app as a folder reference named Models.")
}
let models = try PhononModels.install(bundled: bundled)
let tts = try await Phonon.load(models: models, language: .english)
let player = try PhononPlayer()
try await player.play(tts.stream("Hello world"))
```

Keep `tts` and `player` in your app or view state while speech is playing. `play` returns when generation finishes; scheduled audio can still be playing.

The [package README](ios/PhononTTS/README.md) covers downloading the models instead of bundling them, voices, and the rest of the API.

## 6. Use it in the browser

The `phonon-tts` JavaScript package runs the model in the page, compiled to WebAssembly, in a Web Worker: on the CPU by default, or on the GPU through WebGPU when asked with `device: 'webgpu'` or `'auto'`. Build it from the repository, which needs Rust with the `wasm32-unknown-unknown` target, [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/), Node 22.7 or later, [binaryen](https://github.com/WebAssembly/binaryen/releases) 124 or later, and a pinned nightly toolchain for the package's multithreaded build, which `make threads-toolchain` installs:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
brew install binaryen           # or a release from GitHub: distribution packages are often older than 124

cd ptts-wasm
make threads-toolchain          # once
make build                      # the package, in ptts-wasm/pkg
cd pkg && npm pack              # and as a tarball, phonon-tts-<version>.tgz
```

Install the tarball into your web app, and serve the model folder with the app's static files, here under `/model/`:

```bash
npm install /path/to/xn-ptts/ptts-wasm/pkg/phonon-tts-*.tgz
```

Install the tarball rather than the `pkg` folder: npm links a folder instead of copying it, and Vite's dev server refuses to serve files from outside the app.

```js
import { PhononTTS } from 'phonon-tts';

const tts = await PhononTTS.load({
  lang: 'en',
  model: {
    weights: { q8: '/model/model.q8.gguf' },
    tokenizer: '/model/tokenizer.json',
    config: '/model/config.json',
    voices: {
      Freya: '/model/voices/Freya.safetensors',
      Harper: '/model/voices/Harper.safetensors',
      Sterling: '/model/voices/Sterling.safetensors',
      Toby: '/model/voices/Toby.safetensors',
    },
    defaultVoice: 'Freya',
  },
});

for await (const pcm of tts.stream('Hello from the browser.')) {
  // mono Float32Array chunks of 80 ms at tts.sampleRate, as they are generated
}
const wav = await tts.synthWav('Hello world');   // or a whole WAV Blob
```

Load the model once and reuse it. The first load downloads the model files and keeps them in the browser's Cache API, which needs the page served over `https://` or from `localhost`. Files are cached by URL, so when you replace the model, serve it under a new path (say `/model-v2/`) or call `clearCache()` first; otherwise the browser keeps using the old files. The browser needs WebAssembly Relaxed SIMD; this was tested in current Chrome. Bundlers such as Vite pick up the package's worker and wasm with no configuration. The [package README](ptts-wasm/js/README.md) covers streaming playback, voices and the remaining options.

`tts.device` says whether it runs on `'webgpu'` or `'cpu'`. On the CPU, generation runs on 3 threads when the page is served with these two headers, and on one thread otherwise. Pass `threads` to `load` to choose another number:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

With them, the page can only load cross-origin files that opt in through CORS, which matters if the model is served from another origin. `tts.threads` says how many threads it got, and `tts.threadsReason` why.

## 7. Run it as a server

`ptts-openai-server` serves OpenAI's text-to-speech API, `POST /v1/audio/speech`, so any client with an "OpenAI TTS" setting and a custom base URL can use it. Its Docker image runs on the CPU, for `linux/amd64` and `linux/arm64`. Mount the model folder into it:

```bash
docker run -p 8880:8880 -v "$MODEL_DIR:/model:ro" -e PTTS_CONFIG=/model \
  ghcr.io/gradium-ai/ptts-openai-server

curl http://localhost:8880/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "Freya"}' -o hello.mp3
```

The image contains no model weights. Supply a mounted model folder or an HF repo through `PTTS_CONFIG`. The [server README](ptts-openai-server/README.md) covers running it without Docker, the API, and setup for clients such as Open WebUI and Home Assistant. For streaming text in and audio out over one WebSocket connection, there is `ptts-ws-server`.

Both servers keep fixed audio queues through generation and encoding, so a slow reader applies backpressure. Disconnecting stops generation and releases its workers. WebSocket sessions accept up to 4096 pending text characters between flushes and 64 KiB per message. Once 16 requests are queued, the server pauses reading until generation catches up. A socket write stalled for 30 seconds closes the session.

## Updating from earlier builds

Streaming generation now uses bounded buffers across the Rust API and native bindings. Leaving a stream unread pauses generation once its buffers fill; it resumes when you consume audio.

Model sources are now required: use `--repo` or `--dir` in the Rust CLI, `config=` in Python, `--config` or `PTTS_CONFIG` for servers, and an explicit `ModelSpec` in the browser. The Docker image contains no model weights.

Every checkpoint must supply its own config and tokenizer JSON. Built-in configs, Pocket TTS presets, legacy filenames, and `ptts-model.json` support have been removed. The Rust manifest types and `TTSConfig::v202601()` and the browser's `POCKET_TTS_MODEL` export are no longer available. Move custom artifact paths and voice selection to the caller's options; manifest checksums are no longer checked by the runtime.

`Quant::check_device` now rejects backends that were not compiled into the runtime as well as incompatible weight formats. Call it before acquiring model files to fail before downloading.

For a SentencePiece-only checkpoint, [convert its tokenizer to JSON](scripts/convert-tokenizer.py) once before loading it. This is an explicit preparation tool; the runtime reads only the supplied tokenizer JSON.

## License

The code in this repository is licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. The model weights are not part of this repository and come with their own license.

Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in this repository, as defined in the Apache-2.0 license, is dual licensed as above, without any additional terms or conditions.
