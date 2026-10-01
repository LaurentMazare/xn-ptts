# Phonon

Phonon is Gradium's on-device text-to-speech runtime, written in Rust, with Python bindings. It builds on [Pocket TTS](https://github.com/kyutai-labs/pocket-tts), developed by Kyutai. This preview pairs the code in this repository with a model package supplied by Gradium; the model is not in this repository.

[![Rust CI](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml)

## 1. Set up

You need [Rust](https://rustup.rs) for both paths, and [uv](https://docs.astral.sh/uv/) for Python.

Point `MODEL_DIR` at the model folder, the one holding `config.json`, `model.q8.gguf`, `tokenizer.json` and `voices/default.safetensors`:

```bash
export MODEL_DIR=/path/to/model
```

The commands below read only the files in `MODEL_DIR` and download nothing.

## 2. Run it

With Rust, from the repository root (the first build takes a few minutes):

```bash
cargo run --release -p ptts --example ptts --features hf,audio -- \
  --lang en --dir "$MODEL_DIR" --quant q8 "Hello world" -o out.wav
```

With Python, from the repository root (the first run builds the package, a few minutes):

```bash
uv run --project ptts-pyo3 --locked ptts --lang en \
  --model "$MODEL_DIR/config.json" --quant q8 "Hello world" -o out.wav
```

`--quant q8` runs the model in q8, the format `model.q8.gguf` is stored in. Without it the weights are expanded to f32, which is slower and uses more memory; the Rust and Python examples below set q8 too. `--lang` is required. It picks how numbers, symbols and abbreviations are spelled out before synthesis: `en`, `fr`, `de`, `es` or `pt`, or `none` to use the text as written.

## 3. Use it from Rust

Add the crate from your checkout as a path dependency:

```toml
[dependencies]
ptts = { path = "/path/to/xn-ptts/ptts", features = ["hf"] }
serde_json = "1"
```

```rust
use std::{env, fs, path::PathBuf};
use ptts::preprocess::{Lang, Normalize};
use ptts::synth::{Quant, Synth};
use ptts::tts_model::TTSConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(env::var("MODEL_DIR")?);
    let config: TTSConfig = serde_json::from_slice(&fs::read(dir.join("config.json"))?)?;
    let tts = Synth::builder(config, dir.join("model.q8.gguf"), Normalize::for_lang(Lang::En))
        .tokenizer_file(dir.join("tokenizer.json"))
        .quant(Quant::Q80)
        .add_voice("default", dir.join("voices/default.safetensors"))
        .build()?;

    let pcm = tts.say("Hello world")?;
    ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate() as u32)?;
    Ok(())
}
```

Load the model once and reuse it. `tts.say` returns the whole waveform as mono `f32` samples at `tts.sample_rate()`. `tts.stream(text)?` is an iterator of `Result<Vec<f32>>` chunks, yielded as they are generated, for playback that starts before the sentence is finished. Build with `--release`: a debug build is far too slow for realtime.

## 4. Use it from Python

Install the package from your checkout into your project. This compiles the Rust code, so it needs Rust installed:

```bash
uv add /path/to/xn-ptts/ptts-pyo3      # or: pip install /path/to/xn-ptts/ptts-pyo3
```

```python
import os
import ptts

model = os.environ["MODEL_DIR"]
tts = ptts.TTS(lang="en", config=f"{model}/config.json", quant="q8")

tts.save("out.wav", "Hello world")    # write a 16-bit WAV
pcm = tts.synth("Hello world")        # float32 NumPy array at tts.sample_rate
with tts.stream("A longer sentence, played as it is generated.") as audio:
    for chunk in audio:
        ...                           # each chunk is a float32 NumPy array
```

Load the model once and reuse it. The [Python README](ptts-pyo3/README.md) covers voices and the remaining options.

## 5. Use it in an iOS or macOS app

The `PhononTTS` Swift package runs the model on the device through Core ML, with its transformer on the Apple Neural Engine: about 12 times faster than realtime on an iPhone 16 Pro, with first audio in under 40 ms. It needs iOS 18 or macOS 15, and Xcode.

Build the package's compiled core and convert the model to Core ML, both from the repository root:

```bash
./ios/build-xcframework.sh
cargo run --release -p ptts --example export_coreml -- --dir "$MODEL_DIR" phonon-coreml
```

Then add `ios/PhononTTS` to your Xcode project as a local package, add the `phonon-coreml` folder to your app as a folder reference named `Models`, and speak:

```swift
import PhononTTS

let models = try PhononModels.install(bundled: Bundle.main.url(forResource: "Models", withExtension: nil)!)
let tts = try await Phonon.load(models: models, language: .english)
try await PhononPlayer().play(tts.stream("Hello world"))
```

The [package README](ios/PhononTTS/README.md) covers downloading the models instead of bundling them, voices, and the rest of the API.
