# Phonon

On-device text-to-speech with [Kyutai's Pocket TTS model](https://huggingface.co/kyutai/pocket-tts). The Rust runtime produces 24 kHz audio and powers the `ptts` Python package, a command-line tool, a WebSocket server, and a browser build. It does not require PyTorch.

[![Rust CI](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml)

## Try it

```bash
uvx ptts --lang en "Hello world" -o out.wav
```

Or use Python:

```bash
pip install ptts
```

```python
import ptts

tts = ptts.TTS(lang="en")
tts.save("out.wav", "Hello world")
```

The first run downloads the default checkpoint from [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts). `lang` is required: choose `en`, `fr`, `de`, `es`, or `pt` for text normalization, or `none` to pass text to the tokenizer as written. These are normalization options, not a claim about which languages a checkpoint was trained to speak.

The checkpoint supplies eight voice embeddings: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, and `azelma`. Checkpoints with a speaker encoder can also clone a voice from a short audio sample. See the [Python package README](ptts-pyo3/README.md) for streaming, voices, and local checkpoints.

## Rust

The `ptts` crate reads checkpoint files supplied by the caller; it does not download them. With a local weights file, matching `tokenizer.json`, and voice embedding:

```bash
cargo add ptts --features hf
```

```rust
use ptts::preprocess::{Lang, Normalize};
use ptts::synth::Synth;
use ptts::tts_model::TTSConfig;

fn main() -> ptts::Result<()> {
    let tts = Synth::builder(
        TTSConfig::v202601(0.3),
        "model/model.safetensors",
        Normalize::for_lang(Lang::En),
    )
    .tokenizer_file("model/tokenizer.json")
    .add_voice("alba", "model/embeddings/alba.safetensors")
    .build()?;

    let pcm = tts.say("Hello world")?;
    ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate() as u32)?;
    Ok(())
}
```

`Synth::stream` yields audio chunks as they are decoded. The `audio` feature adds audio file decoding and resampling for voice cloning. GPU features are `cuda`, `vulkan`, `metal`, and `webgpu`; `accelerate` enables Apple's CPU acceleration. Quantized GGUF weights run on CPU. See the [API docs](https://docs.rs/ptts) for the builder and lower-level `TTSModel` API.

For a ready-to-run Rust example that downloads the checkpoint:

```bash
cargo run --release --example pocket_tts --features hf,audio -- --lang en "Hello world" -o out.wav
```

Use `--dir <path>` for a local checkpoint, `--voice <name-or-file>` to select or clone a voice, and `--weights <file> --quant <format>` for GGUF weights. The example's checkpoint layout and defaults are in [`model_helpers.rs`](ptts/examples/model_helpers.rs).

## Server and browser

The WebSocket server streams audio at `/speech/tts` in formats including PCM, WAV, and Ogg Opus. Building it requires a system `libopus`:

```bash
cargo run --release -p ptts-ws-server -- --lang en
```

The wire format is defined in [`protocol.rs`](ptts-ws-server/src/protocol.rs).

The browser build runs the model in WebAssembly. To build and serve its demo:

```bash
cd ptts-wasm
make build
cd pkg
python3 -m http.server 8080
```

Open `http://localhost:8080`. The demo fetches its model files from Hugging Face. See the [WASM README](ptts-wasm/README.md) for build requirements and the frame-by-frame API.

## Development and licence

CI checks formatting, Clippy, tests, optional features, docs, and the WASM target. Run `cargo fmt --all -- --check` and `cargo test -p ptts --features hf,audio` locally. The server needs `libopus` to build.

The code is available under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE). Model weights are distributed separately under the terms on their respective model cards.
