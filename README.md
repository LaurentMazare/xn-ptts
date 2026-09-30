# Phonon

Phonon is Gradium's on-device text-to-speech runtime in Rust, with Python, WebSocket, and WebAssembly frontends. It builds on [Pocket TTS](https://github.com/kyutai-labs/pocket-tts), developed by Kyutai, and can load compatible checkpoints. This preview uses a checkpoint supplied separately by Gradium; the model files are not in this repository.

[![Rust CI](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml)

## Run the preview from source

From the repository root, set `MODEL_DIR` to the supplied checkpoint directory. It must contain:

```text
config.json
model.safetensors              # or model.q8.gguf
tokenizer.json
voices/                        # at least one .safetensors voice file
```

The config, weights, tokenizer, and voice files must belong to the same checkpoint. If the supplied files contain `tokenizer.model` instead of `tokenizer.json`, run `uv run --script scripts/convert-tokenizer.py "$MODEL_DIR/tokenizer.model"` once. The Rust example selects the first available voice unless you pass `--voice <name>`.

```bash
export MODEL_DIR=/absolute/path/to/checkpoint
cargo run --release -p ptts --example pocket_tts --features hf,audio -- \
  --lang en --dir "$MODEL_DIR" "Hello world" -o out.wav
```

This command uses `model.safetensors`. If your checkpoint contains `model.q8.gguf` instead, add `--weights model.q8.gguf --quant q8` after `--dir "$MODEL_DIR"`. It reads the checkpoint locally; it does not download Kyutai's model. Choose `--lang en`, `fr`, `de`, `es`, or `pt` for text normalization, or `none` to pass text through unchanged. The normalization choice does not establish which languages the supplied checkpoint supports.

## Integrate from Rust

Use the crate from this checkout as a path dependency; replace the path with your checkout location:

```toml
[dependencies]
ptts = { path = "/absolute/path/to/xn-ptts/ptts", features = ["hf"] }
serde_json = "1"
```

This example reads the supplied config and tokenizer, selects the available weight format, and registers one matching voice:

```rust
use std::{env, fs, io, path::PathBuf};
use ptts::preprocess::{Lang, Normalize};
use ptts::synth::{Quant, Synth};
use ptts::tts_model::TTSConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(env::var("MODEL_DIR")?);
    let config: TTSConfig = serde_json::from_slice(&fs::read(dir.join("config.json"))?)?;
    let (weights, quant) = if dir.join("model.safetensors").is_file() {
        (dir.join("model.safetensors"), Quant::F32)
    } else {
        (dir.join("model.q8.gguf"), Quant::Q80)
    };
    let voice = fs::read_dir(dir.join("voices"))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .find(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no voice file"))?;

    let tts = Synth::builder(config, weights, Normalize::for_lang(Lang::En))
        .tokenizer_file(dir.join("tokenizer.json"))
        .quant(quant)
        .add_voice("preview", voice)
        .build()?;
    let pcm = tts.say("Hello world")?;
    ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate() as u32)?;
    Ok(())
}
```

`Synth::stream` yields audio chunks as they are decoded. See the [`pocket_tts` example](ptts/examples/pocket_tts.rs) for checkpoint loading and the [API docs](https://docs.rs/ptts) for other `Synth` options. The `audio` feature adds audio-file voice prompts; backend features are `cuda`, `vulkan`, `metal`, `webgpu`, and `accelerate`. Quantized GGUF weights run on CPU.

## Python from the checkout

With [uv](https://docs.astral.sh/uv/) installed, run the Python package directly from this repository:

```bash
uv run --project ptts-pyo3 --locked python - <<'PY'
import os
from pathlib import Path
import ptts

model = Path(os.environ["MODEL_DIR"])
quant = "q8" if not (model / "model.safetensors").is_file() else None
tts = ptts.TTS(lang="en", config=str(model / "config.json"), quant=quant)
tts.save("out.wav", "Hello world")
PY
```

See the [Python README](ptts-pyo3/README.md) for streaming, voices, and the source-built command line.

## Other interfaces

The WebSocket server accepts the same local checkpoint layout:

```bash
cargo run --release -p ptts-ws-server -- --lang en --config "$MODEL_DIR/config.json"
```

It serves `/speech/tts` and requires a system `libopus` to build. For `model.q8.gguf`, add `--quant q8`. The wire format is defined in [`protocol.rs`](ptts-ws-server/src/protocol.rs).

The [WebAssembly crate](ptts-wasm/README.md) exposes a frame-by-frame browser API. Its included demo is wired to a Kyutai compatibility checkpoint, so it is not the entry point for this preview checkpoint.

## Development

`cargo fmt --all -- --check` and `cargo test -p ptts --features hf,audio` cover the core crate. [Rust CI](.github/workflows/rust-ci.yml) checks the workspace, optional features, docs, and WebAssembly build.
