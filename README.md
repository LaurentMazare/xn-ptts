# Phonon

Phonon is Gradium's on-device text-to-speech runtime, written in Rust, with Python bindings. It builds on [Pocket TTS](https://github.com/kyutai-labs/pocket-tts), developed by Kyutai. This preview pairs the code in this repository with a model package supplied by Gradium; the model is not in this repository.

[![Rust CI](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/gradium-ai/xn-ptts/actions/workflows/rust-ci.yml)

## 1. Set up

You need [Rust](https://rustup.rs) for both paths, and [uv](https://docs.astral.sh/uv/) for Python. Clone this repository and unpack the model package next to it:

```bash
git clone https://github.com/gradium-ai/xn-ptts
cd xn-ptts

unzip /path/to/phonon-7e71a02d.200.zip
tar xzf phonon-7e71a02d.200/phonon-7e71a02d.200-gradium.tar.gz
export MODEL_DIR="$PWD/phonon-7e71a02d.200-gradium"
```

`MODEL_DIR` now holds `config.json`, `model.q8.gguf`, `tokenizer.json` and `voices/default.safetensors`. The commands below read only these files and download nothing.

## 2. Run it

With Rust, from the repository root (the first build takes a few minutes):

```bash
cargo run --release -p ptts --example pocket_tts --features hf,audio -- \
  --lang en --dir "$MODEL_DIR" "Hello world" -o out.wav
```

With Python, from the repository root (the first run builds the package, a few minutes):

```bash
uv run --project ptts-pyo3 --locked ptts --lang en \
  --model "$MODEL_DIR/config.json" --quant q8 "Hello world" -o out.wav
```

`--lang` is required. It picks how numbers, symbols and abbreviations are spelled out before synthesis: `en`, `fr`, `de`, `es` or `pt`, or `none` to use the text as written.

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

Load the model once and reuse it. `tts.say` returns the whole waveform as mono `f32` samples at `tts.sample_rate()`. `tts.stream` returns an iterator of chunks as they are generated, for playback that starts before the sentence is finished. Build with `--release`: a debug build is far too slow for realtime.

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
