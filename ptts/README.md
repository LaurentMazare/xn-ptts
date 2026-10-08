# ptts

Phonon is Gradium's on-device text-to-speech runtime. The `ptts` crate provides its Rust API and an optional `ptts` command. The code is licensed under MIT OR Apache-2.0; model and voice licenses are separate.

Each checkpoint supplies its own `config.json`, `tokenizer.json`, weights, and voice assets. Supply a model explicitly. No checkpoint is selected or downloaded by the library.

See the [CLI guide](https://github.com/gradium-ai/xn-ptts/blob/main/docs/cli.md) for desktop downloads and platform requirements.

## Command line

Installing the command requires `--features cli`; without it, this crate builds only the library.

From a checkout of [xn-ptts](https://github.com/gradium-ai/xn-ptts):

```sh
cargo install --path ptts --locked --features cli
ptts --dir /path/to/model --lang en --quant q8 "Hello world" -o out.wav
```

Use `--repo <owner/model>` instead of `--dir` to download from Hugging Face. `--revision <commit>` pins all files to that revision. For private models, set `HF_TOKEN` or log in with the Hugging Face CLI. `ptts --help` lists all options.

`--lang` is required: `en`, `fr`, `de`, `es`, `pt`, or `none` to disable normalization. Select a language supported by your checkpoint. `--quant q8` keeps q8 GGUF weights quantized on CPU; omit it for f32 weights.

The Python package also installs a command named `ptts`, with `--model` instead of `--dir`/`--repo`. When both are installed, `PATH` order selects the command. Use `python -m ptts` with your Python environment's interpreter to invoke Python explicitly.

The short Rust library example remains in `examples/say.rs`.

## Rust API

Enable `hf` for the Hugging Face JSON tokenizer. The library loads local files and does not require the `cli` feature.

```rust,no_run
use ptts::checkpoint::{Checkpoint, ResolveOptions};
use ptts::preprocess::Lang;
use ptts::synth::{DeviceKind, Quant};

fn main() -> ptts::Result<()> {
    let checkpoint = Checkpoint::resolve(
        "/path/to/model",
        ResolveOptions { quant: Quant::Q80, weights: None },
    )?;
    let mut tts = checkpoint.builder(Lang::En).device(DeviceKind::Cpu).build()?;
    checkpoint.register_voices(&mut tts);
    let pcm = tts.say("Hello world")?;
    ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate())
}
```

Reuse the loaded model for later requests. `tts.stream(text)?` yields bounded PCM chunks for streaming playback. See the [repository README](https://github.com/gradium-ai/xn-ptts#readme) for voices, generation controls, and other integrations.
