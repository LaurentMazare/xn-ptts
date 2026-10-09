# Build and develop Phonon

The root [README](../README.md) covers the consumer integrations. This guide is for building the current checkout, testing a supplied checkpoint, and preparing packages before public release.

## Model setup

Set `MODEL_DIR` to the absolute path of a checkpoint folder with its own config, tokenizer JSON, weights, and voice assets. The examples use q8 GGUF weights. `--lang` selects how the runtime reads numbers and symbols: `en`, `fr`, `de`, `es`, `pt`, or `none` to keep text unchanged. Choose it for the language of your input; the checkpoint determines which spoken languages it supports. See [model setup](../README.md#model-setup).

Clone the repository and run the following commands from its root:

```sh
git clone https://github.com/gradium-ai/xn-ptts.git
cd xn-ptts
```

## Command line

Install [Rust](https://rustup.rs), then:

```sh
cargo install --path ptts --locked --features cli
ptts --dir "$MODEL_DIR" --lang en --quant q8 "Hello from Phonon." -o speech.wav
```

For an edit/build/run loop:

```sh
cargo run --release -p ptts --features cli --bin ptts -- \
  --dir "$MODEL_DIR" --lang en --quant q8 "Hello from Phonon." -o speech.wav
```

The short `say` library example remains in [`ptts/examples/say.rs`](../ptts/examples/say.rs). The full CLI is now a binary, not an example. See [CLI options](cli.md).

## Python

Install Rust and [uv](https://docs.astral.sh/uv/), then run from the repository root:

```sh
uv run --project ptts-pyo3 --locked python -m ptts \
  --model "$MODEL_DIR" --lang en --quant q8 "Hello from Phonon." -o speech.wav
```

For another project, use `uv add /path/to/xn-ptts/ptts-pyo3` or `python -m pip install /path/to/xn-ptts/ptts-pyo3`. These are source installs and compile Rust. To install a built wheel instead:

```sh
python -m pip install "/path/to/ptts-VERSION-PLATFORM.whl"
```

Replace the example path with the actual wheel filename. [Python guide](../ptts-pyo3/README.md).

## Browser

Requires Rust with the Wasm target, Node 22.7+, [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/), and binaryen 124+ (`wasm-opt` on `PATH`). The threaded build also uses a pinned nightly toolchain installed by the Makefile.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
# Install binaryen 124+ from your package manager or its GitHub releases.
cd ptts-wasm
make threads-toolchain
make build
(cd pkg && npm pack)
```

Install the resulting tarball in your web app, using its actual filename:

```sh
npm install "/path/to/phonon-tts-VERSION.tgz"
```

Use the tarball rather than a folder dependency so npm copies the files into the consumer project. Model files are served separately; the package contains no weights.

To run the existing demo with your local checkpoint:

```sh
make serve MODEL_DIR="$MODEL_DIR"
```

Open `http://localhost:8080`. The server adds the headers required for Wasm threads. `make profiling` builds the single-thread variant without wasm-opt. [Browser guide](../ptts-wasm/js/README.md).

## Swift

On a Mac with Xcode 16 and the required Rust targets, build the core and export your checkpoint:

```sh
./ios/build-xcframework.sh
cargo run --release -p ptts --example export_coreml -- --dir "$MODEL_DIR" phonon-coreml
```

Add `ios/PhononTTS` to Xcode as a local package, select the `ptts` product, and add the exported `phonon-coreml` folder as a folder reference named `Models`. The module remains `PhononTTS`. Export conditions must match the checkpoint. [Swift guide](../ios/PhononTTS/README.md).

For ordinary app integration, the [published Swift archives](https://github.com/gradium-ai/xn-ptts/releases) include a package that downloads the compiled core. Custom checkpoint exports remain available.

## Docker

From the repository root:

```sh
docker build -f ptts-openai-server/Dockerfile -t ptts-openai-server:local .
docker run --rm -p 127.0.0.1:8880:8880 -v "$MODEL_DIR:/models:ro" \
  -e PTTS_CONFIG=/models -e PTTS_LANG=en -e PTTS_QUANT=q8 \
  ptts-openai-server:local
```

The image contains code, not weights. Hub loading, persistent cache volumes, formats, and admission limits are documented in the [server guide](../ptts-openai-server/README.md).

## Rust

Use a path dependency while working from this checkout:

```toml
[dependencies]
ptts = { path = "/absolute/path/to/xn-ptts/ptts", features = ["hf"] }
```

Run the consumer with `cargo run --release`. The `hf` feature enables the JSON tokenizer; `audio` adds audio-file decoding for voice creation. The library itself does not download checkpoints. [Rust guide](../ptts/README.md).

## Checks

CI is defined in [the Rust workflow](../.github/workflows/rust-ci.yml). Start with the checks for the code you changed:

```sh
cargo fmt --all -- --check
cargo test -p ptts --features cli --all-targets
cargo clippy -p ptts --features cli --all-targets -- -D warnings
node --test ptts-wasm/js/test/*.test.mjs
```

Do not use `--all-features`: CUDA requires its own SDK. The servers require libopus, and the HTTP server also uses LAME; their guides list the system packages. Release workflows set explicit CPU baselines rather than the checkout's `target-cpu=native` flags.

Installed wheel tests run without weights by default. Supply `PTTS_TEST_MODEL` and, for a Hub model, `PTTS_TEST_REVISION` to include the marked checkpoint tests. Keep private model files and credentials out of public artifacts and shared caches.

## Updating earlier builds

- Every model source is explicit: `--dir` or `--repo` in the Rust command, `config=` in Python, `--config` or `PTTS_CONFIG` for servers, and `model` in the browser. Docker images contain no weights.
- Checkpoints supply their own `config.json` and `tokenizer.json`. Built-in configs, Pocket TTS presets, legacy filename fallbacks, and `ptts-model.json` support were removed. `TTSConfig::v202601()`, native manifest types, and the browser's `POCKET_TTS_MODEL` export are no longer available.
- For a SentencePiece-only checkpoint, [convert its tokenizer to JSON](../scripts/convert-tokenizer.py) once. The runtime reads the supplied JSON tokenizer and does not guess another one.
- Streaming buffers are bounded. An unread stream pauses generation when its buffers fill; consume or cancel it when finished. Stop already-scheduled playback separately, using the cancellation examples in the package guides.
- The full Rust command moved from an example to a binary. Use `cargo run --release -p ptts --features cli --bin ptts -- ...` instead of `--example ptts`.
- The Swift product is `ptts`; existing package dependencies must select that product. `import PhononTTS` is unchanged.
- The browser npm name remains `phonon-tts`.
