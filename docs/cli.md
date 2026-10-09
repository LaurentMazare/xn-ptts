# Phonon command line

The `ptts` command generates a WAV from an explicitly selected Phonon checkpoint. Models and voices have their own licenses and are downloaded separately.

## Desktop downloads

The CLI release workflow builds the following CPU archives. Once published, they appear on the matching [GitHub Release](https://github.com/gradium-ai/xn-ptts/releases), with `SHA256SUMS` for verification.

| Platform | Archive target | Requirements |
|---|---|---|
| Linux x86_64 | `x86_64-unknown-linux-gnu` | glibc 2.35 or later; x86-64-v3 CPU |
| Linux ARM64 | `aarch64-unknown-linux-gnu` | glibc 2.35 or later; ARM64 CPU |
| Mac Apple silicon | `aarch64-apple-darwin` | macOS 15 or later |
| Mac Intel | `x86_64-apple-darwin` | macOS 15 or later; x86-64-v3 CPU |
| Windows x64 | `x86_64-pc-windows-msvc` | Windows 10 or later; x86-64-v3 CPU |

Mac downloads target macOS 15, the oldest Mac runner used to check them. Older macOS versions require a source build and their own verification.

x86-64-v3 includes AVX2, FMA, and F16C. Older x86 CPUs need a source build suited to their CPU. These downloads use CPU execution; GPU support requires a source build with the relevant backend feature.

Extract the archive. On Linux or Mac, put the extracted `ptts` file in a directory on your `PATH`, for example `~/.local/bin`. On Windows, run `ptts.exe` in PowerShell, or add its folder to `PATH`. Rust and Python are not required for these downloads. Mac may prompt you to allow the downloaded executable in System Settings because it is not notarized.

For Linux or Mac, download `SHA256SUMS` alongside the archive and verify that archive's entry:

```sh
# Linux
sha256sum --ignore-missing --check SHA256SUMS
# Mac: use the checksum entry for the archive you downloaded
shasum -a 256 ptts-<version>-<target>.tar.gz
```

On Windows, compare `Get-FileHash .\ptts-<version>-x86_64-pc-windows-msvc.zip -Algorithm SHA256` with its entry in `SHA256SUMS`.

## Homebrew

On Mac or Linux, install the prebuilt CPU command from [Gradium's Homebrew tap](https://github.com/gradium-ai/homebrew-tap):

```sh
brew install gradium-ai/tap/ptts
```

Homebrew adds the tap automatically. The formula currently installs the `0.4.0-rc.1` runtime preview and pins each release archive by SHA-256. It does not include a model. Supply one with `--dir` or `--repo` when running `ptts`.

After a new formula version is published, run `brew update` and `brew upgrade gradium-ai/tap/ptts`. The platform and CPU requirements above still apply.

## Generate speech

```sh
ptts --dir /path/to/model --lang en --quant q8 "Hello world" -o out.wav
```

The folder must contain `config.json`, `tokenizer.json`, weights, and any required voice assets. Use q8 for a q8 GGUF checkpoint, or omit `--quant` for f32 weights. `--lang` selects how the runtime reads numbers and symbols: `en`, `fr`, `de`, `es`, `pt`, or `none` to keep text unchanged. Choose it for the language of your input; the checkpoint determines which spoken languages it supports.

You can also use a Hugging Face repo:

```sh
ptts --repo <owner/model> --revision <commit> --lang en --quant q8 "Hello world" -o out.wav
```

All files use the specified revision. Without `--revision`, the repo's current main revision is used. Private models require `HF_TOKEN` or a saved Hugging Face login. Cached files are reused. Local folders require no download.

Use `--voice <name>` for a bundled voice, `--voice /path/to/voice.safetensors` for an embedding, or a short audio file for voice cloning when the checkpoint supports it. With no voice argument, the checkpoint's default voice is used. `ptts --help` lists the remaining options, including `--threads` for CPU tuning. `ptts --version` reports the runtime version.

## Build from source

From the repository root:

```sh
cargo install --path ptts --locked --features cli
```

The `cli` feature includes tokenizer support and audio-file decoding. Optional backend features, such as `metal` and `cuda`, can be added to the installation command. `say` remains a small Rust library example. There is no longer a `ptts` example target; contributors can run the command with `cargo run --release -p ptts --features cli --bin ptts -- ...`.
