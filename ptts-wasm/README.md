# Pocket TTS in WebAssembly

`ptts-wasm` runs the Rust Pocket TTS model in a browser. It exports a low-level [`Model`](src/lib.rs) that generates one 80 ms frame per call. The caller supplies weights, the matching `tokenizer.json`, and a voice safetensors file.

## Build and run the demo

Install [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) and build from this directory:

```bash
make build
cd pkg
python3 -m http.server 8080
```

Open `http://localhost:8080`. The demo fetches model files from Hugging Face when you load a model. It offers f32 and q8 weights; the q8 file is about 146 MB. The build requires WebAssembly Relaxed SIMD support in the browser.

## Raw JavaScript API

The `Model` constructor takes the weights and tokenizer as bytes, an optional `config.json` as bytes (or `undefined` for the original Pocket TTS architecture), `"f32"` or `"q8"`, and a required normalization language (`"en"`, `"fr"`, `"de"`, `"es"`, `"pt"`, or `"none"`). An optional final argument selects text rewrite rules.

`start_generation` splits and tokenizes the text; `next_chunk` prompts each chunk; `generation_step` produces PCM until that chunk ends. `stop_generation()` cancels the current run. The page in [`www/`](www/) shows how to fetch files and run generation in a worker.
