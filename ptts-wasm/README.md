# Pocket TTS in WebAssembly

`ptts-wasm` exposes the Rust model as a low-level browser [`Model`](src/lib.rs). It accepts checkpoint weights, a matching `tokenizer.json`, an optional `config.json`, and a voice safetensors file as bytes supplied by the caller. It generates one 80 ms PCM frame per call.

Build it from this directory with [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) 0.12 or later and Node 22.7 or later:

```bash
make build
```

This runs `wasm-pack build` into `pkg/wasm/`, then `scripts/pack.mjs`, which assembles the `phonon-tts` npm package in `pkg/`. [`js/README.md`](js/README.md) covers using the package, including loading a checkpoint from your own URLs.

`Model` accepts `"f32"` or `"q8"` weights. Its required language argument is one of `"en"`, `"fr"`, `"de"`, `"es"`, `"pt"`, or `"none"`; an optional final argument selects text rewrite rules. `start_generation` splits and tokenizes text, `next_chunk` prompts each chunk, `generation_step` returns PCM frames, and `stop_generation` cancels a run. The build requires WebAssembly Relaxed SIMD support in the browser.

The included [`www/`](www/) demo currently fetches a Kyutai compatibility checkpoint. It does not load the separately supplied Gradium preview checkpoint. For a runnable preview with that checkpoint, use the [Rust or Python instructions](../README.md#1-set-up).
