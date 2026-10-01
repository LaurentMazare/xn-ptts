# ptts for Python

Python bindings for the [Phonon Rust runtime](../README.md). For this preview, the package is built from this repository and reads the model package supplied by Gradium.

Set `MODEL_DIR` as described in the [root README](../README.md#1-set-up). Then, from the repository root:

```bash
uv run --project ptts-pyo3 --locked python - <<'PY'
import os
import ptts

tts = ptts.TTS(lang="en", config=os.environ["MODEL_DIR"] + "/config.json", quant="q8")
print(tts.voices)
tts.save("out.wav", "Hello world")
PY
```

To use it from your own project, install it with `uv add /path/to/xn-ptts/ptts-pyo3` or `pip install /path/to/xn-ptts/ptts-pyo3`. Either one compiles the Rust code, so it needs Rust installed.

`config` is the path to `config.json`. The package loads `model.q8.gguf`, `tokenizer.json` and every voice in `voices/` from the same directory, and downloads nothing. `quant="q8"` matches the weight format of `model.q8.gguf`.

`lang` is required: `en`, `fr`, `de`, `es` or `pt` picks how numbers, symbols and abbreviations are spelled out; `none` uses the text as written.

## Speech and voices

Reuse `tts` across requests. Use `synth` for an array, `save` for a WAV, or `stream` for chunks as they are decoded:

```python
voice = tts.voices[0]
pcm = tts.synth("Hello", voice=voice)  # 1-D float32 NumPy array
seconds = tts.save("out.wav", "Hello", voice=voice)
with tts.stream("A longer sentence.", voice=voice) as audio:
    for chunk in audio:
        print(chunk.shape)  # process each PCM chunk as it arrives
```

`tts.sample_rate` is the PCM sample rate; `save` writes a mono 16-bit WAV and returns its duration. Leaving the `with` block stops a stream early.

`tts.voices` lists the names loaded from `voices/`, and the first one is used when no voice is given. Pass `voice="name"` to any speech method to select one.

## Command line

The package also provides the `ptts` command. From the repository root:

```bash
uv run --project ptts-pyo3 --locked ptts --lang en --quant q8 \
  --model "$MODEL_DIR/config.json" "Hello world" -o out.wav
```

`--voice` selects a loaded voice, and `--list-voices` prints the available names. Run with `--help` for the remaining options. The package ships type stubs and `py.typed`.
