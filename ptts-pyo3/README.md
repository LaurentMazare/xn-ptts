# ptts for Python

Python bindings for the [Phonon Rust runtime](https://github.com/gradium-ai/xn-ptts). Install the package:

```bash
pip install ptts
```

Model weights are supplied separately. Set `MODEL_DIR` to a local checkpoint directory containing `config.json`, `tokenizer.json`, weights, and voice files. Then:

```bash
export MODEL_DIR=/path/to/model
python - <<'PY'
import os
import ptts

tts = ptts.TTS(lang="en", config=os.environ["MODEL_DIR"], quant="q8")
print(tts.voices)
tts.save("out.wav", "Hello world")
PY
```

For a source install, use `uv add /path/to/xn-ptts/ptts-pyo3` or `pip install /path/to/xn-ptts/ptts-pyo3`. Source builds need Rust installed; supported CPython wheels do not.

`config` must name a local directory, a `config.json`, or a Hugging Face repo ID. There is no default model. Local paths download nothing. Rust examples, Python, and both servers use the [shared checkpoint resolver](https://github.com/gradium-ai/xn-ptts/blob/main/ptts/src/checkpoint.rs). Every checkpoint supplies its own `config.json` and `tokenizer.json`. q8 prefers `model.q8.gguf`; other formats prefer `model.safetensors`. Voices come from `voices/` or `embeddings/`, plus `default-voice.safetensors` as `default`. Hub downloads use the supplied `revision` for all files.

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

`tts.sample_rate` is the PCM sample rate; `save` writes a mono 16-bit WAV and returns its duration. Leaving the `with` block stops a stream early and waits for its workers to finish. Streaming keeps bounded audio and latent buffers: leaving a stream unread pauses generation once they fill, and consuming chunks lets it resume.

`tts.voices` lists the voices that were found. When no voice is given, the checkpoint's configured default is used, then `default` if present, then the first by name. Pass `voice="name"` to any speech method to select one.

## Command line

The package also provides the `ptts` command:

```bash
python -m ptts --lang en --quant q8 \
  --model "$MODEL_DIR" "Hello world" -o out.wav
```

`--voice` selects a loaded voice, and `--list-voices` prints the available names. Run with `--help` for the remaining options. The package ships type stubs and `py.typed`.

## Wheels

Released wheels are `cp39-abi3`, so one per platform covers every CPython from 3.9 on, and a
new CPython release needs no new wheel. Free-threaded CPython and PyPy cannot load an abi3
extension, so those two install from the sdist and compile the Rust runtime, which needs a
Rust toolchain and takes a few minutes.

## License

MIT or Apache-2.0, at your option. The model weights are not part of the package and come with their own license.
