# ptts for Python

Python bindings for the Rust [Pocket TTS runtime](../README.md). The package produces 24 kHz mono speech without PyTorch.

```bash
uvx ptts --lang en "Hello world" -o out.wav
# Or install the package for Python code:
pip install ptts
```

```python
import ptts

tts = ptts.TTS(lang="en")
tts.save("out.wav", "Hello world")
```

The first run downloads the default [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts) checkpoint. `lang` is a required keyword argument: `en`, `fr`, `de`, `es`, or `pt` normalizes text before tokenization; `none` leaves it as written.

## Speech and voices

```python
tts = ptts.TTS(lang="en")
print(tts.voices)

pcm = tts.synth("Hello", voice="marius")  # 1-D float32 NumPy array
seconds = tts.save("out.wav", "Hello")    # mono 16-bit WAV

with tts.stream("A longer piece of text.") as audio:
    for chunk in audio:
        print(chunk.shape)
```

Leave the `with` block to stop a stream early. For a checkpoint with a speaker encoder, `tts.supports_voice_cloning` is true. Pass about ten seconds of float32 mono PCM at `tts.voice_prompt_sample_rate` to `tts.clone_voice("me", pcm)` and then use `voice="me"` in `synth`, `save`, or `stream`.

`ptts.TTS(lang="en", config="model/config.json")` loads a local checkpoint whose weights and `tokenizer.json` sit next to the config. `ptts.available_devices()` and `ptts.available_quants()` list options supported by the installed build; quantized weights require a matching GGUF checkpoint.

## Command line

The installed `ptts` command and `python -m ptts` share the same options:

```bash
ptts --lang en "hello world" -o out.wav
ptts --lang en --list-voices
ptts --help
```

`-m` selects a Hugging Face repo with `config.json` or a local config file. `-v` selects a voice, `-d` a compiled-in device, and `-q` a weight format. The package ships type stubs and `py.typed`.
