# ptts for Python

Python bindings for the [Phonon Rust runtime](../README.md). For this preview, build the package from the repository and point it at the checkpoint supplied separately by Gradium.

Set `MODEL_DIR` to the checkpoint directory described in the [root README](../README.md#run-the-preview-from-source). From the repository root:

```bash
export MODEL_DIR=/absolute/path/to/checkpoint
uv run --project ptts-pyo3 --locked python - <<'PY'
import os
from pathlib import Path
import ptts

model = Path(os.environ["MODEL_DIR"])
quant = "q8" if not (model / "model.safetensors").is_file() else None
tts = ptts.TTS(lang="en", config=str(model / "config.json"), quant=quant)
print(tts.voices)
tts.save("out.wav", "Hello world")
PY
```

The package loads `model.safetensors` when present, otherwise `model.q8.gguf`; a GGUF checkpoint needs the matching `quant=` value. It reads `tokenizer.json` and voice files from `voices/` next to `config.json`. Convert a supplied `tokenizer.model` as described in the root README before running. No model is downloaded when `config` names a local file.

`lang` is required: `en`, `fr`, `de`, `es`, or `pt` selects text normalization; `none` leaves text as written. Use a language the supplied checkpoint supports.

## Speech and voices

`tts.synth(text)` returns a one-dimensional float32 NumPy array at `tts.sample_rate`. `tts.save(path, text)` writes a mono 16-bit WAV and returns its duration. `tts.stream(text)` yields PCM chunks and can be used as a context manager to stop early.

`tts.voices` lists the names loaded from `voices/`. Pass `voice="name"` to any speech method to select one. If `tts.supports_voice_cloning` is true, `tts.clone_voice("me", pcm)` accepts a short float32 mono voice prompt sampled at `tts.voice_prompt_sample_rate`.

## Command line

The source-built package also provides the `ptts` command. From the repository root:

```bash
uv run --project ptts-pyo3 --locked ptts --lang en \
  --model "$MODEL_DIR/config.json" "Hello world" -o out.wav
```

Add `--quant q8` when the checkpoint uses `model.q8.gguf`. `--voice` selects a loaded voice, and `--list-voices` prints available names. Run with `--help` for the remaining options. The package ships type stubs and `py.typed`.
