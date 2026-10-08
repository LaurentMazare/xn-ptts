# ptts-openai-server

A speech server for Phonon compatible with OpenAI's text-to-speech API: `POST /v1/audio/speech`, so any client with an "OpenAI TTS" setting and a custom base URL works unchanged. For streaming text in and audio out over one WebSocket connection, use [`ptts-ws-server`](../ptts-ws-server) instead.

## Docker

Set `MODEL_DIR` to a local folder holding the checkpoint's config, tokenizer, weights, and voices:

```bash
docker run -p 8880:8880 -v "$MODEL_DIR:/models:ro" \
  -e PTTS_CONFIG=/models -e PTTS_LANG=en -e PTTS_QUANT=q8 \
  ghcr.io/gradium-ai/ptts-openai-server
```

The image contains the server and no model weights. It is built for `linux/amd64` (an x86 CPU with AVX2) and `linux/arm64`. Every model supplies its own `config.json` and `tokenizer.json`.

To download from HF instead, set `PTTS_CONFIG=OWNER/MODEL` and `PTTS_REVISION` to the desired revision. Set `HF_TOKEN` for a private repo. You can mount a writable `HF_HOME` to retain the download cache.

Every flag below has an environment variable, so the image is configured with `-e`. Additional voices can be mounted with `-v ./voices:/voices:ro -e PTTS_VOICE_DIR=/voices`. [compose.yaml](compose.yaml) mounts the folder named by `MODEL_DIR`:

```bash
docker compose -f ptts-openai-server/compose.yaml up
```

To build the image yourself, from the repository root: `docker build -f ptts-openai-server/Dockerfile -t ptts-openai-server .`

## Build and run

It needs two system libraries: libopus, and LAME 3.99 or later for MP3.

```bash
sudo apt-get install libopus-dev libmp3lame-dev pkg-config   # Debian, Ubuntu
brew install opus lame pkg-config                            # macOS

cargo run --release -p ptts-openai-server -- --config "$MODEL_DIR" --quant q8 --lang en
```

`--config` names a checkpoint folder (or a `config.json` in one) or a Hugging Face repo id. It is required; no model is selected automatically.

It listens on `0.0.0.0:8880` (`--addr` to change it). There is no authentication, and each request can ask for up to 4096 characters of speech. On a machine others can reach, bind to `--addr 127.0.0.1:8880` or put it behind a proxy that checks access. `--lang` is required: it picks how numbers and symbols are spelled out. `--device auto` uses the GPU backend the binary was built with, if any; quantized weights such as `--quant q8` run on the CPU only. A checkpoint whose config lists conditioners, such as `padding_bonus`, takes their values with `--condition padding_bonus=0.5` (repeatable, or `PTTS_CONDITION=padding_bonus=0.5,num_speakers=2` in the environment); those not given take their defaults. `--help` lists the rest.

By default, one speech generation runs at a time. `--max-concurrent-requests N` (or `PTTS_MAX_CONCURRENT_REQUESTS=N`) sets a positive limit. Excess speech requests immediately receive HTTP 429 with OpenAI error code `server_busy`; retry when an active generation finishes. Slots are released after generation and worker cleanup, including after a disconnect or failure. Health, model, and voice endpoints remain available while generation is busy.

## The OpenAI-compatible API

```bash
curl http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "Hello from Phonon.", "voice": "default"}' \
  -o hello.mp3
```

Or with the OpenAI SDK, pointed at the server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="unused")
client.audio.speech.create(model="tts-1", voice="default", input="Hello.").write_to_file("hello.mp3")
```

| Field | What the server does |
|---|---|
| `input` | Required, up to 4096 characters. Long text is split at sentence ends. |
| `voice` | A voice of this checkpoint, in any case (`GET /v1/audio/voices` lists them), or `default`, which is also what a missing `voice` means. OpenAI's own names such as `alloy` are a 400. |
| `response_format` | `mp3` (the default), `opus`, `wav` or `pcm` (headerless 16-bit mono at 24 kHz). `aac` and `flac` are a 400. |
| `speed` | Only `1.0`: the model has no rate control yet, so other values are a 400. |
| `model`, `instructions`, `stream_format` | Accepted and ignored. The reply is always the audio bytes, streamed as they are generated. |

Errors use OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`, so client SDKs report them as usual. `GET /v1/models` names the loaded checkpoint and `GET /health` answers `{"status": "ok"}`. There is no authentication: an `Authorization` header is ignored. A failure after the audio has started can only cut the response short. A streamed `wav` has no length in its header, which a few strict parsers reject; `mp3` or `pcm` suit those.

## Clients

- **Open WebUI:** Admin Settings, Audio, Text-to-Speech engine `OpenAI`, API base URL `http://<host>:8880/v1`, any API key. Set the voice to `default` or a voice from `/v1/audio/voices`.
- **Home Assistant** ([sfortis/openai_tts](https://github.com/sfortis/openai_tts)): Custom endpoint, URL `http://<host>:8880/v1/audio/speech`, model `tts-1`. The voice list is read from `/v1/audio/voices`.
- **LiveKit Agents:** `openai.TTS(base_url="http://<host>:8880/v1", api_key="unused", model="tts-1", voice="default")`.
- **SillyTavern:** TTS provider "OpenAI Compatible", endpoint `http://<host>:8880/v1/audio/speech`, voices `default` or the checkpoint's own. Leave the speed at 1.
