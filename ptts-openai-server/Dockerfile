# syntax=docker/dockerfile:1

# The OpenAI-compatible speech server with a checkpoint baked in, so that
#
#   docker run -p 8880:8880 ghcr.io/gradium-ai/ptts-openai-server
#
# speaks with no further arguments. Build it from the repository root:
#
#   docker build -f ptts-openai-server/Dockerfile -t ptts-openai-server .

FROM rust:1-bookworm AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends libopus-dev libmp3lame-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# `.cargo/config.toml` is left out of the build context (see Dockerfile.dockerignore), so nothing
# pins `target-cpu=native`. x86_64 gets x86-64-v3, as the Python wheels do: `xn` picks its
# quantized kernels at compile time, and a lower baseline would cost q8 its AVX kernels.
ARG TARGETARCH
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target,sharing=locked \
    if [ "$TARGETARCH" = "amd64" ]; then export RUSTFLAGS="-C target-cpu=x86-64-v3"; fi \
    && cargo build --release --locked -p ptts-openai-server \
    && cp target/release/ptts-openai-server /usr/local/bin/

# Kyutai's Pocket TTS checkpoint without its voice-cloning weights, which this server does not
# use. It is not gated and is licensed CC-BY-4.0; its README, which says so, is baked in with it.
# Pinned to a revision, so a rebuild bakes the same files.
FROM debian:bookworm-slim AS weights
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
ARG HF_REPO=kyutai/pocket-tts-without-voice-cloning
ARG HF_REVISION=8843db76457a91db32077edf8dfcd1c0e3e755fd
RUN set -eu; \
    url="https://huggingface.co/$HF_REPO/resolve/$HF_REVISION"; \
    dir=/models/pocket-tts; \
    mkdir -p "$dir/embeddings"; \
    for file in README.md tokenizer.json tts_b6369a24.safetensors; do \
        curl -fsSL "$url/$file" -o "$dir/$file"; \
    done; \
    for voice in alba azelma cosette eponine fantine javert jean marius; do \
        curl -fsSL "$url/embeddings/$voice.safetensors" -o "$dir/embeddings/$voice.safetensors"; \
    done

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libmp3lame0 libopus0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 phonon
COPY --from=build /usr/local/bin/ptts-openai-server /usr/local/bin/
COPY --from=weights /models /models
LABEL org.opencontainers.image.source="https://github.com/gradium-ai/xn-ptts" \
      org.opencontainers.image.description="OpenAI-compatible text-to-speech server for Phonon"
USER phonon
# Every flag has an environment variable (`--help` lists them), so `docker run -e PTTS_LANG=fr`
# or a compose file's `environment:` configures the server. q8 because this image runs on the
# CPU, where it is faster than f32.
ENV PTTS_CONFIG=/models/pocket-tts \
    PTTS_LANG=en \
    PTTS_QUANT=q8
EXPOSE 8880
ENTRYPOINT ["ptts-openai-server"]
