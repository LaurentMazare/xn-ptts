# Phonon WebSocket server

Run with a checkpoint folder or Hugging Face repo ID:

```bash
cargo run --release -p ptts-ws-server -- \
  --config "$MODEL_DIR" --quant q8 --lang en
```

The server listens on `0.0.0.0:8080`, with WebSocket connections at `/speech/tts`. `--addr` changes the address. The checkpoint supplies its config, tokenizer, weights, and voices; `--revision` selects a Hub revision. Building requires libopus (`apt-get install libopus-dev` or `brew install opus`).

By default, one WebSocket session is accepted at a time. `--max-concurrent-requests N` sets a positive limit. Excess connections receive HTTP 429 before the WebSocket upgrade, with a message asking the client to retry after an active session closes.

A session retains its primed voice and KV cache between messages. Its slot is held for the entire connection, including while idle, and released after the session and generation workers stop. Close connections when finished so other clients can connect. The limit applies to connections, including those awaiting their first setup message.
