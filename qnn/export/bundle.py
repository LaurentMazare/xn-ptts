"""Assemble the Phonon bundle: model, text front end, voices and metadata.json.

    python bundle.py [--out bundle]

The layout qnn/runner reads:

    metadata.json      model, voices, graph interface, generation settings
    phonon.bin         QNN HTP context binary with the `prefill` and `step` graphs
    cpu/prefill.dlc    the graphs for QNN's CPU backend (fp32 I/O)
    cpu/step.dlc
    tokenizer.json     the checkpoint's own tokenizer
    lib/<platform>/libptts_text.so
                       ptts's text front end (normalization, chunking, tokenization)
    bos.bin            fp16 [1, 1, d_model]: step's `emb` for the first frame (step adds
                       the frame bias itself)
    voices/<name>.bin  fp16 [2L, H, n, D]: the voice's keys and values, for slots 0..n-1

Needs out/package.json, out/phonon.bin (or the context binary package.py downloaded),
out/cpu/*.dlc and the builds of qnn/text-ffi (its build.sh) in qnn/text-ffi/dist/.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import qai_hub as hub
import torch

from host import CACHE, PREFILL, TEMPERATURE, voice_cache
from load import CHECKPOINT, VOICES, load, voice_embedding, voice_latents

OUT = Path("out")
TEXT_FFI = Path(__file__).resolve().parent.parent / "text-ffi/dist"  # qnn/text-ffi/build.sh


def context_binary() -> Path:
    for name in ("phonon.bin", "model.bin"):
        if (OUT / name).exists():
            return OUT / name
    found = sorted(OUT.glob("*.bin"))
    if not found:
        raise SystemExit("no context binary in out/: run package.py")
    return found[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bundle")
    ap.add_argument("--soc-model", required=True, help="compiled target, e.g. SM8750; must match package.py target")
    args = ap.parse_args()
    out = Path(args.out)
    for d in ("voices", "cpu", "lib/android-arm64", "lib/linux-arm64"):
        (out / d).mkdir(parents=True, exist_ok=True)
    jobs = json.loads((OUT / "package.json").read_text())

    phonon = load(io_half=True)
    dims = phonon.dims
    shutil.copy(context_binary(), out / "phonon.bin")
    for f in ("prefill.dlc", "step.dlc"):
        shutil.copy(OUT / "cpu" / f, out / "cpu" / f)
    shutil.copy(CHECKPOINT / "tokenizer.json", out / "tokenizer.json")
    shutil.copy(TEXT_FFI / "android-arm64-v8a/libptts_text.so", out / "lib/android-arm64/libptts_text.so")
    shutil.copy(TEXT_FFI / "linux-arm64/libptts_text.so", out / "lib/linux-arm64/libptts_text.so")
    phonon.bos_emb.half().numpy().tofile(out / "bos.bin")

    names = sorted(p.stem for p in VOICES.glob("*.safetensors")) + ["default"]
    voices = []
    for name in names:
        emb = voice_embedding(phonon.weights, voice_latents(name))
        kv, n = voice_cache(phonon, emb, torch.float16)
        np.ascontiguousarray(kv[:, :, :n].numpy()).tofile(out / "voices" / f"{name}.bin")
        voices.append({"name": name, "display_name": name.capitalize(), "language": "en", "language_name": "English",
                       "sample_rate": phonon.config["mimi"]["sample_rate"], "file": f"voices/{name}.bin",
                       "length": n, "description": f"Phonon voice '{name}'"})
        print("voice", name, n, flush=True)

    graph_io = {}
    for name, jid in jobs["compile"].items():
        specs = hub.get_job(jid).get_target_shapes()
        graph_io[name] = {"inputs": [{"name": k, "shape": list(s), "dtype": d} for k, (s, d) in specs.items()]}

    fl, mc = phonon.config["flow_lm"], phonon.config["mimi"]
    frame_rate = mc["frame_rate"]
    metadata = {
        "name": "phonon_7e71a02d_en",
        "display_name": "Phonon 7e71a02d (English)",
        "version": "0.1.0",
        "description": "Phonon 7e71a02d for the Snapdragon NPU: a flow-matching language model generating "
                       "Mimi latents, decoded to 24 kHz audio, one 80 ms frame per step.",
        "model_type": "phonon",
        "voices": voices,
        "runtime": {
            "language": "en",
            "qnn_version": {"major": 2, "minor": 50, "patch": 0},
            "qairt_build": "2.50.0.260828",
            "arch_bit": 64,
            "precision": "fp16",
            "context_binaries": {jobs["device"]: {"file": "phonon.bin", "soc_models": [args.soc_model.upper()], "ai_hub_model": jobs.get("model")}},
            "dlcs": {"prefill": "cpu/prefill.dlc", "step": "cpu/step.dlc"},
            "dlc_io": "float32: the CPU backend cannot run fp16 casts; a runner converts from its fp16 buffers",
        },
        "graphs": graph_io,
        "generation": {
            "sample_rate": mc["sample_rate"],
            "frame_samples": int(mc["sample_rate"] / frame_rate),
            "frame_rate": frame_rate,
            "cache_slots": CACHE,
            "prefill_tokens": PREFILL,
            "max_tokens_per_chunk": 50,
            "temperature": TEMPERATURE,
            "noise_std": math.sqrt(TEMPERATURE),
            "eos_threshold": phonon.config["eos_threshold"],
            "min_frames_before_eos": 0,
            # Frames played from the EOS frame on, that one included (ptts: the EOS frame plus 3 or 1).
            "frames_after_eos": {"words_at_most_4": 4, "otherwise": 2},
            "max_frames": "ceil((tokens / 3 + 2) * frame_rate)",
            "bos": {"file": "bos.bin", "shape": [1, 1, fl["d_model"]], "dtype": "float16"},
            "conditions": {"num_speakers": "1", "note": "baked into step as a per-frame bias"},
            "layout": {"kv_cache": "[2 * layers, heads, slots, head_dim]; layer l keys at 2l, values at 2l + 1",
                       "layers": fl["num_layers"], "heads": fl["num_heads"],
                       "head_dim": fl["d_model"] // fl["num_heads"]},
        },
        "text": {
            "frontend": "ptts_text",
            "library": {"android-arm64": "lib/android-arm64/libptts_text.so",
                        "linux-arm64": "lib/linux-arm64/libptts_text.so"},
            "lang": "en",
            "tokenizer": "tokenizer.json",
        },
        "source": {"model": "Phonon", "sig": "7e71a02d", "epoch": 200},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"bundle {out}: {size / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
