"""The host side of the packaged model: the KV cache, the positions, and the EOS rule.

`run` is generic over how a graph is executed (`call(name, *inputs) -> outputs`),
so the same loop drives the eager PyTorch modules and the compiled model. It
follows `ptts`'s `synth::run_backbone` for one text chunk:

1. The voice prompt fills cache slots 0..V-1 (`prompt_audio`; here a
   precomputed KV cache, see `voice_cache`).
2. The chunk's text tokens fill the next n slots in one prefill (`prompt_text`).
3. Steps from BOS: each feeds the previous latent, and emits a latent and an EOS
   logit. A frame whose logit is above `eos_threshold` (4.0) starts a countdown of
   `frames_after_eos` (3 for texts of up to 4 words, else 1); the EOS frame and
   that many more are output, then generation stops (`plan::EosPolicy`).
   At most `frame_budget(n)` frames.
4. Mimi starts from zero states for every chunk.
"""

import math
from dataclasses import dataclass

import torch

CACHE = 512  # flow LM KV slots: voice prompt (125) + text (<= 64) + generated frames
PREFILL = 64  # text tokens per prefill call; ptts chunks text at 50 tokens
TEMPERATURE = 0.3  # ptts's default (SynthBuilder, the ptts example); config.json's `temp` is not read


@dataclass
class Dims:
    layers: int
    heads: int
    head_dim: int
    d_model: int
    ldim: int
    max_period: float
    mimi_heads: int
    mimi_head_dim: int
    mimi_window: int
    mimi_steps: int
    mimi_max_period: float
    mimi_states: list
    samples_per_frame: int
    frame_rate: float
    eos_threshold: float


def int32(x: int) -> torch.Tensor:
    return torch.tensor([x], dtype=torch.int32)


def frame_budget(num_tokens: int, frame_rate: float = 12.5) -> int:
    """`plan::frame_budget`: the most frames a chunk of `num_tokens` may generate."""
    return math.ceil((num_tokens / 3.0 + 2.0) * frame_rate)


def prepare_text_prompt(text: str) -> tuple[str, int]:
    """`tts_model::prepare_text_prompt`: (text, frames_after_eos). Runs after normalization."""
    text = text.strip()
    if not text:
        return text, 3
    text = " ".join(text.replace("\n", " ").replace("\r", " ").split())
    frames_after_eos = 3 if len(text.split()) <= 4 else 1
    text = text[0].upper() + text[1:]
    if text[-1].isalnum():
        text += "."
    return text, frames_after_eos


@torch.no_grad()
def voice_cache(phonon, emb: torch.Tensor, dtype=torch.float16) -> tuple[torch.Tensor, int]:
    """A voice prompt [1, V, E] -> its KV cache [2L, H, CACHE, D] in the graphs' layout, and V.

    Runs the shared backbone in fp32 from an empty cache, as `prompt_audio` does.
    """
    d = phonon.dims
    n = emb.shape[1]
    zeros = torch.zeros(2 * d.layers, d.heads, CACHE, d.head_dim)
    cos, sin, mask = phonon.tables.flow(int32(0), n)
    _, kv_new = phonon.backbone(emb.float(), zeros, cos, sin, mask)
    kv = torch.zeros(2 * d.layers, d.heads, CACHE, d.head_dim, dtype=dtype)
    kv[:, :, :n] = kv_new.to(dtype)
    return kv, n


def mimi_zero_states(dims: Dims, dtype=torch.float16) -> list[torch.Tensor]:
    return [torch.zeros(s, dtype=dtype) for s in dims.mimi_states]


def noise_source(ldim: int, temperature: float = TEMPERATURE, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    std = math.sqrt(temperature)
    return lambda: torch.randn(1, ldim, generator=g) * std


def run(call, dims: Dims, kv, pos, tokens, bos_emb, noises, max_frames=None, frames_after_eos=1,
        eos_threshold=None, io_dtype=torch.float16, record=None, stop_at_eos=True, teacher=None, input_linear=None):
    """Generate audio for `tokens` [1, n] after a voice cache `kv` of length `pos`.

    `noises[i]` is frame i's noise [1, ldim], already scaled by sqrt(temperature).
    `kv` must be in `io_dtype` and is written in place.

    Returns a dict: latents [N, ldim], audio [samples], eos_logits [N], eos_step,
    step_inputs [N, E] (each step's input embedding before the frame bias).
    With `record`, a dict, appends every call's inputs under the graph's name.
    With `teacher` (latents [N, ldim]) and `input_linear` (a callable), step i + 1 is
    fed input_linear(teacher[i]) instead of the graph's own next_emb: teacher forcing,
    which keeps a free run's drift from compounding when comparing per step.
    """
    n = tokens.shape[1]
    if max_frames is None:
        max_frames = frame_budget(n, dims.frame_rate)
    if eos_threshold is None:
        eos_threshold = dims.eos_threshold
    assert n <= PREFILL and pos + PREFILL <= CACHE and pos + n + max_frames <= CACHE

    def invoke(name, *inputs):
        if record is not None:
            record.setdefault(name, []).append([t.clone() for t in inputs])
        return call(name, *inputs)

    padded = torch.zeros(1, PREFILL, dtype=torch.int32)
    padded[:, :n] = tokens
    (kv_new,) = invoke("prefill", padded, kv, int32(pos))
    kv[:, :, pos : pos + n] = kv_new[:, :, :n]
    pos += n

    states = mimi_zero_states(dims, io_dtype)
    emb = bos_emb.to(io_dtype)
    latents, audio, eos_logits, inputs = [], [], [], []
    eos_step, countdown = None, None
    for frame in range(max_frames):
        inputs.append(emb.reshape(-1).float())
        chunk, eos, emb, kv_new, latent, *states = invoke(
            "step", emb, noises[frame].to(io_dtype), kv, int32(pos), int32(frame), *states)
        kv[:, :, pos : pos + 1] = kv_new
        pos += 1
        if teacher is not None and frame < len(teacher):
            emb = input_linear(teacher[frame][None].float())[:, None].to(io_dtype)
        eos_logits.append(float(eos))
        latents.append(latent.float())
        audio.append(chunk.reshape(-1).float())
        # plan::EosPolicy: the EOS frame and frames_after_eos more are output.
        if float(eos) > eos_threshold and countdown is None:
            eos_step, countdown = frame, frames_after_eos
        if stop_at_eos and countdown is not None:
            if countdown == 0:
                break
            countdown -= 1
    return {
        "latents": torch.cat(latents),
        "audio": torch.cat(audio),
        "eos_logits": torch.tensor(eos_logits),
        "eos_step": eos_step,
        "step_inputs": torch.stack(inputs),
    }
