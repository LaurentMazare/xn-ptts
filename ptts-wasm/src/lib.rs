//! The raw WebAssembly surface of the browser build.
//!
//! This is what `wasm-bindgen` exports, and it is deliberately low level: it takes bytes the
//! caller has already fetched and it generates one frame per call, because the worker it runs
//! in must yield to its event loop between frames to hear a cancel. The `threads` build
//! splits the work inside a frame across Web Workers; see `start_cpu_pool`. A
//! JavaScript wrapper -- the `phonon-tts` npm package, added later in this stack -- runs it
//! in a worker and handles downloads, caching and voices by name. Most callers want that,
//! not this.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format!($($t)*)))
}

use ptts::flow_lm::{FlowLMState, NormalRng, StepInput};
use ptts::loader::{load_speaker_proj, load_voice_emb_from_bytes, remap_key};
use ptts::mimi::MimiDecoderState;
use ptts::plan::{self, EosPolicy};
use ptts::preprocess::{Normalize, Rules};
use ptts::tok::Tok;
use ptts::transformer::{LayerAttentionState, StreamingMHAState, StreamingTransformerState};
use ptts::tts_model::{MAX_TOKENS_PER_CHUNK, TTSConfig, TTSModel, TTSState};
use xn::nn::{Linear, VB};
use xn::quantized::Q80F32;
use xn::{BackendQ, CPU, CpuDevice, Result, Tensor, TypedTensor, Unquantized};

/// Underlying type-erased transformer state, shared across all supported quantizations
/// (all of them use `T = f32, B = CpuDevice`).
type RawState = StreamingTransformerState<f32, CpuDevice>;

fn wrap_state<Q: BackendQ<T = f32, B = CpuDevice>>(raw: RawState) -> TTSState<Q> {
    TTSState { flow_lm_state: FlowLMState { transformer_state: raw } }
}

/// Slots a voice state already occupies: the voice prompt's frames. Every flow-LM layer
/// advances together, so the first one says it for all of them.
fn raw_len(state: &RawState) -> usize {
    state
        .layer_states
        .iter()
        .find_map(|layer| match layer {
            LayerAttentionState::FlowLm(mha) => Some(mha.current_end),
            _ => None,
        })
        .unwrap_or(0)
}

/// Quantization variants exposed to JS.
#[derive(Clone, Copy, Debug)]
enum Quant {
    F32,
    Q8,
}

impl Quant {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "f32" => Ok(Self::F32),
            "q8" => Ok(Self::Q8),
            other => xn::bail!("unsupported quantization '{other}', expected 'f32' or 'q8'"),
        }
    }
}

enum ModelInner {
    F32(TTSModel<Unquantized<f32, CpuDevice>>),
    Q8(TTSModel<Q80F32>),
}

enum StateInner {
    F32(TTSState<Unquantized<f32, CpuDevice>>),
    Q8(TTSState<Q80F32>),
}

/// Run a block against the active model. Within the block, `$m` is `&TTSModel<Q>`.
macro_rules! with_model {
    ($inner:expr, |$m:ident| $body:expr) => {
        match $inner {
            ModelInner::F32($m) => $body,
            ModelInner::Q8($m) => $body,
        }
    };
}

/// Dispatch a block of code over the currently active (model, state) pair. Within the
/// block, `$m` is `&TTSModel<Q>` and `$s` is `&mut TTSState<Q>` for the matching `Q`.
macro_rules! dispatch {
    ($inner:expr, $state:expr, |$m:ident, $s:ident| $body:block) => {
        match ($inner, $state) {
            (ModelInner::F32($m), StateInner::F32($s)) => $body,
            (ModelInner::Q8($m), StateInner::Q8($s)) => $body,
            _ => xn::bail!("model/state quantization mismatch"),
        }
    };
}

/// The chunk currently being generated.
struct ChunkState {
    tts_state: StateInner,
    mimi_state: MimiDecoderState<f32, CpuDevice>,
    prev_latent: Option<Tensor<f32, CpuDevice>>,
    frame_budget: usize,
    eos: EosPolicy,
    step: usize,
    /// Set once `eos` has run out: the frame that did it has been returned, and the next
    /// call ends the chunk.
    done: bool,
}

struct GenState {
    /// The voice state, resized to fit the longest chunk. Every chunk starts from a clone of
    /// it, as `ptts::synth` does: chunks run one after the other, so sharing the KV storage
    /// is safe, and each overwrites only what lies past the voice prompt.
    base: RawState,
    chunks: std::vec::IntoIter<plan::Chunk>,
    current: Option<ChunkState>,
    /// One noise source for the whole text, so a seed fixes every chunk.
    rng: NormalRng,
}

#[wasm_bindgen]
pub struct Model {
    inner: ModelInner,
    cfg: TTSConfig,
    speaker_proj: Option<Linear<f32, CpuDevice>>,
    gen_state: Option<GenState>,
    voice_states: Vec<RawState>,
    /// How `start_generation` normalizes, named by the page when it built the
    /// model. See `Model::new`.
    normalize: Normalize,
}

impl Model {
    fn new_(
        model_weights: &[u8],
        tokenizer_json: &[u8],
        config_json: Option<Vec<u8>>,
        quant: &str,
        lang: &str,
        rewrites: Option<&str>,
    ) -> Result<Model> {
        let quant = Quant::parse(quant)?;
        let rules = match rewrites {
            Some(rewrites) => rewrites.parse::<Rules>()?,
            None => Rules::ALL,
        };
        let normalize = lang.parse::<Normalize>()?.with_rules(rules);
        let cfg = match config_json {
            Some(json) => match serde_json::from_slice(&json) {
                Ok(cfg) => cfg,
                Err(e) => xn::bail!("cannot parse config.json: {e}"),
            },
            None => TTSConfig::v202601(),
        };
        console_log!("[phonon] loading model with quant={quant:?}");

        let is_gguf = model_weights.len() >= 4 && &model_weights[..4] == b"GGUF";
        let vb = if is_gguf {
            let cursor = std::io::Cursor::new(model_weights.to_vec());
            VB::load_gguf_with_key_map(cursor, CPU, remap_key)?
        } else {
            VB::from_bytes_with_key_map(vec![model_weights.to_vec()], CPU, remap_key)?
        };
        let root = vb.root();
        let tokenizer: Box<dyn ptts::Tokenizer + Send + Sync> =
            Box::new(Tok::from_bytes(tokenizer_json)?);
        let speaker_proj = load_speaker_proj(&root, &cfg)?;

        let inner = match quant {
            Quant::F32 => {
                ModelInner::F32(TTSModel::load(&root, tokenizer, &cfg, &Default::default())?)
            }
            Quant::Q8 => {
                ModelInner::Q8(TTSModel::load(&root, tokenizer, &cfg, &Default::default())?)
            }
        };

        Ok(Model { inner, cfg, speaker_proj, gen_state: None, voice_states: Vec::new(), normalize })
    }

    fn add_voice_(&mut self, bytes: &[u8]) -> Result<usize> {
        let tensors = xn::safetensors::load_from_buffer(bytes, &CPU)?;
        let raw = if tensors.contains_key(&kv_cache_name(0)) {
            self.voice_from_kv_cache(&tensors)?
        } else {
            self.voice_from_emb(bytes)?
        };
        self.voice_states.push(raw);
        Ok(self.voice_states.len() - 1)
    }

    /// A voice stored as the flow LM's KV cache after the voice prompt, the format the
    /// `embeddings_v2/` voices use: `transformer.layers.{i}.self_attn/cache`, shaped
    /// `[2, 1, seq, heads, head_dim]`, for each layer. Nothing to run.
    fn voice_from_kv_cache(
        &self,
        tensors: &std::collections::HashMap<String, TypedTensor<CpuDevice>>,
    ) -> Result<RawState> {
        let num_layers = self.cfg.flow_lm.num_layers;
        let mut layer_states = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let cache_name = kv_cache_name(i);
            let cache = match tensors.get(&cache_name) {
                Some(TypedTensor::F32(t)) => t,
                _ => xn::bail!("expected f32 tensor: {cache_name}"),
            };
            let (two, batch, seq_len, num_heads, head_dim) = cache.dims5()?;
            if two != 2 {
                xn::bail!("{cache_name}: expected a first dim of size 2, got {two}");
            }
            let kv = |i: usize| -> Result<Tensor<f32, CpuDevice>> {
                cache
                    .narrow(0, i..i + 1)?
                    .contiguous()?
                    .reshape((batch, seq_len, num_heads, head_dim))
            };
            layer_states.push(LayerAttentionState::FlowLm(StreamingMHAState {
                k_cache: kv(0)?,
                v_cache: kv(1)?,
                current_end: seq_len,
            }));
        }
        Ok(StreamingTransformerState { layer_states })
    }

    /// A voice stored as an embedding (`emb`, `audio_prompt`) or as speaker-Mimi latents
    /// (`speaker_wavs`), the formats every other frontend reads. It is run through the flow
    /// LM once, here, so generation can start from the resulting state.
    fn voice_from_emb(&self, bytes: &[u8]) -> Result<RawState> {
        let model_ext = self.cfg.model_ext();
        let emb = load_voice_emb_from_bytes(
            bytes,
            model_ext.as_deref(),
            self.speaker_proj.as_ref(),
            &CPU,
        )?;
        let frames = emb.dim(1usize)?;
        with_model!(&self.inner, |m| {
            let mut state = m.init_flow_lm_state(1, frames)?;
            m.prompt_audio(&mut state, &emb)?;
            Ok(state.flow_lm_state.transformer_state)
        })
    }

    fn start_generation_(
        &mut self,
        voice_index: usize,
        text: &str,
        temperature: f32,
        seed: u64,
    ) -> Result<usize> {
        // Dropped before anything else can fail, so a caller that swallows the error cannot
        // go on stepping and quietly resume the *previous* utterance.
        self.gen_state = None;
        // Built here rather than after planning: a temperature that cannot produce a
        // distribution should be refused before any work is done.
        let rng = NormalRng::new(temperature, seed)?;
        let Some(voice) = self.voice_states.get(voice_index) else {
            xn::bail!("invalid voice index: {voice_index}")
        };
        let chunks = self.plan_chunks(text)?;

        // The KV budget has to hold the voice prompt plus the longest chunk's text and audio.
        // This is `plan::seq_budget` with the voice's real length in place of its
        // `PROMPT_SEQ_HEADROOM` guess, the same bound `ptts::synth` checks a session against.
        let voice_len = raw_len(voice);
        let seq_budget =
            chunks.iter().map(|c| voice_len + c.tokens.len() + c.frame_budget).max().unwrap_or(0);
        let base = voice.with_seq_budget(seq_budget)?;

        let num_chunks = chunks.len();
        self.gen_state = Some(GenState { base, chunks: chunks.into_iter(), current: None, rng });
        Ok(num_chunks)
    }

    /// The same chunks `ptts::synth` makes, from [`plan::chunks`].
    fn plan_chunks(&self, text: &str) -> Result<Vec<plan::Chunk>> {
        let frame_rate = self.cfg.mimi.frame_rate;
        with_model!(&self.inner, |m| {
            let Some(tokenizer) = m.flow_lm.conditioner.tokenizer.as_deref() else {
                xn::bail!("this model was loaded without a tokenizer")
            };
            Ok(plan::chunks(tokenizer, text, self.normalize, MAX_TOKENS_PER_CHUNK, frame_rate)?)
        })
    }

    fn next_chunk_(&mut self) -> Result<Option<usize>> {
        let Some(gen_state) = self.gen_state.as_mut() else { return Ok(None) };
        gen_state.current = None;
        let Some(chunk) = gen_state.chunks.next() else {
            self.gen_state = None;
            return Ok(None);
        };
        let raw = gen_state.base.clone();
        let mut tts_state = match &self.inner {
            ModelInner::F32(_) => StateInner::F32(wrap_state(raw)),
            ModelInner::Q8(_) => StateInner::Q8(wrap_state(raw)),
        };
        let mimi_state = dispatch!(&self.inner, &mut tts_state, |m, s| {
            m.prompt_text(s, &chunk.tokens)?;
            m.init_mimi_state(1)?
        });
        gen_state.current = Some(ChunkState {
            tts_state,
            mimi_state,
            prev_latent: None,
            frame_budget: chunk.frame_budget,
            eos: EosPolicy::new(chunk.frames_after_eos),
            step: 0,
            done: false,
        });
        Ok(Some(chunk.tokens.len()))
    }

    fn generation_step_(&mut self) -> Result<Option<js_sys::Float32Array>> {
        let Some(gen_state) = self.gen_state.as_mut() else { return Ok(None) };
        let Some(state) = gen_state.current.as_mut() else { return Ok(None) };
        if state.done || state.step >= state.frame_budget {
            gen_state.current = None;
            return Ok(None);
        }

        let rng = &mut gen_state.rng;
        let (next_latent, audio_chunk, is_eos) =
            dispatch!(&self.inner, &mut state.tts_state, |m, s| {
                // Inside the dispatch: `StepInput` is generic over the quantization,
                // so one built outside would pin this to a single arm.
                let input = match &state.prev_latent {
                    None => StepInput::Bos { batch: 1 },
                    Some(t) => StepInput::Latent(t),
                };
                let (next_latent, is_eos) = m.generate_step(s, input, rng)?;
                let audio_chunk = m.decode_latent(&next_latent, &mut state.mimi_state)?;
                (next_latent, audio_chunk, is_eos)
            });

        // `should_stop` is called after the frame has gone to the decoder: the EOS frame
        // itself is part of the output. The frame is returned either way; the next call
        // reports the end of the chunk.
        state.done = state.eos.should_stop(is_eos);
        state.prev_latent = Some(next_latent);
        state.step += 1;

        let pcm = audio_chunk.narrow(0, ..1)?.contiguous()?.to_vec()?;
        Ok(Some(js_sys::Float32Array::from(pcm.as_slice())))
    }
}

impl Model {
    /// A failed step leaves a chunk half prompted or half generated. Dropping the generation
    /// makes every later call report the end instead, so a caller that swallows the error
    /// cannot go on and silently skip a sentence. `start_generation_` does the same.
    fn drop_generation_on_error<T>(&mut self, result: &Result<T>) {
        if result.is_err() {
            self.gen_state = None;
        }
    }
}

fn kv_cache_name(layer: usize) -> String {
    format!("transformer.layers.{layer}.self_attn/cache")
}

fn js_err(e: xn::Error) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen]
impl Model {
    /// `model_weights` is a safetensors or GGUF checkpoint, `tokenizer_json` the contents of
    /// the `tokenizer.json` for its vocabulary, and `config_json` its `config.json`, or
    /// `undefined` for the original Pocket TTS architecture.
    ///
    /// One thing a config cannot ask this build for: classifier-free guidance. Guidance is a
    /// caller's option in `ptts::synth` (`SpeechOptions::cfg_coef`), not a field of the
    /// config, and the browser build never turns it on, so `cfg_null_audio_empty` is inert.
    /// Everything else -- the flow LM and Mimi shapes, `lsd_decode_steps`, `eos_threshold`,
    /// `model_id`, `speaker_mimi` -- is honored.
    ///
    /// `quant` is `"f32"` or `"q8"`.
    ///
    /// `lang` is required: the language text is normalized as before it is
    /// tokenized, one of `"en"`, `"fr"`, `"de"`, `"es"`, `"pt"`, or `"none"`
    /// to hand text to the tokenizer as written. The spoken forms of `@`, `+`
    /// and `=` differ per language, so there is nothing safe to default to.
    ///
    /// `rewrites` is optional and picks which word rewrites run on the
    /// normalized text: `"all"` (the default), `"none"`, or a comma-separated
    /// list of rule names, of which there is one today, `"numbers"`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        model_weights: &[u8],
        tokenizer_json: &[u8],
        config_json: Option<Vec<u8>>,
        quant: &str,
        lang: &str,
        rewrites: Option<String>,
    ) -> std::result::Result<Model, JsError> {
        Self::new_(model_weights, tokenizer_json, config_json, quant, lang, rewrites.as_deref())
            .map_err(js_err)
    }

    /// Registers a voice from a safetensors file and returns its index for
    /// `start_generation`. Either a precomputed KV cache (`embeddings_v2/`) or a voice
    /// embedding (`emb`, `audio_prompt` or `speaker_wavs`), told apart by tensor name.
    pub fn add_voice(&mut self, voice: &[u8]) -> std::result::Result<usize, JsError> {
        self.add_voice_(voice).map_err(js_err)
    }

    /// Normalizes `text`, splits it into sentence-aligned chunks and tokenizes them. Returns
    /// the number of chunks. Runs no model: call `next_chunk` to start the first one.
    pub fn start_generation(
        &mut self,
        voice_index: usize,
        text: &str,
        temperature: f32,
        seed: u64,
    ) -> std::result::Result<usize, JsError> {
        self.start_generation_(voice_index, text, temperature, seed).map_err(js_err)
    }

    /// Prompts the model with the next chunk's text and returns its token count, or
    /// `undefined` once every chunk has been generated.
    pub fn next_chunk(&mut self) -> std::result::Result<Option<usize>, JsError> {
        let result = self.next_chunk_();
        self.drop_generation_on_error(&result);
        result.map_err(js_err)
    }

    /// Generates and decodes one frame of the current chunk: 80 ms of mono PCM at
    /// `sample_rate`. Returns `undefined` when the chunk is finished.
    pub fn generation_step(
        &mut self,
    ) -> std::result::Result<Option<js_sys::Float32Array>, JsError> {
        let result = self.generation_step_();
        self.drop_generation_on_error(&result);
        result.map_err(js_err)
    }

    /// Drops the generation in progress, if any.
    pub fn stop_generation(&mut self) {
        self.gen_state = None;
    }

    pub fn sample_rate(&self) -> usize {
        with_model!(&self.inner, |m| m.sample_rate())
    }
}

/// CPU SIMD features the wasm module was compiled with. The relevant one
/// for browser builds is `simd128`; `avx`/`neon`/`f16c` are reported for
/// completeness so it's clear which native-target builds enabled them.
#[wasm_bindgen]
pub fn cpu_features() -> js_sys::Object {
    let obj = js_sys::Object::new();
    let set = |k: &str, v: bool| {
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str(k), &JsValue::from_bool(v));
    };
    set("avx", xn::with_avx());
    set("neon", xn::with_neon());
    set("simd128", xn::with_simd128());
    set("f16c", xn::with_f16c());
    obj
}

// ---- threads ----
//
// Only in the `threads` build. It is a separate module because wasm threads need shared
// memory, which needs std rebuilt with atomics and a page that is cross-origin isolated, and
// a page that is not would fail to load it at all. The JS side picks this build only when
// the page can run it, and the single-threaded one otherwise.
//
// Two steps, from the worker that owns the `Model`, before it loads one:
// `initThreadPool(workers)` gives rayon its Web Workers, then `start_cpu_pool(workers)`
// parks xn's CPU pool on them. A rayon fork/join wakes a parked Web Worker per operator,
// which costs more than most of a frame's operators take; xn's pool workers spin between
// operators instead, so a dispatch is cheap and small operators are worth splitting.

#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Iterations a pool worker spins before parking: long enough to bridge the gaps between
/// operators inside a frame, short enough that an idle page stops burning its cores soon
/// after an utterance ends.
#[cfg(feature = "threads")]
const POOL_SPIN_BUDGET: u32 = 4_000_000;

/// The smallest operator, in multiply-adds, that the pool splits across its workers. Below
/// it, handing work out costs more than doing it on one thread.
#[cfg(feature = "threads")]
const MIN_PARALLEL_WORK: usize = 256 << 10;

/// Runs xn's CPU pool on `workers` of rayon's Web Workers, for good: they never return, so
/// nothing else may go through rayon afterwards. Call it once, after `initThreadPool` and
/// before loading a model, from a worker rather than the page, since a dispatch can block.
/// Returns how many threads now share the work: the workers plus the calling one.
#[cfg(feature = "threads")]
#[wasm_bindgen]
pub fn start_cpu_pool(workers: usize) -> usize {
    if workers == 0 {
        return 1;
    }
    // A worker rayon cannot schedule would never take a job, and the first dispatch would
    // wait for it forever.
    let workers = workers.min(rayon::current_num_threads());
    let size = xn::threadpool::start_pool_with(
        xn::threadpool::PoolConfig { workers, spin_budget: Some(POOL_SPIN_BUDGET) },
        |job| rayon::spawn(job),
    );
    xn::threadpool::set_min_parallel_work(MIN_PARALLEL_WORK);
    size
}
