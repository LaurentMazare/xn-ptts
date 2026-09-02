//! Streaming ASR (Graphon), end to end in one process.
//!
//! ```bash
//! export HF_TOKEN=...   # read access to the gr4d repos
//! cargo run --release --example asr --features sp -- sample.wav
//! ```

#[path = "audio_helpers.rs"]
mod audio_helpers;

use anyhow::{Context, Result};
use clap::Parser;
use ptts::asr_lm::{Config, LmModel};
use ptts::mimi::{MimiConfig, MimiEncoder};
use ptts::mimi_quantizer::MimiQuantizer;
use xn::nn::VB;
use xn::{Tensor, Unquantized};

/// Samples per model step: 80 ms at 24 kHz.
const SAMPLE_RATE: usize = 24_000;
const FRAME_RATE: f64 = 12.5;
const FRAME_SIZE: usize = 1920;
/// Frames of silence prepended before the audio.
const INITIAL_SILENCE_FRAMES: usize = 2;

/// Text tokens that close the current word.
const TOKEN_EOP: u32 = 0;
const TOKEN_EOS: u32 = 2;
const TOKEN_PAD: u32 = 3;
const TOKEN_SILENCE_PAD: u32 = 4;

#[derive(Parser, Debug)]
#[command(name = "asr")]
#[command(about = "Transcribe an audio file with the phonon ASR model")]
struct Args {
    /// Audio file to transcribe (any format symphonia can decode).
    audio: std::path::PathBuf,

    /// HuggingFace repo holding config.json, mimi.safetensors, model.safetensors
    /// and tokenizer.model.
    #[arg(long, default_value = "gr4d/asr-23b5a198.500")]
    repo: String,

    /// Local overrides for the files that would otherwise come from `--repo`.
    #[arg(long)]
    config: Option<std::path::PathBuf>,

    #[arg(long)]
    mimi_weights: Option<std::path::PathBuf>,

    #[arg(long)]
    lm_weights: Option<std::path::PathBuf>,

    #[arg(long)]
    tokenizer: Option<std::path::PathBuf>,

    /// Pin the spoken language, e.g. `en` or `fr`. Defaults to the model's
    /// learnt padding, which lets it decide on its own.
    #[arg(short, long)]
    language: Option<String>,

    /// Sampling temperature. 0 is greedy and reproducible.
    #[arg(short, long, default_value_t = 0.0)]
    temperature: f32,

    /// Frames of silence appended to flush the model's delay. Defaults to the
    /// delay itself.
    #[arg(long)]
    flush_frames: Option<usize>,

    /// Print word timings alongside the transcript.
    #[arg(long)]
    timestamps: bool,

    /// Print the extra (VAD) head probabilities for every step.
    #[arg(long)]
    vad: bool,

    /// Use the cpu backend even if a gpu one is available.
    #[arg(long)]
    cpu: bool,

    /// CPU worker threads. Defaults to `RAYON_NUM_THREADS` if set, otherwise
    /// one per core. Only affects the cpu backend.
    #[arg(long)]
    threads: Option<usize>,

    /// Quantize the LM weights on load: q8_0, q4_0, q4k, ... Implies `--cpu`,
    /// quantized kernels are cpu only.
    #[arg(long)]
    quant: Option<String>,
}

struct Files {
    config: std::path::PathBuf,
    mimi: std::path::PathBuf,
    lm: std::path::PathBuf,
    tokenizer: std::path::PathBuf,
}

impl Files {
    fn resolve(args: &Args) -> Result<Self> {
        // Everything already supplied locally? Then never touch the network.
        if let (Some(config), Some(mimi), Some(lm), Some(tokenizer)) = (
            args.config.clone(),
            args.mimi_weights.clone(),
            args.lm_weights.clone(),
            args.tokenizer.clone(),
        ) {
            return Ok(Self { config, mimi, lm, tokenizer });
        }
        use hf_hub::api::sync::ApiBuilder;
        use hf_hub::{Repo, RepoType};
        // The gr4d repos are private, so the token matters. HF_TOKEN wins over
        // whatever is cached in ~/.cache/huggingface/token.
        let mut builder = ApiBuilder::from_env();
        if let Ok(token) = std::env::var("HF_TOKEN") {
            builder = builder.with_token(Some(token));
        }
        let api = builder.build()?;
        let repo = api.repo(Repo::new(args.repo.clone(), RepoType::Model));
        let get = |name: &str, local: &Option<std::path::PathBuf>| -> Result<std::path::PathBuf> {
            match local {
                Some(path) => Ok(path.clone()),
                None => {
                    tracing::info!(repo = args.repo, name, "downloading");
                    repo.get(name).with_context(|| format!("fetching {name} from {}", args.repo))
                }
            }
        };
        Ok(Self {
            config: get("config.json", &args.config)?,
            mimi: get("mimi.safetensors", &args.mimi_weights)?,
            lm: get("model.safetensors", &args.lm_weights)?,
            tokenizer: get("tokenizer.model", &args.tokenizer)?,
        })
    }
}

/// The checkpoints wrap convs and attention one level deeper than `ptts` does.
/// Renaming on load lets the example reuse `ptts::mimi::MimiEncoder` as is
/// instead of restating the whole SEANet encoder.
fn remap_key(name: &str) -> Option<String> {
    let name = name.replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight");
    let name = name.replace(".conv.conv.", ".conv.");
    Some(name)
}

/// Mimi encoder v0.1 at 24 kHz with 32 codebooks, the tokenizer the ASR was trained
/// against.
fn mimi_config() -> MimiConfig {
    MimiConfig {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        frame_rate: FRAME_RATE,
        dimension: 512,
        quantizer_dimension: 256,
        quantizer_output_dimension: 512,
        n_filters: 64,
        n_residual_layers: 1,
        ratios: vec![8, 6, 5, 4],
        kernel_size: 7,
        last_kernel_size: 3,
        residual_kernel_size: 3,
        dilation_base: 2,
        compress: 2,
        transformer_d_model: 512,
        transformer_num_heads: 8,
        transformer_num_layers: 8,
        transformer_layer_scale: 0.01,
        transformer_context: 250,
        transformer_max_period: 10000.0,
        transformer_dim_feedforward: 2048,
        downsample_channel_wise: false,
    }
}

/// Turns the LM's text stream into words.
///
/// The model emits one text token per 80 ms frame; `EOP`/`PAD` close the word
/// that came before them. The first `delay` steps are dropped, that is the
/// lookahead the model was trained with.
struct WordDecoder {
    delay: usize,
    /// 0-based index of the next token to be pushed.
    step_idx: usize,
    word_tokens: Vec<u32>,
    open_words: usize,
    last_stop_time: f64,
}

enum Event {
    Word { tokens: Vec<u32>, start_time: f64 },
    EndWord { stop_time: f64 },
    EndOfStream,
}

impl WordDecoder {
    fn new(delay: usize) -> Self {
        Self { delay, step_idx: 0, word_tokens: vec![], open_words: 0, last_stop_time: 0.0 }
    }

    /// Where in the *input* audio the token at `step_idx` points. The model
    /// reports with `delay` frames of lookahead, and the stream it reads has
    /// `INITIAL_SILENCE_FRAMES` of silence prepended that the caller's file
    /// does not have.
    fn token_time(&self, step_idx: usize) -> f64 {
        let frames = step_idx as f64 - self.delay as f64 - INITIAL_SILENCE_FRAMES as f64;
        (frames / FRAME_RATE).max(0.0)
    }

    fn push(&mut self, token: u32) -> Vec<Event> {
        let step_idx = self.step_idx;
        self.step_idx += 1;
        let mut events = vec![];
        if step_idx < self.delay {
            return events;
        }
        let closes_word = matches!(token, TOKEN_EOP | TOKEN_EOS | TOKEN_PAD | TOKEN_SILENCE_PAD);
        if closes_word {
            if !self.word_tokens.is_empty() {
                let tokens = std::mem::take(&mut self.word_tokens);
                self.open_words += 1;
                events.push(Event::Word { tokens, start_time: self.last_stop_time });
            }
        } else {
            self.word_tokens.push(token);
        }
        if token == TOKEN_EOP || token == TOKEN_EOS {
            let stop_time = self.token_time(step_idx);
            if self.open_words > 0 {
                self.open_words = 0;
                events.push(Event::EndWord { stop_time });
            }
            self.last_stop_time = stop_time;
        }
        if token == TOKEN_EOS {
            events.push(Event::EndOfStream);
        }
        events
    }
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, prelude::*};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::Layer::new().with_target(false))
        .with(filter)
        .init();
}

fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing();
    if let Some(threads) = args.threads {
        anyhow::ensure!(threads > 0, "--threads must be at least 1");
        // Has to happen before any tensor work: this sets RAYON_NUM_THREADS,
        // which rayon only reads when it builds its global pool.
        xn::set_num_threads(threads);
    }

    #[cfg(feature = "cuda")]
    {
        if args.cpu || args.quant.is_some() {
            run_cpu(args)?;
        } else {
            tracing::info!("using cuda backend");
            let dev = xn::cuda_backend::Device::new(0)?;
            unsafe {
                dev.disable_event_tracking();
            }
            run::<Unquantized<half::bf16, _>>(args, dev)?;
        }
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        if args.cpu || args.quant.is_some() {
            run_cpu(args)?;
        } else {
            tracing::info!("using metal backend");
            let dev = xn::metal_backend::Device::new(0)?;
            run::<Unquantized<half::bf16, _>>(args, dev)?;
        }
    }
    #[cfg(all(feature = "vulkan", not(any(feature = "cuda", feature = "metal"))))]
    {
        if args.cpu || args.quant.is_some() {
            run_cpu(args)?;
        } else {
            tracing::info!("using vulkan backend");
            let dev = xn::vulkan_backend::Device::new(0)?;
            run::<Unquantized<f32, _>>(args, dev)?;
        }
    }
    #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
    {
        run_cpu(args)?;
    }
    Ok(())
}

/// Quantized weights only have cpu kernels, so `--quant` lands here whatever
/// the build.
fn run_cpu(args: Args) -> Result<()> {
    tracing::info!(
        quant = args.quant.as_deref().unwrap_or("f32"),
        threads = xn::get_num_threads(),
        cores = xn::get_num_cpus(),
        "using cpu backend"
    );
    match args.quant.as_deref() {
        None => run::<Unquantized<f32, _>>(args, xn::CPU),
        Some("q8" | "q8_0") => run::<xn::quantized::Q80F32>(args, xn::CPU),
        Some("q8k") => run::<xn::quantized::Q8kF32>(args, xn::CPU),
        Some("q6k") => run::<xn::quantized::Q6kF32>(args, xn::CPU),
        Some("q5" | "q5_0") => run::<xn::quantized::Q50F32>(args, xn::CPU),
        Some("q5k") => run::<xn::quantized::Q5kF32>(args, xn::CPU),
        Some("q4" | "q4_0") => run::<xn::quantized::Q40F32>(args, xn::CPU),
        Some("q4k") => run::<xn::quantized::Q4kF32>(args, xn::CPU),
        Some(other) => anyhow::bail!("unsupported quantization option '{other}'"),
    }
}

fn run<Q: xn::BackendQ>(args: Args, dev: Q::B) -> Result<()> {
    use std::io::Write;

    let files = Files::resolve(&args)?;
    let cfg: Config = serde_json::from_str(&std::fs::read_to_string(&files.config)?)
        .context("parsing config.json")?;
    tracing::info!(
        model = cfg.model_ext(),
        codebooks = cfg.n_q,
        layers = cfg.num_layers,
        dim = cfg.dim,
        delay_in_tokens = cfg.asr_delay_in_tokens,
        "loaded config"
    );

    let tokenizer = sentencepiece::SentencePieceProcessor::open(&files.tokenizer)
        .map_err(|e| anyhow::anyhow!("opening tokenizer: {e}"))?;

    // Mimi stays in f32 whatever the LM is quantized to, matching the TTS path.
    let mimi_vb = VB::load_with_key_map(&[&files.mimi], dev.clone(), remap_key)?.root();
    let mimi_cfg = mimi_config();
    let mimi: MimiEncoder<Unquantized<f32, Q::B>> = MimiEncoder::load(&mimi_vb, &mimi_cfg)?;
    let quantizer: MimiQuantizer<f32, Q::B> = MimiQuantizer::load(
        &mimi_vb.pp("quantizer"),
        mimi_cfg.dimension,
        mimi_cfg.quantizer_dimension,
        cfg.n_q,
        2048,
    )?;
    tracing::info!("mimi encoder loaded");

    let lm_vb = VB::load_with_key_map(&[&files.lm], dev.clone(), remap_key)?.root();
    let lm: LmModel<Q> = LmModel::load(&lm_vb, &cfg)?;
    tracing::info!("lm loaded");

    // The ASR is conditioned on the spoken language. With none given the model
    // gets the learnt padding and picks for itself.
    let condition = match lm.conditioners.get("languages_in_segment") {
        None => None,
        Some(conditioner) => {
            if let Some(language) = args.language.as_deref() {
                anyhow::ensure!(
                    conditioner.possible_values().iter().any(|v| v == language),
                    "unknown language {language:?}, expected one of {:?}",
                    conditioner.possible_values()
                );
            }
            Some(conditioner.condition(args.language.as_deref())?)
        }
    };

    let (pcm, sample_rate) = audio_helpers::pcm_decode(&args.audio)?;
    let audio_duration = pcm.len() as f64 / sample_rate as f64;
    tracing::info!(samples = pcm.len(), sample_rate, duration_s = audio_duration, "decoded audio");
    let pcm = if sample_rate as usize == SAMPLE_RATE {
        pcm
    } else {
        audio_helpers::resample(&pcm, sample_rate as usize, SAMPLE_RATE)?
    };

    let flush_frames = args.flush_frames.unwrap_or(cfg.asr_delay_in_tokens);
    let pcm =
        [vec![0.0; FRAME_SIZE * INITIAL_SILENCE_FRAMES], pcm, vec![0.0; FRAME_SIZE * flush_frames]]
            .concat();

    let mut enc_state = mimi.init_state(1, 1)?;
    let mut lm_state = lm.init_state()?;
    let mut decoder = WordDecoder::new(cfg.asr_delay_in_tokens);
    let mut text_token = lm.text_start_token();
    let audio_pad = vec![lm.audio_pad_token(); cfg.n_q];

    // SentencePiece needs the separators to place spaces, so the transcript is
    // re-decoded from scratch each time and only the new tail is printed.
    let mut all_tokens: Vec<u32> = vec![];
    let mut printed = 0usize;
    // (start, stop, word). `stop` stays open until the model closes the word.
    let mut timings: Vec<(f64, Option<f64>, String)> = vec![];

    let num_frames = pcm.len() / FRAME_SIZE;
    tracing::info!(num_frames, "transcribing");
    let start = std::time::Instant::now();
    'outer: for frame_idx in 0..num_frames {
        let frame = &pcm[frame_idx * FRAME_SIZE..(frame_idx + 1) * FRAME_SIZE];
        let audio = Tensor::from_vec(frame.to_vec(), (1, 1, FRAME_SIZE), &dev)?;
        let latent = mimi.encode_to_latent_step(&audio, &mut enc_state)?;
        let codes: Vec<u32> =
            quantizer.encode(&latent)?.to_vec()?.into_iter().map(|c| c as u32).collect();
        // The first slice is dropped: the encoder has no history yet, so the
        // model is handed pad tokens instead.
        let audio_tokens = if frame_idx == 0 { &audio_pad } else { &codes };

        let (logits, ys) = lm.step(&mut lm_state, text_token, audio_tokens, condition.as_ref())?;
        if args.vad && !cfg.vad_horizons.is_empty() {
            let prs = lm.extra_heads(&ys)?;
            let prs: Vec<String> = cfg
                .vad_horizons
                .iter()
                .zip(prs.iter())
                .map(|(h, p)| format!("{h}s={p:.2}"))
                .collect();
            tracing::info!(frame_idx, "vad {}", prs.join(" "));
        }
        let logits = logits.reshape((1, cfg.text_card))?;
        let sampled = xn::nn::sampling::gumbel_max(&logits, args.temperature, 1)?;
        text_token = sampled.to_vec()?[0] as u32;

        for event in decoder.push(text_token) {
            match event {
                Event::Word { tokens, start_time } => {
                    all_tokens.push(TOKEN_PAD);
                    all_tokens.extend_from_slice(&tokens);
                    let text = tokenizer.decode_piece_ids(&all_tokens).unwrap_or_default();
                    // `get` rather than a slice: the index comes from a shorter
                    // decode of the same tokens and is not guaranteed to land on
                    // a char boundary.
                    let tail = text.get(printed..).unwrap_or_default();
                    if !args.timestamps {
                        print!("{tail}");
                        std::io::stdout().flush()?;
                    }
                    timings.push((start_time, None, tail.trim().to_string()));
                    printed = text.len();
                }
                Event::EndWord { stop_time } => {
                    for open in timings.iter_mut().filter(|(_, stop, _)| stop.is_none()) {
                        open.1 = Some(stop_time);
                    }
                }
                Event::EndOfStream => {
                    tracing::info!(frame_idx, "model signalled end of stream");
                    break 'outer;
                }
            }
        }
    }
    if !args.timestamps {
        println!();
    }

    let elapsed = start.elapsed().as_secs_f64();
    if args.timestamps {
        for (start_time, stop_time, word) in timings.iter() {
            match stop_time {
                Some(stop_time) => println!("[{start_time:7.2} {stop_time:7.2}] {word}"),
                None => println!("[{start_time:7.2}       -] {word}"),
            }
        }
        println!("{}", tokenizer.decode_piece_ids(&all_tokens).unwrap_or_default());
    }
    tracing::info!(
        elapsed_s = elapsed,
        realtime_factor = audio_duration / elapsed,
        "done ({num_frames} frames)"
    );
    Ok(())
}
