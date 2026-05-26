use anyhow::{Context as _, Result};
use ptts::tts_model::{TTSConfig, TTSModel, TTSState};
use std::collections::HashMap;
use std::sync::Arc;
use xn::nn::VB;
use xn::{BackendQ, Tensor};

pub const VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

pub const REPO_ID: &str = "kyutai/pocket-tts";
pub const MODEL_FILE: &str = "tts_b6369a24.safetensors";

pub struct StdRng {
    inner: rand::rngs::StdRng,
    distr: rand_distr::Normal<f32>,
}

impl StdRng {
    pub fn new(temperature: f32, seed: u64) -> Self {
        use rand::SeedableRng;
        let distr = rand_distr::Normal::new(0f32, temperature.sqrt()).unwrap();
        let inner = rand::rngs::StdRng::seed_from_u64(seed);
        Self { inner, distr }
    }
}

impl ptts::flow_lm::Rng for StdRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng;
        self.inner.sample(self.distr)
    }
}

pub enum Tok {
    Sp(std::sync::Arc<sentencepiece::SentencePieceProcessor>),
    Hf(Box<tokenizers::Tokenizer>),
}

impl From<sentencepiece::SentencePieceProcessor> for Tok {
    fn from(sp: sentencepiece::SentencePieceProcessor) -> Self {
        Tok::Sp(std::sync::Arc::new(sp))
    }
}

impl From<tokenizers::Tokenizer> for Tok {
    fn from(tok: tokenizers::Tokenizer) -> Self {
        Tok::Hf(Box::new(tok))
    }
}

impl ptts::Tokenizer for Tok {
    fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
        let tokens = match self {
            Tok::Sp(sp) => {
                sp.encode(text).map_err(xn::Error::wrap)?.into_iter().map(|v| v.id).collect()
            }
            Tok::Hf(tok) => tok.encode(text, false).map_err(xn::Error::wrap)?.get_ids().to_vec(),
        };
        Ok(tokens)
    }

    fn decode(&self, ids: &[u32]) -> xn::Result<String> {
        let decoded = match self {
            Tok::Sp(sp) => sp.decode_piece_ids(ids).map_err(xn::Error::wrap)?,
            Tok::Hf(tok) => tok.decode(ids, true).map_err(xn::Error::wrap)?,
        };
        Ok(decoded)
    }
}

fn remap_key(name: &str) -> Option<String> {
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
    {
        return None;
    }
    let mut name = name.to_string();
    name = name.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    );
    name = name.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    );
    name = name.replace("flow_lm.backbone.", "flow_lm.transformer.");
    name = name.replace("flow_lm.flow.", "flow_lm.flow_net.");
    name = name.replace("mimi.model.", "mimi.");
    Some(name)
}

fn load_voice_embedding<B: xn::Backend>(
    voice_path: &std::path::Path,
    device: &B,
) -> Result<Tensor<f32, B>> {
    let voice_vb = VB::load(&[voice_path], device.clone())?;
    let voice_names = voice_vb.tensor_names();
    let voice_key = voice_names.first().context("no tensors found in voice embedding file")?;
    let voice_shape = voice_vb.shape(voice_key).context("voice tensor not found")?;
    let voice_dims = voice_shape.dims();
    let voice_emb: Tensor<f32, B> = voice_vb.tensor(voice_key, voice_shape.clone())?;
    if voice_dims.len() == 2 {
        Ok(voice_emb.reshape((1, voice_dims[0], voice_dims[1]))?)
    } else {
        Ok(voice_emb)
    }
}

pub struct AppStateB<Q: BackendQ> {
    pub model: Arc<TTSModel<Q>>,
    pub voices: HashMap<String, Tensor<Q::T, Q::B>>,
    pub max_seq_len: usize,
    pub temperature: f32,
    pub seed_base: u64,
    pub sample_rate: u32,
    pub frame_size: u32,
}

#[derive(Clone)]
pub enum AppState {
    Cpu(Arc<AppStateB<xn::Unquantized<f32, xn::CpuDevice>>>),
    Q80(Arc<AppStateB<xn::quantized::Q80F32>>),
    Q81(Arc<AppStateB<xn::quantized::Q81F32>>),
    Q8k(Arc<AppStateB<xn::quantized::Q8kF32>>),
    Q6k(Arc<AppStateB<xn::quantized::Q6kF32>>),
    Q50(Arc<AppStateB<xn::quantized::Q50F32>>),
    Q51(Arc<AppStateB<xn::quantized::Q51F32>>),
    Q5k(Arc<AppStateB<xn::quantized::Q5kF32>>),
    Q40(Arc<AppStateB<xn::quantized::Q40F32>>),
    Q41(Arc<AppStateB<xn::quantized::Q41F32>>),
    Q4k(Arc<AppStateB<xn::quantized::Q4kF32>>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<AppStateB<xn::Unquantized<half::bf16, xn::CudaDevice>>>),
}

struct LoadedModel<Q: BackendQ> {
    cfg: TTSConfig,
    voices: HashMap<String, Tensor<Q::T, Q::B>>,
    tokenizer_path: std::path::PathBuf,
    model_path: std::path::PathBuf,
}

impl<Q: BackendQ> LoadedModel<Q> {
    fn load_from_hf(temperature: f32, dev: &Q::B) -> Result<Self> {
        use hf_hub::{Repo, RepoType, api::sync::Api};
        tracing::info!(repo_id = %REPO_ID, "downloading model artifacts");
        let api = Api::new()?;
        let repo = api.repo(Repo::new(REPO_ID.to_string(), RepoType::Model));
        let model_path = repo.get(MODEL_FILE).map_err(anyhow::Error::from)?;
        tracing::info!(?model_path, "model weights ready");
        let tokenizer_path = repo.get("tokenizer.model").map_err(anyhow::Error::from)?;

        let mut voices: HashMap<String, Tensor<Q::T, Q::B>> = HashMap::new();
        for &voice in VOICES {
            let voice_file = format!("embeddings/{voice}.safetensors");
            match repo.get(&voice_file) {
                Ok(voice_path) => match load_voice_embedding(&voice_path, dev) {
                    Ok(emb) => match emb.to::<Q::T>() {
                        Ok(emb) => {
                            voices.insert(voice.to_string(), emb);
                        }
                        Err(e) => {
                            tracing::warn!(?voice, error = %e, "failed to convert voice embedding")
                        }
                    },
                    Err(e) => tracing::warn!(?voice, error = %e, "failed to load voice embedding"),
                },
                Err(e) => tracing::warn!(?voice, error = %e, "failed to download voice embedding"),
            }
        }
        tracing::info!(num_voices = voices.len(), "voice embeddings loaded");

        let cfg = TTSConfig::v202601(temperature);
        Ok(Self { cfg, voices, tokenizer_path, model_path })
    }

    fn load_from_path(config: &std::path::PathBuf, temperature: f32, dev: &Q::B) -> Result<Self> {
        let parent_dir = config
            .parent()
            .with_context(|| format!("failed to get parent directory of config path {config:?}"))?;
        let mut cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(config)?)
            .with_context(|| "failed to read config from file {config:?}")?;
        cfg.temp = temperature;
        let model_path = if parent_dir.join("model.safetensors").is_file() {
            parent_dir.join("model.safetensors")
        } else if parent_dir.join("model.q8.gguf").is_file() {
            parent_dir.join("model.q8.gguf")
        } else {
            anyhow::bail!(
                "model file not found in directory {parent_dir:?}; expected model.safetensors or model.gguf"
            );
        };
        let tokenizer_path = parent_dir.join("tokenizer.model");
        let mut voices: HashMap<String, Tensor<Q::T, Q::B>> = HashMap::new();
        for voice in parent_dir.join("embeddings").read_dir()? {
            let voice = match voice {
                Ok(v) => v,
                Err(_) => continue,
            };
            let voice = voice.path();
            if voice.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                continue;
            }
            let voice_name =
                voice.file_stem().and_then(|s| s.to_str()).context("invalid voice file name")?;
            match load_voice_embedding(&voice, dev) {
                Ok(emb) => match emb.to::<Q::T>() {
                    Ok(emb) => {
                        voices.insert(voice_name.to_string(), emb);
                    }
                    Err(e) => {
                        tracing::warn!(?voice_name, error = %e, "failed to convert voice embedding")
                    }
                },
                Err(e) => tracing::warn!(?voice_name, error = %e, "failed to load voice embedding"),
            }
        }
        tracing::info!(num_voices = voices.len(), "voice embeddings loaded");
        Ok(Self { cfg, voices, tokenizer_path, model_path })
    }
}

pub fn load_ptts<Q: BackendQ>(
    config: Option<&std::path::PathBuf>,
    temperature: f32,
    seed_base: u64,
    max_seq_len: usize,
    dev: Q::B,
) -> Result<AppStateB<Q>> {
    let m = match config {
        Some(config) => LoadedModel::<Q>::load_from_path(config, temperature, &dev)?,
        None => LoadedModel::<Q>::load_from_hf(temperature, &dev)?,
    };
    let tokenizer_path = m.tokenizer_path.to_str().context("invalid tokenizer path")?;
    let tokenizer = if tokenizer_path.ends_with(".model") {
        tracing::info!("loading SentencePiece tokenizer");
        let sp = sentencepiece::SentencePieceProcessor::open(tokenizer_path)
            .with_context(|| format!("failed to open tokenizer at {tokenizer_path}"))?;
        Tok::Sp(sp.into())
    } else {
        tracing::info!("loading Hugging Face tokenizer");
        let tok = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::format_err!("failed to load tokenizer: {e}"))?;
        Tok::Hf(Box::new(tok))
    };

    let vb = if m.model_path.extension().and_then(|v| v.to_str()) == Some("gguf") {
        let reader = std::fs::File::open(&m.model_path)?;
        let reader = std::io::BufReader::new(reader);
        VB::load_gguf_with_key_map(reader, dev, remap_key)?
    } else {
        VB::load_with_key_map(&[&m.model_path], dev, remap_key)?
    };
    let vb = vb.root();
    let model: TTSModel<Q> = TTSModel::load(&vb, Box::new(tokenizer), &m.cfg)?;
    vb.check_all_used_with_ignore(|v| {
        v == "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding"
            || v.starts_with("mimi.encoder")
            || v.starts_with("mimi.downsample.")
            || v == "flow_lm.speaker_proj_weight"
            || v.starts_with("mimi.quantizer")
    })?;

    let sample_rate = model.sample_rate() as u32;
    let frame_size = (sample_rate as f64 / m.cfg.mimi.frame_rate).round() as u32;

    Ok(AppStateB {
        model: Arc::new(model),
        voices: m.voices,
        max_seq_len,
        temperature,
        seed_base,
        sample_rate,
        frame_size,
    })
}

/// Run a single text-to-audio generation, sending each decoded PCM chunk as it
/// becomes available. Designed to be called inside `tokio::task::spawn_blocking`.
pub fn generate_chunks<Q: BackendQ>(
    model: Arc<TTSModel<Q>>,
    mut state: TTSState<Q>,
    tokens: Vec<u32>,
    temperature: f32,
    seed: u64,
    frames_after_eos: usize,
    audio_tx: tokio::sync::mpsc::UnboundedSender<Vec<f32>>,
) -> Result<(), xn::Error> {
    let device = model.device();
    let num_tokens = tokens.len();
    let max_frames = ((num_tokens as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
    let mut rng = StdRng::new(temperature, seed);
    let mut mimi_state = model.init_mimi_state(1, 250)?;

    model.prompt_text(&mut state, &tokens)?;

    let ldim = model.flow_lm.ldim;
    let nan_data = vec![f32::NAN; ldim];
    let mut prev_latent: Tensor<Q::T, Q::B> =
        Tensor::from_vec(nan_data, (1, 1, ldim), device)?.to::<Q::T>()?;

    let (latent_tx, latent_rx) = std::sync::mpsc::channel::<Tensor<Q::T, Q::B>>();

    let decode_model = Arc::clone(&model);
    let decode_audio_tx = audio_tx.clone();
    let decode_handle = std::thread::spawn(move || -> Result<(), xn::Error> {
        while let Ok(latent) = latent_rx.recv() {
            let audio_chunk = decode_model.decode_latent(&latent, &mut mimi_state)?;
            let pcm = audio_chunk.narrow(0, ..1)?.contiguous()?.to_vec()?;
            if decode_audio_tx.send(pcm).is_err() {
                // Client gone — stop draining.
                break;
            }
        }
        Ok(())
    });

    let mut eos_countdown: Option<usize> = None;
    for _ in 0..max_frames {
        let (next_latent, is_eos) = model.generate_step(&mut state, &prev_latent, &mut rng)?;
        if latent_tx.send(next_latent.clone()).is_err() {
            break;
        }
        if is_eos && eos_countdown.is_none() {
            eos_countdown = Some(frames_after_eos);
        }
        if let Some(ref mut countdown) = eos_countdown {
            if *countdown == 0 {
                break;
            }
            *countdown -= 1;
        }
        prev_latent = next_latent;
    }
    drop(latent_tx);
    decode_handle.join().map_err(|_| xn::Error::msg("decode thread panicked"))??;
    Ok(())
}
