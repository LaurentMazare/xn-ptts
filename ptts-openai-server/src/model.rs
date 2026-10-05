//! Locating a checkpoint on disk or on the Hub, and loading it into a [`Synth`].
//!
//! Everything about *how* speech is generated lives in `ptts::synth`. What is
//! left here is deciding which files to load, which voices to register, and
//! holding the result for the request handlers.
//!
//! A trimmed copy of `ptts-ws-server`'s `model.rs`, kept separate on purpose: a fix to one
//! likely belongs in the other.

use anyhow::{Context as _, Result};
use ptts::preprocess::Normalize;
use ptts::synth::{DeviceKind, Quant, Synth, SynthBuilder};
use ptts::tts_model::TTSConfig;
use std::sync::Arc;

pub const VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];

/// Kyutai's checkpoint without the voice-cloning weights, which this server does not use. Unlike
/// `kyutai/pocket-tts` it is not gated, so it downloads without a Hugging Face token.
pub const DEFAULT_REPO_ID: &str = "kyutai/pocket-tts-without-voice-cloning";
pub const DEFAULT_MODEL_FILE: &str = "tts_b6369a24.safetensors";

/// Weight file names tried in a local folder, in order.
const WEIGHT_CANDIDATES: [&str; 3] = ["model.safetensors", "model.q8.gguf", DEFAULT_MODEL_FILE];

/// The loaded model and the request defaults, shared by every request.
#[derive(Clone)]
pub struct AppState(Arc<Inner>);

pub struct Inner {
    pub synth: Synth,
    /// The checkpoint that loaded: its repo id, or for a local config the name of its folder.
    pub model_name: String,
    pub voices: Vec<String>,
    pub seed_base: u64,
    pub sample_rate: u32,
    pub frame_size: u32,
}

impl std::ops::Deref for AppState {
    type Target = Inner;

    fn deref(&self) -> &Inner {
        &self.0
    }
}

/// A checkpoint whose files are located. The voices are loaded once the model is, since a
/// file of stored speaker latents goes through the checkpoint's speaker projection.
struct LoadedModel {
    cfg: TTSConfig,
    /// Voice name to embedding file.
    voice_files: Vec<(String, std::path::PathBuf)>,
    tokenizer_path: std::path::PathBuf,
    model_path: std::path::PathBuf,
}

impl LoadedModel {
    async fn load_from_hf(repo_id: &str) -> Result<Self> {
        tracing::info!("downloading model artifacts");
        let repo = crate::utils::HfRepo::model(repo_id)?;
        let config_path = repo.get("config.json").await?;
        let cfg: TTSConfig = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
            .with_context(|| format!("failed to read config from file {config_path:?}"))?;

        let model_path = repo.get("model.q8.gguf").await?;
        tracing::info!(?model_path, "model weights ready");
        let tokenizer_path = repo.get("tokenizer.json").await?;

        let default_voice_path = repo.get("default-voice.safetensors").await?;
        let voice_files = vec![("default".to_string(), default_voice_path)];

        Ok(Self { cfg, voice_files, tokenizer_path, model_path })
    }

    async fn load_pocket_from_hf() -> Result<Self> {
        tracing::info!("downloading model artifacts");
        let repo = crate::utils::HfRepo::model(DEFAULT_REPO_ID)?;
        let model_path = repo.get(DEFAULT_MODEL_FILE).await?;
        tracing::info!(?model_path, "model weights ready");
        let tokenizer_path = repo.get("tokenizer.json").await?;

        let mut voice_files = Vec::new();
        for &voice in VOICES {
            let voice_file = format!("embeddings/{voice}.safetensors");
            match repo.get(&voice_file).await {
                Ok(voice_path) => voice_files.push((voice.to_string(), voice_path)),
                Err(e) => tracing::warn!(?voice, error = %e, "failed to download voice embedding"),
            }
        }

        let cfg = TTSConfig::v202601();
        Ok(Self { cfg, voice_files, tokenizer_path, model_path })
    }

    /// A local checkpoint folder. Its `config.json` is optional, as Kyutai's checkpoint has none.
    fn load_from_dir(dir: &std::path::Path) -> Result<Self> {
        let config = dir.join("config.json");
        let cfg = if config.is_file() {
            serde_json::from_str(&std::fs::read_to_string(&config)?)
                .with_context(|| format!("failed to read config from file {config:?}"))?
        } else {
            TTSConfig::v202601()
        };
        let model_path = WEIGHT_CANDIDATES
            .iter()
            .map(|name| dir.join(name))
            .find(|p| p.is_file())
            .with_context(|| {
                format!("no weights in {dir:?}; expected one of {}", WEIGHT_CANDIDATES.join(", "))
            })?;
        let tokenizer_path = dir.join("tokenizer.json");
        let voice_files = ptts::loader::checkpoint_voices(dir);
        Ok(Self { cfg, voice_files, tokenizer_path, model_path })
    }
}

/// Load the model named by `config`: a local checkpoint folder or a `config.json` in one, a Hub
/// repo id, or nothing for Kyutai's checkpoint.
pub async fn load_ptts(
    config: Option<&std::path::PathBuf>,
    voice_dir: Option<&std::path::PathBuf>,
    device: DeviceKind,
    quant: Quant,
    temperature: f32,
    seed_base: u64,
    normalize: Normalize,
) -> Result<AppState> {
    if let Some(config) = config
        && config.extension().is_some_and(|v| v == "json")
        && !config.is_file()
    {
        anyhow::bail!("no config file at {config:?}");
    }
    // The folder of a local checkpoint, made absolute so that `.` still has a name.
    let local_dir = match config {
        Some(c) if c.is_dir() => Some(std::fs::canonicalize(c)?),
        Some(c) if c.is_file() => std::fs::canonicalize(c)?.parent().map(|d| d.to_path_buf()),
        _ => None,
    };
    let mut m = match (config, &local_dir) {
        (_, Some(dir)) => LoadedModel::load_from_dir(dir)?,
        (Some(repo_id), None) => {
            let repo_id = repo_id.to_str().context("invalid repo ID path")?;
            LoadedModel::load_from_hf(repo_id).await?
        }
        (None, None) => LoadedModel::load_pocket_from_hf().await?,
    };
    if let Some(voice_dir) = voice_dir {
        let found = ptts::loader::voices_in(voice_dir);
        if found.is_empty() {
            tracing::warn!(?voice_dir, "no voice files found in --voice-dir");
        }
        m.voice_files.extend(found);
    }
    let frame_rate = m.cfg.mimi.frame_rate;
    let mut synth = SynthBuilder::new(m.cfg, &m.model_path, normalize)
        .tokenizer_file(&m.tokenizer_path)
        .device(device)
        .quant(quant)
        .temperature(temperature)
        .build()?;
    // Registered after the build, not through it: the builder propagates a bad
    // voice file and one should not take the server down. Order is preserved,
    // so a --voice-dir entry still overrides a bundled voice of the same name.
    for (name, path) in m.voice_files.iter() {
        if let Err(e) = synth.add_voice_file(name, path) {
            tracing::warn!(voice = %name, error = %e, "failed to load voice embedding");
        }
    }

    let voices = synth.voices();
    let default_voice = synth.default_voice().context("no voice embeddings found in model")?;
    let sample_rate = synth.sample_rate();
    let frame_size = (sample_rate as f64 / frame_rate).round() as u32;
    tracing::info!(
        device = %synth.device_name(),
        weights = %synth.quant().as_str(),
        num_voices = voices.len(),
        %default_voice,
        lang = normalize.as_str(),
        "model loaded"
    );

    // A repo id as given. For a local checkpoint, only its folder's name: clients have no use
    // for the server's filesystem layout.
    let model_name = match (config, &local_dir) {
        (_, Some(dir)) => dir.file_name().unwrap_or_default().to_string_lossy().into_owned(),
        (Some(repo_id), None) => repo_id.display().to_string(),
        (None, None) => DEFAULT_REPO_ID.to_string(),
    };
    Ok(AppState(Arc::new(Inner { synth, model_name, voices, seed_base, sample_rate, frame_size })))
}
