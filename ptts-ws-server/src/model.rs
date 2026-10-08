//! Locating a checkpoint on disk or on the Hub, and loading it into a [`Synth`].
//!
//! Everything about *how* speech is generated lives in `ptts::synth`. What is
//! left here is deciding which files to load, which voices to register, and
//! holding the result for the request handlers.

use anyhow::{Context as _, Result};
use ptts::checkpoint::{
    Checkpoint, POCKET_TTS_VOICES, ResolveOptions, is_local_source, read_config, weight_candidates,
};
use ptts::preprocess::Normalize;
use ptts::synth::{DeviceKind, Quant, Synth};
use ptts::tts_model::TTSConfig;
use std::sync::Arc;

pub const DEFAULT_REPO_ID: &str = ptts::checkpoint::POCKET_TTS_REPO;

/// The loaded model and the request defaults, shared by every connection.
///
/// `Synth` erases the weight format, so this is one struct rather than the
/// fourteen-variant enum the handlers used to match on.
#[derive(Clone)]
pub struct AppState(Arc<Inner>);

pub struct Inner {
    pub synth: Synth,
    /// The checkpoint that loaded: its repo id, or for a local config the name of its folder.
    pub model_name: String,
    pub voices: Vec<String>,
    pub default_voice: String,
    pub max_seq_len: usize,
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

/// Existing async Hub transport. Local candidates use the shared checkpoint resolver.
/// Manifest-aware Hub acquisition is the next release work item.
async fn load_from_hf(
    repo_id: &str,
    revision: Option<&str>,
    quant: Quant,
    pocket: bool,
) -> Result<Checkpoint> {
    tracing::info!(repo_id, revision = revision.unwrap_or("main"), "downloading model artifacts");
    let repo = crate::utils::HfRepo::model(repo_id, revision)?;
    let config =
        if pocket { TTSConfig::v202601() } else { read_config(repo.get("config.json").await?)? };
    let mut weights = None;
    let mut first_error = None;
    for name in weight_candidates(quant) {
        match repo.get(name).await {
            Ok(path) => {
                weights = Some(path);
                break;
            }
            Err(e) => {
                first_error.get_or_insert(e);
            }
        }
    }
    let weights = weights
        .ok_or_else(|| first_error.unwrap_or_else(|| anyhow::anyhow!("no weights in {repo_id}")))?;
    let tokenizer = Some(repo.get("tokenizer.json").await?);
    let mut voices = Vec::new();
    if pocket {
        for name in POCKET_TTS_VOICES {
            let file = format!("embeddings/{name}.safetensors");
            match repo.get(&file).await {
                Ok(path) => voices.push((name.to_string(), path)),
                Err(e) => {
                    tracing::warn!(voice = %name, error = %e, "failed to download voice embedding")
                }
            }
        }
    }
    // A checkpoint with baked-in voices need not ship an external default.
    if let Ok(path) = repo.get(ptts::loader::DEFAULT_VOICE_FILE).await {
        voices.push(("default".to_string(), path));
    }
    voices.sort();
    Ok(Checkpoint { config, weights, tokenizer, voices, quant, manifest: None })
}

/// Load the model named by `config` -- a local `config.json`, a Hub repo id, or
/// nothing for the published checkpoint.
#[allow(clippy::too_many_arguments)]
pub async fn load_ptts(
    config: Option<&std::path::PathBuf>,
    revision: Option<&str>,
    voice_dir: Option<&std::path::PathBuf>,
    device: DeviceKind,
    quant: Quant,
    temperature: f32,
    seed_base: u64,
    max_seq_len: usize,
    normalize: Normalize,
    conditions: &[(String, String)],
) -> Result<AppState> {
    quant.check_device(device)?;
    let (m, model_name) = match config {
        Some(path) if is_local_source(path) => {
            anyhow::ensure!(revision.is_none(), "--revision requires a Hugging Face repo ID");
            let checkpoint = Checkpoint::resolve(path, ResolveOptions { quant, weights: None })?;
            let absolute = std::fs::canonicalize(path)?;
            let dir = if absolute.is_dir() {
                absolute.as_path()
            } else {
                absolute.parent().context("config has a parent directory")?
            };
            let name =
                checkpoint.manifest.as_ref().map(|m| m.model_id.clone()).unwrap_or_else(|| {
                    dir.file_name().unwrap_or_default().to_string_lossy().into_owned()
                });
            (checkpoint, name)
        }
        Some(repo_id) => {
            let repo_id = repo_id.to_str().context("invalid repo ID path")?;
            (
                load_from_hf(repo_id, revision, quant, repo_id == DEFAULT_REPO_ID).await?,
                repo_id.to_string(),
            )
        }
        None => (
            load_from_hf(DEFAULT_REPO_ID, revision, quant, true).await?,
            DEFAULT_REPO_ID.to_string(),
        ),
    };
    let extra_voices = if let Some(voice_dir) = voice_dir {
        let found = ptts::loader::voices_in(voice_dir);
        if found.is_empty() {
            tracing::warn!(?voice_dir, "no voice files found in --voice-dir");
        }
        found
    } else {
        Vec::new()
    };
    let frame_rate = m.config.mimi.frame_rate;
    let mut builder = m.builder(normalize).device(device).quant(quant).temperature(temperature);
    for (name, value) in conditions {
        builder = builder.condition(name, value);
    }
    let mut synth = builder.build()?;
    m.register_voices(&mut synth);
    // Additional user voices remain optional and override names from the checkpoint.
    // Only manifest-declared voices are required during the builder's validation.
    for (name, path) in extra_voices {
        if let Err(e) = synth.add_voice_file(&name, &path) {
            tracing::warn!(voice = %name, error = %e, "failed to load additional voice embedding");
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

    Ok(AppState(Arc::new(Inner {
        synth,
        model_name,
        voices,
        default_voice,
        max_seq_len,
        seed_base,
        sample_rate,
        frame_size,
    })))
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires PTTS_TEST_MODEL pointing to a private q8 checkpoint"]
    async fn private_checkpoint_loads_with_the_shared_default_voice() {
        let path = std::path::PathBuf::from(
            std::env::var("PTTS_TEST_MODEL").expect("set PTTS_TEST_MODEL"),
        );
        let checkpoint =
            Checkpoint::resolve(&path, ResolveOptions { quant: Quant::Q80, weights: None })
                .unwrap();
        let suffix =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let extra_dir =
            std::env::temp_dir().join(format!("ptts-extra-voices-{}-{suffix}", std::process::id()));
        std::fs::create_dir(&extra_dir).unwrap();
        std::fs::write(extra_dir.join("invalid.safetensors"), b"unreadable optional voice")
            .unwrap();
        let loaded = load_ptts(
            Some(&path),
            None,
            Some(&extra_dir),
            DeviceKind::Cpu,
            Quant::Q80,
            0.3,
            7,
            1024,
            Normalize::OFF,
            &[],
        )
        .await;
        std::fs::remove_dir_all(extra_dir).unwrap();
        let state = loaded.unwrap();
        assert!(!state.voices.iter().any(|v| v == "invalid"));
        assert_eq!(state.sample_rate, checkpoint.config.mimi.sample_rate as u32);
        if let Some(manifest) = &checkpoint.manifest {
            assert_eq!(state.model_name, manifest.model_id);
            assert_eq!(state.synth.default_voice(), manifest.default_voice);
        }
        let audio = state.synth.say("Hello from Phonon.").unwrap();
        assert!(audio.len() > state.sample_rate as usize / 4);
        assert!(audio.iter().all(|x| x.is_finite()));
        assert!(audio.iter().any(|x| x.abs() > 0.01));
    }
}
