//! Download transport for the native examples.
//!
//! Local checkpoint resolution and manifest validation live in `ptts::checkpoint`.
//! The library does not download; these examples keep their blocking Hub transport.
#![allow(dead_code, unused_imports)]

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub use ptts::checkpoint::{Checkpoint, POCKET_TTS_REPO as REPO_ID};
use ptts::checkpoint::{
    POCKET_TTS_NO_CLONING_REPO, POCKET_TTS_VOICES, ResolveOptions, TOKENIZER_CANDIDATES,
    read_config, weight_candidates,
};
pub use ptts::loader::{is_unused_by_tts_model, load_voice_emb, load_weights, remap_key};
use ptts::synth::Quant;
#[cfg(feature = "hf")]
pub use ptts::tok::Tok;
use ptts::tts_model::TTSConfig;

/// Default tracing directives. `RUST_LOG` overrides these.
pub const LOG_DIRECTIVES: &str =
    "info,xet=warn,xet_client=warn,xet_data=warn,xet_runtime=warn,xet_core_structures=warn";

/// Where the example obtains its checkpoint files.
#[derive(Clone, Copy, Debug)]
pub enum Source<'a> {
    Dir(&'a Path),
    Hub { repo: &'a str, revision: Option<&'a str> },
}

pub fn locate(source: Source<'_>, weights: Option<&str>, quant: Quant) -> Result<Checkpoint> {
    match source {
        Source::Dir(dir) => Ok(Checkpoint::resolve(dir, ResolveOptions { quant, weights })?),
        Source::Hub { repo, revision } => from_hub(repo, revision, weights, quant),
    }
}

/// Blocking Hub transport for checkpoints with standard artifact names.
pub fn from_hub(
    repo_id: &str,
    revision: Option<&str>,
    weights: Option<&str>,
    quant: Quant,
) -> Result<Checkpoint> {
    let repo = HubRepo::open(repo_id, revision)?;
    tracing::info!(
        repo_id,
        revision = revision.unwrap_or("main"),
        "resolving checkpoint on the Hugging Face Hub"
    );
    let config = match repo.get_optional("config.json") {
        Some(path) => read_config(&path)?,
        None if matches!(repo_id, REPO_ID | POCKET_TTS_NO_CLONING_REPO) => TTSConfig::v202601(),
        None => read_config(repo.get("config.json")?)?,
    };
    let weights = match weights {
        Some(name) => repo.get(name)?,
        None => weight_candidates(quant)
            .iter()
            .find_map(|name| repo.get_optional(name))
            .with_context(|| {
                format!(
                    "no weights in `{repo_id}`; expected one of {}",
                    weight_candidates(quant).join(", ")
                )
            })?,
    };
    let tokenizer = TOKENIZER_CANDIDATES.iter().find_map(|name| repo.get_optional(name));
    let mut voices = vec![];
    for voice in POCKET_TTS_VOICES {
        if let Some(path) = repo.get_optional(&format!("embeddings/{voice}.safetensors")) {
            voices.push((voice.to_string(), path));
        }
    }
    if let Some(path) = repo.get_optional(ptts::loader::DEFAULT_VOICE_FILE) {
        voices.push(("default".to_string(), path));
    }
    voices.sort();
    Ok(Checkpoint { config, weights, tokenizer, voices, quant, manifest: None })
}

/// A Hugging Face model repo, wrapped so a download failure names the repo and the file --
/// `hf_hub` does so for a missing file but not for an HTTP or authentication failure, which
/// makes a gated repo hard to diagnose -- and so the callers need not spell out the download
/// builder.
pub struct HubRepo {
    repo: hf_hub::HFRepositorySync<hf_hub::repository::RepoTypeModel>,
    repo_id: String,
    revision: Option<String>,
}

impl HubRepo {
    /// The client reads `HF_TOKEN`, `HF_ENDPOINT` and the cache location from the environment,
    /// falling back to the token `huggingface-cli login` stores.
    pub fn open(repo_id: &str, revision: Option<&str>) -> Result<Self> {
        let client = hf_hub::HFClientSync::new().context("cannot reach the Hugging Face Hub")?;
        let (owner, name) = hf_hub::split_id(repo_id);
        Ok(Self {
            repo: client.model(owner, name),
            repo_id: repo_id.to_string(),
            revision: revision.map(str::to_owned),
        })
    }

    /// Download `filename`, or find it in the local cache.
    pub fn get(&self, filename: &str) -> Result<PathBuf> {
        self.repo
            .download_file()
            .filename(filename)
            .maybe_revision(self.revision.clone())
            .send()
            .map_err(|e| {
                anyhow::anyhow!(
                    "failed to fetch `{filename}` from `{}`: {e}\n\
                 If the repo is gated, accept its terms on huggingface.co and run \
                 `huggingface-cli login` (or set HF_TOKEN).",
                    self.repo_id
                )
            })
    }

    /// Like [`Self::get`] but maps any failure to `None`, for files that may legitimately be
    /// absent from a given repo layout.
    fn get_optional(&self, filename: &str) -> Option<PathBuf> {
        self.repo
            .download_file()
            .filename(filename)
            .maybe_revision(self.revision.clone())
            .send()
            .ok()
    }
}
