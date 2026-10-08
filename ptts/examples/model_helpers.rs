//! Download transport for the native examples.
//!
//! Local checkpoint resolution lives in `ptts::checkpoint`.
//! The library does not download; these examples keep their blocking Hub transport.
#![allow(dead_code, unused_imports)]

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub use ptts::checkpoint::Checkpoint;
use ptts::checkpoint::{ResolveOptions, read_config, weight_candidates};
pub use ptts::loader::{is_unused_by_tts_model, load_voice_emb, load_weights, remap_key};
use ptts::synth::Quant;
#[cfg(feature = "hf")]
pub use ptts::tok::Tok;

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
    let config = read_config(repo.get("config.json")?)?;
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
    let tokenizer = Some(repo.get("tokenizer.json")?);
    let mut voices = vec![];
    let voice_files = repo.voice_files().unwrap_or_else(|e| {
        tracing::warn!(error = %e, "cannot list voice files; only the default voice file is tried");
        Vec::new()
    });
    for (name, file) in voice_files {
        if let Some(path) = repo.get_optional(&file) {
            voices.push((name, path));
        }
    }
    if !voices.iter().any(|(name, _)| name == "default")
        && let Some(path) = repo.get_optional(ptts::loader::DEFAULT_VOICE_FILE)
    {
        voices.push(("default".to_string(), path));
    }
    voices.sort();
    Ok(Checkpoint { config, weights, tokenizer, voices, quant })
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

    fn voice_files(&self) -> Result<Vec<(String, String)>> {
        let entries = self
            .repo
            .list_tree()
            .maybe_revision(self.revision.clone())
            .recursive(true)
            .send()
            .with_context(|| format!("cannot list voice files in {}", self.repo_id))?;
        let files = entries.into_iter().filter_map(|entry| match entry {
            hf_hub::repository::files::RepoTreeEntry::File { path, .. } => Some(path),
            _ => None,
        });
        Ok(ptts::loader::voice_paths(files))
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
