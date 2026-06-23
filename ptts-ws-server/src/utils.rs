use anyhow::{Context as _, Result};
use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};
use std::path::PathBuf;

/// Thin wrapper around an `hf_hub` model repository.
///
/// The point of the wrapper is error context: `hf_hub`'s own download errors
/// don't mention which file (or URL) failed, which makes server logs hard to
/// act on. Every fallible call here is annotated with the repo id, the
/// filename, and the fully-qualified URL that was being fetched.
///
/// We only ever talk to model repos, so the repo type is hard-coded.
pub struct HfRepo {
    repo: ApiRepo,
    repo_id: String,
}

impl HfRepo {
    /// Open the model repo `repo_id` (e.g. `"kyutai/pocket-tts"`) on the Hub.
    pub fn model(repo_id: &str) -> Result<Self> {
        let api = Api::new().context("failed to initialize the Hugging Face Hub API")?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
        Ok(Self { repo, repo_id: repo_id.to_string() })
    }

    /// Download `filename` (or fetch it from the local cache), returning its
    /// path on disk. On failure the error names the repo, file, and URL.
    pub fn get(&self, filename: &str) -> Result<PathBuf> {
        self.repo.get(filename).with_context(|| {
            format!(
                "failed to fetch `{filename}` from model repo `{}` ({})",
                self.repo_id,
                self.repo.url(filename),
            )
        })
    }
}
