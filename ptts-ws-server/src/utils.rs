use anyhow::{Context as _, Result};
use futures_util::TryStreamExt as _;
use hf_hub::HFClient;
use hf_hub::repository::{HFRepository, RepoTypeModel};
use std::path::PathBuf;

/// Thin wrapper around an `hf_hub` model repository.
///
/// The server runs on tokio, so this uses `hf_hub`'s async client and awaits
/// downloads on the server's own runtime rather than going through the
/// blocking façade and its dedicated runtime thread.
///
/// The point of the wrapper is error context: `hf_hub` names the file and the
/// repo when a file is missing, but an HTTP or authentication failure says
/// neither, which makes server logs hard to act on. Every fallible call here is
/// annotated with the repo id and the filename that was being fetched.
///
/// We only ever talk to model repos, so the repo type is hard-coded.
pub struct HfRepo {
    repo: HFRepository<RepoTypeModel>,
    repo_id: String,
    revision: Option<String>,
}

impl HfRepo {
    /// Open the model repo `repo_id` on the Hub.
    /// The client reads `HF_TOKEN`, `HF_ENDPOINT` and the cache location from
    /// the environment.
    pub fn model(repo_id: &str, revision: Option<&str>) -> Result<Self> {
        let client = HFClient::new().context("failed to initialize the Hugging Face Hub client")?;
        let (owner, name) = hf_hub::split_id(repo_id);
        Ok(Self {
            repo: client.model(owner, name),
            repo_id: repo_id.to_string(),
            revision: revision.map(str::to_owned),
        })
    }

    /// Download `filename` (or fetch it from the local cache), returning its
    /// path on disk. On failure the error names the repo and the file.
    pub async fn get(&self, filename: &str) -> Result<PathBuf> {
        self.repo
            .download_file()
            .filename(filename)
            .maybe_revision(self.revision.clone())
            .send()
            .await
            .with_context(|| {
                format!("failed to fetch `{filename}` from model repo `{}`", self.repo_id)
            })
    }
    /// Voice files listed by this explicitly selected checkpoint revision.
    pub async fn voice_files(&self) -> Result<Vec<(String, String)>> {
        let entries = self
            .repo
            .list_tree()
            .maybe_revision(self.revision.clone())
            .recursive(true)
            .send()
            .with_context(|| format!("cannot list voice files in {}", self.repo_id))?;
        let entries: Vec<_> = entries
            .try_collect()
            .await
            .with_context(|| format!("cannot list voice files in {}", self.repo_id))?;
        let files = entries.into_iter().filter_map(|entry| match entry {
            hf_hub::repository::files::RepoTreeEntry::File { path, .. } => Some(path),
            _ => None,
        });
        Ok(ptts::loader::voice_paths(files))
    }
}
