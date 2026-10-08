//! Shared native checkpoint file resolution.
//!
//! [`Checkpoint::open`] accepts an explicitly supplied model directory or config file.
//! Each checkpoint supplies its own config, tokenizer, weights, and optional voices.
//! This module reads local files only. Frontends retain their own Hub transport.

use crate::preprocess::Normalize;
use crate::synth::{Quant, Synth, SynthBuilder};
use crate::tts_model::TTSConfig;
use crate::{Error, Result};
use std::path::{Path, PathBuf};

/// Standard weight filenames, ordered by the requested quantization.
pub fn weight_candidates(quant: Quant) -> [&'static str; 2] {
    if quant == Quant::Q80 {
        ["model.q8.gguf", "model.safetensors"]
    } else {
        ["model.safetensors", "model.q8.gguf"]
    }
}

/// Explicit native resolution choices.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResolveOptions<'a> {
    pub quant: Quant,
    /// A weights path relative to the supplied model directory.
    pub weights: Option<&'a str>,
}

/// Checkpoint files ready to hand to a native frontend or [`SynthBuilder`].
#[derive(Debug)]
pub struct Checkpoint {
    pub config: TTSConfig,
    pub weights: PathBuf,
    /// The checkpoint's tokenizer, unless the caller supplies one separately.
    pub tokenizer: Option<PathBuf>,
    /// External voices, sorted by name. Baked voices come from the config and weights.
    pub voices: Vec<(String, PathBuf)>,
    pub quant: Quant,
}

impl Checkpoint {
    /// Resolve a supplied model directory or config file with default options.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::resolve(path, ResolveOptions::default())
    }

    pub fn resolve(path: impl AsRef<Path>, options: ResolveOptions<'_>) -> Result<Self> {
        let input = path.as_ref();
        let path = std::fs::canonicalize(input).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                Error::NotFound(format!("no checkpoint at {}", input.display()))
            }
            _ => Error::Io(e),
        })?;
        let (dir, config_path) = if path.is_dir() {
            (path.clone(), path.join("config.json"))
        } else {
            // HF cache files can link to blobs outside the snapshot. Their neighbors
            // belong to the supplied path's parent rather than the blob directory.
            let parent =
                input.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
            (std::fs::canonicalize(parent)?, path)
        };
        let config = read_config(config_path)?;
        let weights = match options.weights {
            Some(name) => {
                validate_relative_path(name)?;
                required_file(&dir.join(name))?
            }
            None => weight_candidates(options.quant)
                .iter()
                .map(|name| dir.join(name))
                .find(|p| p.is_file())
                .ok_or_else(|| {
                    Error::NotFound(format!(
                        "no weights in {}; expected one of {} or an explicit weights path",
                        dir.display(),
                        weight_candidates(options.quant).join(", ")
                    ))
                })?,
        };
        let tokenizer = dir.join("tokenizer.json");
        let tokenizer = tokenizer.is_file().then_some(tokenizer);
        let mut voices = crate::loader::checkpoint_voices(&dir);
        voices.sort();
        Ok(Self { config, weights, tokenizer, voices, quant: options.quant })
    }

    pub fn builder(&self, normalize: impl Into<Normalize>) -> SynthBuilder {
        let mut builder =
            SynthBuilder::new(self.config.clone(), &self.weights, normalize).quant(self.quant);
        if let Some(path) = &self.tokenizer {
            builder = builder.tokenizer_file(path);
        }
        builder
    }

    /// Register checkpoint voices, warning if an optional voice is unreadable.
    pub fn register_voices(&self, synth: &mut Synth) {
        for (name, path) in &self.voices {
            if let Err(e) = synth.add_voice_file(name, path) {
                tracing::warn!(voice = %name, error = %e, "skipping voice embedding");
            }
        }
    }
}

/// Classify an explicitly supplied source before choosing local or Hub resolution.
/// Nonexistent rooted, dot-prefixed, or JSON paths are local errors, not repo IDs.
pub fn is_local_source(path: &Path) -> bool {
    path.exists()
        || path.has_root()
        || path.starts_with(".")
        || path.components().next().is_some_and(|c| c.as_os_str() == "..")
        || path.extension().is_some_and(|ext| ext == "json")
}

/// Read the checkpoint's config with consistent error classes and context.
pub fn read_config(path: impl AsRef<Path>) -> Result<TTSConfig> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => {
            Error::NotFound(format!("cannot read config {}: {e}", path.display()))
        }
        _ => Error::Io(e),
    })?;
    serde_json::from_str(&text)
        .map_err(|e| Error::InvalidData(format!("cannot parse config {}: {e}", path.display())))
}

fn validate_relative_path(path: &str) -> Result<()> {
    if path.is_empty()
        || path.contains(['\\', ':'])
        || path.starts_with('/')
        || path.split('/').any(|p| p.is_empty() || p == "." || p == "..")
    {
        return Err(Error::InvalidData(format!(
            "weights path '{path}' must be relative to the checkpoint root"
        )));
    }
    Ok(())
}

fn required_file(path: &Path) -> Result<PathBuf> {
    if !path.is_file() {
        return Err(Error::NotFound(format!("checkpoint file is missing: {}", path.display())));
    }
    Ok(path.to_path_buf())
}
