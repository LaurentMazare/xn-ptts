//! Checkpoint manifests and shared native file resolution.
//!
//! [`Checkpoint::open`] accepts a directory, a config file, or a `ptts-model.json` manifest.
//! This module reads local files only. Frontends retain their own Hub/download transport.
//! A manifest selects exact artifacts; directories without one use standard Phonon artifact
//! names with legacy Pocket TTS compatibility. [`ModelManifest`] describes the optional metadata.

use crate::preprocess::Normalize;
use crate::synth::{Quant, Synth, SynthBuilder};
use crate::tts_model::TTSConfig;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Filename used to discover a checkpoint manifest inside a directory.
pub const MANIFEST_FILE: &str = "ptts-model.json";
/// The legacy Pocket TTS repo used by native examples and Python until Phonon's release.
pub const POCKET_TTS_REPO: &str = "kyutai/pocket-tts";
/// Pocket TTS without its voice-cloning encoder, used by the HTTP server.
pub const POCKET_TTS_NO_CLONING_REPO: &str = "kyutai/pocket-tts-without-voice-cloning";
/// The original Pocket TTS f32 weights filename.
pub const POCKET_TTS_WEIGHTS: &str = "tts_b6369a24.safetensors";
/// Voice names used when a legacy Hub repo cannot be listed.
pub const POCKET_TTS_VOICES: &[&str] =
    &["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];
/// Legacy tokenizer names. A `.model` file gives the tokenizer's conversion instructions.
pub const TOKENIZER_CANDIDATES: &[&str] = &["tokenizer.json", "tokenizer.model"];

/// Legacy weights preference. Exact q8 precedes f32 only when q8 was requested.
pub fn weight_candidates(quant: Quant) -> [&'static str; 3] {
    if quant == Quant::Q80 {
        ["model.q8.gguf", "model.safetensors", POCKET_TTS_WEIGHTS]
    } else {
        ["model.safetensors", POCKET_TTS_WEIGHTS, "model.q8.gguf"]
    }
}

/// A file relative to the checkpoint root, with an optional development-time checksum.
/// Release manifests should include a SHA-256 for every artifact.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Artifact {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

/// The eventual Hub source. Local resolution does not contact or authenticate with it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubSource {
    pub repo: String,
    pub revision: String,
}

/// Claims about this particular checkpoint, independent of normalization languages.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Capabilities {
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_cloning: Option<bool>,
}

/// Version 1 of the native checkpoint contract.
///
/// Weight keys use [`Quant::as_str`], for example `f32` or `q8_0`. The resolver selects the
/// requested format, or f32 for runtime quantization. It does not silently expand a manifest's
/// quantized weights to f32 or requantize them to another format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub model_id: String,
    /// A checkpoint identifier, also for candidates that have never been uploaded.
    pub revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<HubSource>,
    /// Minimum compatible `ptts` crate version. Omit for unversioned development candidates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_min_version: Option<String>,
    pub config: Artifact,
    pub tokenizer: Artifact,
    pub weights: BTreeMap<String, Artifact>,
    #[serde(default)]
    pub voices: BTreeMap<String, Artifact>,
    /// May name an external voice above or a baked voice in `config.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_voice: Option<String>,
    pub sample_rate: u32,
    #[serde(default)]
    pub capabilities: Capabilities,
}

impl ModelManifest {
    /// Read and validate metadata. Artifact existence and hashes are checked on resolution.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let manifest: Self = read_json(path, "manifest")?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(Error::Unsupported(format!(
                "checkpoint manifest schema {} is unsupported; expected 1",
                self.schema_version
            )));
        }
        if self.model_id.trim().is_empty() || self.revision.trim().is_empty() {
            return Err(Error::InvalidData(
                "manifest model_id and revision must be nonempty".into(),
            ));
        }
        if self.sample_rate == 0 || self.weights.is_empty() {
            return Err(Error::InvalidData(
                "manifest needs a sample rate and weight artifacts".into(),
            ));
        }
        if let Some(minimum) = &self.runtime_min_version {
            let minimum = semver::Version::parse(minimum).map_err(|e| {
                Error::InvalidData(format!("invalid manifest runtime_min_version: {e}"))
            })?;
            let current = semver::Version::parse(env!("CARGO_PKG_VERSION"))
                .expect("Cargo package versions are valid semver");
            if current < minimum {
                return Err(Error::Unsupported(format!(
                    "checkpoint requires ptts >= {minimum}; this runtime is {current}"
                )));
            }
        }
        if let Some(source) = &self.source {
            let parts: Vec<_> = source.repo.split('/').collect();
            if parts.len() != 2
                || parts.iter().any(|p| p.is_empty() || *p == "." || *p == "..")
                || source.revision.len() != 40
                || !source.revision.bytes().all(|c| c.is_ascii_hexdigit())
            {
                return Err(Error::InvalidData(
                    "manifest source needs an owner/repo and a pinned 40-character commit revision"
                        .into(),
                ));
            }
        }
        for format in self.weights.keys() {
            let quant = format.parse::<Quant>().map_err(|_| {
                Error::InvalidData(format!("unknown manifest weight format '{format}'"))
            })?;
            if format != quant.as_str() {
                return Err(Error::InvalidData(format!(
                    "manifest weight format '{format}' must use '{}'",
                    quant.as_str()
                )));
            }
        }
        for name in self.voices.keys().chain(self.default_voice.iter()) {
            validate_voice_name(name)?;
        }
        for artifact in std::iter::once(&self.config)
            .chain(std::iter::once(&self.tokenizer))
            .chain(self.weights.values())
            .chain(self.voices.values())
        {
            validate_artifact(artifact)?;
        }
        Ok(())
    }
}

/// Explicit native resolution choices. Quantization also controls legacy file preference.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResolveOptions<'a> {
    pub quant: Quant,
    /// A relative weights path. A manifested checkpoint requires it to be declared.
    pub weights: Option<&'a str>,
}

/// Files selected and validated, ready to hand to a native frontend or [`SynthBuilder`].
#[derive(Debug)]
pub struct Checkpoint {
    pub config: TTSConfig,
    pub weights: PathBuf,
    /// Missing only for a legacy directory whose caller supplies a tokenizer separately.
    pub tokenizer: Option<PathBuf>,
    /// External voices, sorted by name. Baked voices are loaded from the config and weights.
    pub voices: Vec<(String, PathBuf)>,
    pub quant: Quant,
    pub manifest: Option<ModelManifest>,
}

impl Checkpoint {
    /// Resolve a local directory, config file, or `ptts-model.json` using default options.
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
        let (dir, config_override) = if path.is_dir() {
            (path.as_path(), None)
        } else {
            let dir =
                path.parent().ok_or_else(|| Error::NotFound("checkpoint has no parent".into()))?;
            let config =
                (path.file_name().is_none_or(|n| n != MANIFEST_FILE)).then_some(path.as_path());
            (dir, config)
        };
        let manifest_path = dir.join(MANIFEST_FILE);
        if manifest_path.try_exists()? {
            Self::from_manifest(dir, &manifest_path, config_override, options)
        } else {
            Self::from_legacy(dir, config_override, options)
        }
    }

    fn from_manifest(
        dir: &Path,
        path: &Path,
        config_override: Option<&Path>,
        options: ResolveOptions<'_>,
    ) -> Result<Self> {
        let manifest = ModelManifest::read(path)?;
        let config_path = resolve_artifact(dir, &manifest.config)?;
        if let Some(config_override) = config_override
            && std::fs::canonicalize(&config_path)? != config_override
        {
            return Err(Error::InvalidArgument(
                "config path differs from the checkpoint manifest".into(),
            ));
        }
        let config = read_config(&config_path)?;
        for voice in &config.voices {
            validate_voice_name(&voice.name)?;
        }
        if config.mimi.sample_rate != manifest.sample_rate as usize {
            return Err(Error::InvalidData(format!(
                "manifest sample rate {} differs from config {}",
                manifest.sample_rate, config.mimi.sample_rate
            )));
        }
        let weight = match options.weights {
            Some(name) => manifest.weights.iter().find(|(_, a)| a.path == name),
            None => manifest
                .weights
                .get_key_value(options.quant.as_str())
                .or_else(|| manifest.weights.get_key_value("f32")),
        }
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "checkpoint '{}' has no weights for {}{}",
                manifest.model_id,
                options.quant.as_str(),
                options.weights.map(|n| format!(" at '{n}'")).unwrap_or_default()
            ))
        })?;
        if weight.0 != options.quant.as_str() && weight.0 != "f32" {
            return Err(Error::Unsupported(format!(
                "manifest weights are {}, but {} was requested",
                weight.0,
                options.quant.as_str()
            )));
        }
        let weights = resolve_artifact(dir, weight.1)?;
        let tokenizer = Some(resolve_artifact(dir, &manifest.tokenizer)?);
        let voices = manifest
            .voices
            .iter()
            .map(|(name, file)| Ok((name.clone(), resolve_artifact(dir, file)?)))
            .collect::<Result<Vec<_>>>()?;
        if let Some(default) = &manifest.default_voice
            && !manifest.voices.contains_key(default)
            && !config.voices.iter().any(|v| &v.name == default)
        {
            return Err(Error::InvalidData(format!(
                "manifest default voice '{default}' is not declared"
            )));
        }
        Ok(Self {
            config,
            weights,
            tokenizer,
            voices,
            quant: options.quant,
            manifest: Some(manifest),
        })
    }

    fn from_legacy(
        dir: &Path,
        config_override: Option<&Path>,
        options: ResolveOptions<'_>,
    ) -> Result<Self> {
        let config_path =
            config_override.map(Path::to_path_buf).unwrap_or_else(|| dir.join("config.json"));
        let config = if config_path.try_exists()? {
            read_config(&config_path)?
        } else {
            tracing::info!(?dir, "no config.json, using the legacy Pocket TTS config");
            TTSConfig::v202601()
        };
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
                        "no weights in {}; expected one of {}",
                        dir.display(),
                        weight_candidates(options.quant).join(", ")
                    ))
                })?,
        };
        let tokenizer =
            TOKENIZER_CANDIDATES.iter().map(|name| dir.join(name)).find(|p| p.is_file());
        let mut voices = crate::loader::checkpoint_voices(dir);
        voices.sort();
        Ok(Self { config, weights, tokenizer, voices, quant: options.quant, manifest: None })
    }

    /// Build using the resolved files. Manifest voices are required and registered here;
    /// legacy voices retain the tolerant [`Self::register_voices`] path.
    pub fn builder(&self, normalize: impl Into<Normalize>) -> SynthBuilder {
        let mut builder =
            SynthBuilder::new(self.config.clone(), &self.weights, normalize).quant(self.quant);
        if let Some(path) = &self.tokenizer {
            builder = builder.tokenizer_file(path);
        }
        if let Some(manifest) = &self.manifest {
            for (name, path) in &self.voices {
                builder = builder.add_voice(name, path);
            }
            if let Some(default) = &manifest.default_voice {
                builder = builder.voice(default);
            }
        }
        builder
    }

    /// Register legacy external voices, warning if an optional voice is unreadable.
    /// Manifest-declared voices are already registered by [`Self::builder`].
    pub fn register_voices(&self, synth: &mut Synth) {
        if self.manifest.is_none() {
            for (name, path) in &self.voices {
                if let Err(e) = synth.add_voice_file(name, path) {
                    tracing::warn!(voice = %name, error = %e, "skipping voice embedding");
                }
            }
        }
    }
}

/// Classify a user-supplied source before choosing local resolution or a Hub transport.
/// Nonexistent absolute, dot-prefixed, or JSON paths are local errors, not repo IDs.
pub fn is_local_source(path: &Path) -> bool {
    path.exists()
        || path.is_absolute()
        || path.starts_with(".")
        || path.components().next().is_some_and(|c| c.as_os_str() == "..")
        || path.extension().is_some_and(|ext| ext == "json")
}

/// Read a model config with the same error classes and context in every frontend.
pub fn read_config(path: impl AsRef<Path>) -> Result<TTSConfig> {
    read_json(path.as_ref(), "config")
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path, kind: &str) -> Result<T> {
    let text = std::fs::read_to_string(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => {
            Error::NotFound(format!("cannot read {kind} {}: {e}", path.display()))
        }
        _ => Error::Io(e),
    })?;
    serde_json::from_str(&text)
        .map_err(|e| Error::InvalidData(format!("cannot parse {kind} {}: {e}", path.display())))
}

fn validate_relative_path(path: &str) -> Result<()> {
    if path.is_empty()
        || path.contains(['\\', ':'])
        || path.split('/').any(|p| p.is_empty() || p == "." || p == "..")
    {
        return Err(Error::InvalidData(format!(
            "artifact path '{path}' must be relative to the checkpoint root"
        )));
    }
    Ok(())
}

fn validate_voice_name(name: &str) -> Result<()> {
    if name.trim().is_empty()
        || name != name.trim()
        || name == "."
        || name == ".."
        || name.contains(['/', '\\', ':'])
        || name.chars().any(char::is_control)
    {
        return Err(Error::InvalidData(format!(
            "voice name '{name}' must be a nonempty portable filename component"
        )));
    }
    Ok(())
}

fn validate_artifact(artifact: &Artifact) -> Result<()> {
    validate_relative_path(&artifact.path)?;
    if let Some(hash) = &artifact.sha256
        && (hash.len() != 64 || !hash.bytes().all(|c| c.is_ascii_hexdigit()))
    {
        return Err(Error::InvalidData(format!("invalid SHA-256 for '{}'", artifact.path)));
    }
    Ok(())
}

fn required_file(path: &Path) -> Result<PathBuf> {
    if !path.is_file() {
        return Err(Error::NotFound(format!("checkpoint file is missing: {}", path.display())));
    }
    Ok(path.to_path_buf())
}

fn resolve_artifact(dir: &Path, artifact: &Artifact) -> Result<PathBuf> {
    let path = required_file(&dir.join(&artifact.path))?;
    if let Some(expected) = &artifact.sha256 {
        let mut file = std::fs::File::open(&path)?;
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 65536];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            digest.update(&buffer[..n]);
        }
        let actual = format!("{:x}", digest.finalize());
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(Error::InvalidData(format!("SHA-256 mismatch for '{}'", artifact.path)));
        }
    }
    Ok(path)
}
