//! Python bindings for Phonon.
//!
//! ```python
//! import ptts
//!
//! tts = ptts.TTS(config="/path/to/model", lang="en")
//! tts.save("out.wav", "Hello world")
//! ```
//!
//! The pipeline is `ptts::synth::Synth`, so this file is a translation layer:
//! locating a checkpoint, numpy in and out, releasing the GIL around the slow
//! parts, and letting Ctrl-C through between audio chunks.
//!
//! One invariant throughout: **never hold a lock across a GIL reacquisition**.
//! A thread parked in `py.detach` with a guard alive deadlocks against any
//! other Python thread wanting the same lock, and Ctrl-C reaches neither.

use numpy::{PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use ptts::checkpoint::{
    Checkpoint, ResolveOptions, is_local_source, read_config, weight_candidates,
};
use ptts::loader::DEFAULT_VOICE_FILE;
use ptts::preprocess::{Normalize, Rules};
use ptts::synth::{DeviceKind, Quant, SpeechOptions, SpeechStream, Synth};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Map a `ptts` error onto the Python exception its class calls for.
///
/// `ptts::Error`'s variants are failure classes rather than one per message, so this is a table
/// rather than a pattern-match on prose -- and a misspelt voice stops being the same exception
/// as an unreadable checkpoint, which is what `ValueError` throughout made it.
fn to_py_err(e: ptts::Error) -> PyErr {
    use pyo3::exceptions as exc;

    let msg = e.to_string();
    match e {
        ptts::Error::InvalidArgument(_) => exc::PyValueError::new_err(msg),
        // `LookupError` is the base of `KeyError` and covers both "no such voice" and "no such
        // checkpoint file" without claiming either is a dict lookup.
        ptts::Error::UnknownVoice { .. } | ptts::Error::NotFound(_) => {
            exc::PyLookupError::new_err(msg)
        }
        ptts::Error::Unsupported(_) => exc::PyNotImplementedError::new_err(msg),
        ptts::Error::Io(e) => PyErr::from(e),
        // `Busy` is retryable and the rest are not, but Python has no better shared base than
        // `RuntimeError` for any of them. `Error` is `#[non_exhaustive]`, so the `_` arm is
        // where a variant added upstream lands rather than being folded into one above.
        _ => exc::PyRuntimeError::new_err(msg),
    }
}

trait IntoPy<R> {
    fn py(self) -> PyResult<R>;
}

impl<R, E: Into<ptts::Error>> IntoPy<R> for Result<R, E> {
    fn py(self) -> PyResult<R> {
        self.map_err(|e| to_py_err(e.into()))
    }
}

/// Resolve an explicitly supplied local directory/config or Hub repo ID.
fn resolve(config: Option<&str>, revision: Option<&str>, quant: Quant) -> ptts::Result<Checkpoint> {
    match config {
        Some(path) if is_local_source(std::path::Path::new(path)) => {
            if revision.is_some() {
                return Err(ptts::Error::InvalidArgument(
                    "revision requires a Hugging Face repo ID".into(),
                ));
            }
            Checkpoint::resolve(path, ResolveOptions { quant, weights: None })
        }
        Some(repo_id) => resolve_hub(&hub(repo_id)?, repo_id, revision, quant),
        None => Err(ptts::Error::InvalidArgument(
            "config is required: supply a local model directory or Hugging Face repo ID".into(),
        )),
    }
}

/// A Hub repo: its own `config.json`, weights, a
/// tokenizer, and voices under `voices/` or `embeddings/` plus an optional
/// `default-voice.safetensors`. Only the files that are used get downloaded.
fn resolve_hub(
    repo: &HubRepo,
    repo_id: &str,
    revision: Option<&str>,
    quant: Quant,
) -> ptts::Result<Checkpoint> {
    let cfg = read_config(hub_get(repo, "config.json", revision)?)?;
    // One listing rather than a request per guessed name, and the only way to learn a repo's
    // voices. If listing is unavailable, only standard weight and default-voice paths are tried.
    let listing: Option<Vec<String>> = repo
        .list_tree()
        .maybe_revision(revision.map(str::to_owned))
        .recursive(true)
        .send()
        .ok()
        .map(|entries| {
            entries
                .into_iter()
                .filter_map(|entry| match entry {
                    hf_hub::repository::files::RepoTreeEntry::File { path, .. } => Some(path),
                    _ => None,
                })
                .collect()
        });
    // Without a listing every name is a guess, and one that was never cached fails offline with
    // a network error rather than a not-found. So it counts as absent, letting a later name the
    // cache does hold be found, and the first such error is reported only if no weights resolve.
    let first_err = std::cell::RefCell::new(None);
    let get_optional = |name: &str| -> ptts::Result<Option<std::path::PathBuf>> {
        if listing.as_ref().is_some_and(|files| !files.iter().any(|f| f == name)) {
            return Ok(None);
        }
        match hub_get(repo, name, revision) {
            Ok(path) => Ok(Some(path)),
            Err(ptts::Error::NotFound(_)) => Ok(None),
            Err(e) if listing.is_none() => {
                first_err.borrow_mut().get_or_insert(e);
                Ok(None)
            }
            Err(e) => Err(e),
        }
    };

    let candidates = weight_candidates(quant);
    let mut model_path = None;
    for name in candidates {
        if let Some(path) = get_optional(name)? {
            model_path = Some(path);
            break;
        }
    }
    let Some(model_path) = model_path else {
        return Err(first_err.borrow_mut().take().unwrap_or_else(|| {
            ptts::Error::NotFound(format!(
                "no weights in `{repo_id}`; expected one of {}",
                candidates.join(", ")
            ))
        }));
    };
    let tokenizer_path = hub_get(repo, "tokenizer.json", revision)?;

    let voice_files = ptts::loader::voice_paths(listing.clone().unwrap_or_default());
    let mut voices = vec![];
    for (name, file) in
        voice_files.into_iter().chain([("default".to_string(), DEFAULT_VOICE_FILE.to_string())])
    {
        // A voice that will not download is skipped, as one that will not load is: `TTS.voices`
        // shows which ones made it.
        if let Ok(Some(path)) = get_optional(&file) {
            push_voice(&mut voices, name, path);
        }
    }
    Ok(Checkpoint {
        config: cfg,
        weights: model_path,
        tokenizer: Some(tokenizer_path),
        voices,
        quant,
    })
}

/// Add a voice unless one of that name is already there: the first found wins, so `voices/`
/// beats `embeddings/`, and both beat `default-voice.safetensors`. `Synth` would otherwise keep
/// whichever it registered last.
fn push_voice(
    voices: &mut Vec<(String, std::path::PathBuf)>,
    name: String,
    path: std::path::PathBuf,
) {
    if !voices.iter().any(|(n, _)| *n == name) {
        voices.push((name, path));
    }
}

type HubRepo = hf_hub::HFRepositorySync<hf_hub::repository::RepoTypeModel>;

fn hub(repo_id: &str) -> ptts::Result<HubRepo> {
    // Not `NotFound`: a client that will not start is an environment failure, not a name that
    // failed to resolve. `Io` reaches Python as `OSError`, which is what a caller retries on.
    let client = hf_hub::HFClientSync::new().map_err(|e| {
        ptts::Error::Io(std::io::Error::other(format!("cannot reach the Hugging Face Hub: {e}")))
    })?;
    let (owner, name) = hf_hub::split_id(repo_id);
    Ok(client.model(owner, name))
}

/// Download `filename` from `repo`, or find it in the local cache.
fn hub_get(
    repo: &HubRepo,
    filename: &str,
    revision: Option<&str>,
) -> ptts::Result<std::path::PathBuf> {
    // Only a genuine not-found is a name that failed to resolve. Offline, a timeout, a 429 or a
    // 5xx are environment failures: they reach Python as `OSError`, which is what a caller
    // retries on, rather than as a `LookupError` that says the file does not exist.
    repo.download_file()
        .filename(filename)
        .maybe_revision(revision.map(str::to_owned))
        .send()
        .map_err(|e| match e {
            hf_hub::HFError::EntryNotFound { .. } | hf_hub::HFError::LocalEntryNotFound { .. } => {
                ptts::Error::NotFound(format!("`{filename}` is not in the repo: {e}"))
            }
            hf_hub::HFError::Io(io) => ptts::Error::Io(io),
            other => ptts::Error::Io(std::io::Error::other(format!(
                "cannot fetch `{filename}`: {other}"
            ))),
        })
}

/// Flatten a conditioning embedding of shape `[T, dim]` or `[1, T, dim]`.
fn embedding_dims(arr: &PyReadonlyArrayDyn<'_, f32>) -> PyResult<(Vec<f32>, usize, usize)> {
    let (frames, dim) = match arr.shape() {
        [t, d] => (*t, *d),
        [1, t, d] => (*t, *d),
        shape => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected an embedding of shape [T, dim] or [1, T, dim], got {shape:?}"
            )));
        }
    };
    Ok((arr.as_array().iter().copied().collect(), frames, dim))
}

/// A loaded Phonon model.
#[pyclass(name = "TTS", module = "ptts")]
struct Tts {
    inner: Arc<Mutex<Synth>>,
    /// Voice for calls that name none. `SynthBuilder` picks its own during
    /// `build()`, which is before the voices below are registered.
    default_voice: Option<String>,
}

#[pymethods]
impl Tts {
    /// `TTS(config=None, device=None, quant=None, voice=None, temperature=0.3, seed=..., cfg_coef=None, eos_threshold=None, *, lang, rewrites=None, conditions=None, revision=None)`
    ///
    /// `lang` is required and keyword-only: the language text is normalized as
    /// before it is tokenized, one of `"en"`, `"fr"`, `"de"`, `"es"` or
    /// `"pt"`. Normalization makes the model noticeably better, but the spoken
    /// forms of `@`, `+` and `=` differ per language, so there is nothing safe
    /// to default to. `lang="none"` or `lang=None` hands text to the tokenizer
    /// as written, for callers that normalize it themselves.
    ///
    /// `rewrites` picks which word rewrites run on the normalized text:
    /// `"default"` (what `None` means: numbers, currency, dashed-digits,
    /// emails, urls, abbreviations, elongations), `"all"` (those and phones,
    /// times, dates), `"none"`, or a comma-separated list of rule names.
    #[new]
    #[pyo3(signature = (
        config = None,
        device = None,
        quant = None,
        voice = None,
        temperature = 0.3,
        seed = 4242424242424242,
        cfg_coef = None,
        eos_threshold = None,
        *,
        lang,
        rewrites = None,
        conditions = None,
        revision = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        config: Option<String>,
        device: Option<&str>,
        quant: Option<&str>,
        voice: Option<String>,
        temperature: f32,
        seed: u64,
        cfg_coef: Option<f32>,
        eos_threshold: Option<f32>,
        lang: Option<&str>,
        rewrites: Option<&str>,
        conditions: Option<HashMap<String, Bound<'_, PyAny>>>,
        revision: Option<String>,
    ) -> PyResult<Self> {
        let conditions = conditions
            .unwrap_or_default()
            .into_iter()
            .map(|(name, value)| Ok((name, value.str()?.to_string())))
            .collect::<PyResult<Vec<_>>>()?;
        // `None` opts out as well as `"none"`: `lang` has to be passed, but a
        // caller forwarding a config value should not have to special-case the
        // absent one.
        let normalize = match lang {
            None => Normalize::OFF,
            Some(lang) => lang.parse::<Normalize>().py()?,
        };
        let rules = match rewrites {
            None => Rules::DEFAULT,
            Some(rewrites) => rewrites.parse::<Rules>().py()?,
        };
        let normalize = normalize.with_rules(rules);
        let device = match device {
            None => DeviceKind::Auto,
            Some(name) => name.parse::<DeviceKind>().py()?,
        };
        let quant = match quant {
            None => Quant::F32,
            Some(name) => name.parse::<Quant>().py()?,
        };
        // Device and weight-format checks run before downloading; `SynthBuilder`
        // would catch them, but only once the checkpoint is on disk.
        quant.check_device(device).py()?;
        // Loading reads hundreds of megabytes and runs no Python.
        py.detach(move || {
            let artifacts = resolve(config.as_deref(), revision.as_deref(), quant).py()?;
            let mut builder = artifacts
                .builder(normalize)
                .device(device)
                .quant(quant)
                .temperature(temperature)
                .seed(seed);
            if let Some(cfg_coef) = cfg_coef {
                builder = builder.cfg_coef(cfg_coef);
            }
            if let Some(eos_threshold) = eos_threshold {
                builder = builder.eos_threshold(eos_threshold);
            }
            for (name, value) in conditions {
                builder = builder.condition(name, value);
            }
            let mut synth = builder.build().py()?;
            artifacts.register_voices(&mut synth);
            if let Some(name) = voice.as_deref()
                && !synth.voices().iter().any(|v| v == name)
            {
                return Err(to_py_err(ptts::Error::UnknownVoice {
                    name: name.to_string(),
                    known: synth.voices(),
                }));
            }
            Ok(Self { inner: Arc::new(Mutex::new(synth)), default_voice: voice })
        })
    }

    /// Sample rate of the audio this model produces, in Hz.
    #[getter]
    fn sample_rate(&self) -> PyResult<u32> {
        Ok(self.lock()?.sample_rate())
    }

    /// Names of the registered voices, sorted.
    #[getter]
    fn voices(&self) -> PyResult<Vec<String>> {
        Ok(self.lock()?.voices())
    }

    /// Device the model is running on, e.g. `"cpu"`.
    #[getter]
    fn device(&self) -> PyResult<String> {
        Ok(self.lock()?.device_name())
    }

    /// Weight format actually loaded, e.g. `"q8_0"`.
    #[getter]
    fn quant(&self) -> PyResult<&'static str> {
        Ok(self.lock()?.quant().as_str())
    }

    /// Sample rate `clone_voice` expects its PCM in, in Hz.
    #[getter]
    fn voice_prompt_sample_rate(&self) -> PyResult<u32> {
        Ok(self.lock()?.voice_prompt_sample_rate())
    }

    /// True if this checkpoint can clone voices from audio.
    #[getter]
    fn supports_voice_cloning(&self) -> PyResult<bool> {
        Ok(self.lock()?.supports_voice_cloning())
    }

    /// Synthesize `text` and return the waveform as a 1-D float32 array.
    #[pyo3(signature = (text, *, voice=None, temperature=None, seed=None, cfg_coef=None))]
    fn synth<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        voice: Option<String>,
        temperature: Option<f32>,
        seed: Option<u64>,
        cfg_coef: Option<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let opts = self.opts(voice, temperature, seed, cfg_coef);
        let stream = self.start(py, text, &opts)?;
        let pcm = drain(py, stream)?;
        Ok(PyArray1::from_vec(py, pcm))
    }

    /// Synthesize `text` straight to a mono 16-bit WAV file, returning its
    /// duration in seconds.
    #[pyo3(signature = (path, text, *, voice=None, temperature=None, seed=None, cfg_coef=None))]
    #[allow(clippy::too_many_arguments)]
    fn save(
        &self,
        py: Python<'_>,
        path: std::path::PathBuf,
        text: &str,
        voice: Option<String>,
        temperature: Option<f32>,
        seed: Option<u64>,
        cfg_coef: Option<f32>,
    ) -> PyResult<f64> {
        let opts = self.opts(voice, temperature, seed, cfg_coef);
        let stream = self.start(py, text, &opts)?;
        let sample_rate = stream.sample_rate();
        let pcm = drain(py, stream)?;
        let seconds = pcm.len() as f64 / sample_rate as f64;
        py.detach(|| ptts::wav::write_wav_file(&path, &pcm, sample_rate).py())?;
        Ok(seconds)
    }

    /// Synthesize `text`, yielding float32 chunks as the decoder produces them.
    ///
    /// The returned object is an iterator. Generation pauses once its bounded buffers
    /// fill if it is unread, and resumes as chunks are consumed. Closing or dropping it
    /// stops generation and waits for its worker threads to finish.
    #[pyo3(signature = (text, *, voice=None, temperature=None, seed=None, cfg_coef=None))]
    fn stream(
        &self,
        py: Python<'_>,
        text: &str,
        voice: Option<String>,
        temperature: Option<f32>,
        seed: Option<u64>,
        cfg_coef: Option<f32>,
    ) -> PyResult<AudioStream> {
        let opts = self.opts(voice, temperature, seed, cfg_coef);
        let stream = self.start(py, text, &opts)?;
        Ok(AudioStream { inner: Mutex::new(Some(stream)) })
    }

    /// Register a voice from a precomputed embedding file.
    fn add_voice(&self, py: Python<'_>, name: &str, path: std::path::PathBuf) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        py.detach(move || inner.lock().map_err(|_| poisoned())?.add_voice_file(name, &path).py())
    }

    /// Replace the conditions, without reloading the model. Values go through `str()`,
    /// as `conditions=` does, and an invalid one leaves the previous conditions in place.
    fn set_conditions(
        &self,
        py: Python<'_>,
        conditions: HashMap<String, Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let conditions = conditions
            .into_iter()
            .map(|(name, value)| Ok((name, value.str()?.to_string())))
            .collect::<PyResult<HashMap<_, _>>>()?;
        let inner = Arc::clone(&self.inner);
        py.detach(move || inner.lock().map_err(|_| poisoned())?.set_conditions(conditions).py())
    }

    /// Register a voice from an in-memory conditioning embedding of shape
    /// `[T, dim]` or `[1, T, dim]`.
    ///
    /// `null_embedding` is the encoding of equal-length silence, which CFG
    /// needs on models whose null branch is conditioned on silence.
    #[pyo3(signature = (name, embedding, *, null_embedding=None))]
    fn add_voice_from_embedding(
        &self,
        name: &str,
        embedding: PyReadonlyArrayDyn<'_, f32>,
        null_embedding: Option<PyReadonlyArrayDyn<'_, f32>>,
    ) -> PyResult<()> {
        let (emb, frames, dim) = embedding_dims(&embedding)?;
        let null = match null_embedding.as_ref() {
            None => None,
            Some(arr) => Some(embedding_dims(arr)?.0),
        };
        self.lock()?.add_voice_from_embedding(name, &emb, frames, dim, null.as_deref()).py()
    }

    /// Clone a voice from ~10s of speech, given as float32 PCM at
    /// `voice_prompt_sample_rate`.
    fn clone_voice(
        &self,
        py: Python<'_>,
        name: &str,
        pcm: numpy::PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let pcm = pcm.as_slice()?.to_vec();
        let inner = Arc::clone(&self.inner);
        py.detach(move || inner.lock().map_err(|_| poisoned())?.add_voice_from_pcm(name, &pcm).py())
    }

    fn __repr__(&self) -> PyResult<String> {
        let synth = self.lock()?;
        Ok(format!(
            "TTS(device='{}', quant='{}', sample_rate={}, voices={:?})",
            synth.device_name(),
            synth.quant().as_str(),
            synth.sample_rate(),
            synth.voices()
        ))
    }
}

impl Tts {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Synth>> {
        self.inner.lock().map_err(|_| poisoned())
    }

    /// Per-call settings, with this model's default voice filled in.
    fn opts(
        &self,
        voice: Option<String>,
        temperature: Option<f32>,
        seed: Option<u64>,
        cfg_coef: Option<f32>,
    ) -> SpeechOptions {
        SpeechOptions {
            voice: voice.or_else(|| self.default_voice.clone()),
            temperature,
            seed,
            cfg_coef,
            max_tokens_per_chunk: None,
            post_process: None,
        }
    }

    /// Start a generation with the lock taken *inside* `py.detach`. Holding it
    /// across a GIL reacquisition — which `drain` does per chunk — deadlocks
    /// against any other Python thread calling in.
    fn start(&self, py: Python<'_>, text: &str, opts: &SpeechOptions) -> PyResult<SpeechStream> {
        let inner = Arc::clone(&self.inner);
        let opts = opts.clone();
        py.detach(move || {
            let synth = inner.lock().map_err(|_| poisoned())?;
            synth.stream_with(text, &opts).py()
        })
    }
}

/// A panic inside a `&mut self` method would leave the model half-updated, so a
/// poisoned lock is reported rather than papered over.
fn poisoned() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("the model is unusable: a previous call panicked")
}

/// Collect a whole stream, letting Ctrl-C through between chunks.
///
/// The GIL is released while waiting on each chunk and reacquired to check for
/// signals, so a long generation stays interruptible without the caller passing
/// a polling interval.
fn drain(py: Python<'_>, mut stream: SpeechStream) -> PyResult<Vec<f32>> {
    let mut pcm = Vec::new();
    loop {
        match py.detach(|| stream.next()) {
            None => return Ok(pcm),
            Some(chunk) => pcm.extend_from_slice(&chunk.py()?),
        }
        py.check_signals()?;
    }
}

/// An in-progress generation. Iterate it for float32 chunks.
#[pyclass(module = "ptts")]
struct AudioStream {
    // `SpeechStream` owns an mpsc receiver, which is `Send` but not `Sync`,
    // while `#[pyclass]` wants both; the mutex bridges that. `None` after
    // `close`, so a closed stream iterates as empty rather than raising.
    inner: Mutex<Option<SpeechStream>>,
}

#[pymethods]
impl AudioStream {
    /// Sample rate of the chunks, in Hz.
    #[getter]
    fn sample_rate(&self) -> PyResult<u32> {
        match self.inner.lock().map_err(|_| poisoned())?.as_ref() {
            Some(stream) => Ok(stream.sample_rate()),
            None => Err(pyo3::exceptions::PyValueError::new_err("this stream is closed")),
        }
    }

    /// Stop generating and release the worker threads.
    fn close(&self) -> PyResult<()> {
        *self.inner.lock().map_err(|_| poisoned())? = None;
        Ok(())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let inner = &slf.inner;
        let next = py.detach(|| -> PyResult<Option<Result<Vec<f32>, ptts::Error>>> {
            let mut guard = inner.lock().map_err(|_| poisoned())?;
            Ok(guard.as_mut().and_then(|stream| stream.next()))
        })?;
        match next {
            None => Ok(None),
            Some(chunk) => Ok(Some(PyArray1::from_vec(py, chunk.py()?))),
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

/// Number of CPU threads used for tensor ops.
#[pyfunction]
fn get_num_threads() -> usize {
    xn::utils::get_num_threads()
}

/// Set the number of CPU threads used for tensor ops. Call before loading a
/// model: it sizes a global thread pool that is built once.
#[pyfunction]
fn set_num_threads(num_threads: usize) {
    xn::utils::set_num_threads(num_threads);
}

/// The device names this build accepts, most capable first. `"auto"` picks the
/// first of these.
#[pyfunction]
fn available_devices() -> Vec<&'static str> {
    let mut devices = vec![];
    if cfg!(feature = "cuda") {
        devices.push("cuda");
    }
    if cfg!(feature = "vulkan") {
        devices.push("vulkan");
    }
    if cfg!(feature = "metal") {
        devices.push("metal");
    }
    devices.push("cpu");
    devices
}

/// The weight formats `quant=` accepts. All are CPU-only.
#[pyfunction]
fn available_quants() -> Vec<&'static str> {
    vec!["f32", "q8_0", "q8_1", "q8k", "q6k", "q5_0", "q5_1", "q5k", "q4_0", "q4_1", "q4k"]
}

/// The runtime's build configuration, for bug reports.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = PyDict::new(py);
    info.set_item("version", env!("CARGO_PKG_VERSION"))?;
    info.set_item("devices", available_devices())?;
    info.set_item("avx", xn::with_avx())?;
    info.set_item("neon", xn::with_neon())?;
    info.set_item("f16c", xn::with_f16c())?;
    info.set_item("threads", xn::utils::get_num_threads())?;
    Ok(info)
}

#[pymodule(name = "_ptts")]
fn ptts_(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Tts>()?;
    m.add_class::<AudioStream>()?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(available_devices, m)?)?;
    m.add_function(wrap_pyfunction!(available_quants, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_files_directly_in_a_voice_dir_are_voices() {
        assert_eq!(
            ptts::loader::voice_file_name("voices/Freya.safetensors"),
            Some(("voices", "Freya"))
        );
        assert_eq!(
            ptts::loader::voice_file_name("embeddings/alba.safetensors"),
            Some(("embeddings", "alba"))
        );
        for path in [
            "embeddings_v2/alba.safetensors",
            "languages/french/embeddings/alba.safetensors",
            "voices/sub/alba.safetensors",
            "voices/readme.md",
            "default-voice.safetensors",
        ] {
            assert_eq!(ptts::loader::voice_file_name(path), None, "{path}");
        }
    }

    #[test]
    fn q8_weights_lead_only_for_q8() {
        assert_eq!(weight_candidates(Quant::Q80)[0], "model.q8.gguf");
        for quant in [Quant::F32, Quant::Q4k, Quant::Q81] {
            assert_eq!(weight_candidates(quant)[0], "model.safetensors", "{quant:?}");
        }
    }

    #[test]
    fn the_first_voice_of_a_name_wins() {
        let mut voices = vec![];
        push_voice(&mut voices, "a".into(), "voices/a.safetensors".into());
        push_voice(&mut voices, "a".into(), "embeddings/a.safetensors".into());
        assert_eq!(voices, [("a".to_string(), "voices/a.safetensors".into())]);
    }
}
