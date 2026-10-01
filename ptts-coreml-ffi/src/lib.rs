//! C ABI over the CoreML Pocket TTS driver, for an iOS app.
//!
//! Empty on other targets, so the workspace still builds there.
//!
//! `ptts_prepare` compiles the bundled graphs once per install; `ptts_new` loads them, which is
//! the expensive part; `ptts_speak` may then be called repeatedly, streaming each frame's audio
//! to a callback as it is decoded. Every pointer handed out stays valid until the next call on
//! the same handle or `ptts_free`.
//!
//! Only the platform glue lives here: text preparation, normalization and tokenization are
//! `ptts`'s, and generation is `ptts_coreml`'s.
#![cfg(target_vendor = "apple")]

use ptts::Tokenizer;
use ptts::preprocess::Normalize;
use ptts_coreml::Weights;
use ptts_coreml::phonon::driver::{Config, Phonon};
use ptts_coreml::run::{Compute, Model};
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};

/// Where the flow LM runs.
pub const PTTS_UNIT_CPU: u32 = 0;
pub const PTTS_UNIT_ANE: u32 = 2;

pub struct PttsHandle {
    phonon: Phonon,
    tokenizer: ptts::tok::Tok,
    normalize: Normalize,
    dir: PathBuf,
    /// The voice names as NUL-separated bytes, so `ptts_voices` can hand out a borrowed pointer.
    voice_blob: Vec<u8>,
    /// The last utterance's PCM, owned here so `PttsResult::pcm` stays valid until the next call.
    audio: Vec<f32>,
    last_error: Option<CString>,
}

/// One utterance's audio and timings. `pcm` is 24 kHz mono, owned by the handle.
#[repr(C)]
pub struct PttsResult {
    pub pcm: *const f32,
    pub pcm_len: usize,
    pub sample_rate: u32,
    pub frames: u32,
    pub total_ms: f64,
    pub ttfa_ms: f64,
    pub per_frame_ms: f64,
    /// A median hides the slow frames that set the realtime factor, so the mean and the slowest.
    pub mean_frame_ms: f64,
    pub max_frame_ms: f64,
    pub rtf: f64,
}

/// Called once per decoded frame with its PCM, on the generating thread. Return false to stop
/// speaking: generation ends there and `ptts_speak` returns normally with what was produced.
pub type PttsFrameFn = extern "C" fn(*const f32, usize, *mut c_void) -> bool;

static GLOBAL_ERROR: std::sync::Mutex<Option<CString>> = std::sync::Mutex::new(None);

fn set_global_error(e: &str) {
    *GLOBAL_ERROR.lock().unwrap() = CString::new(e).ok();
}

fn voice_names(dir: &Path) -> Result<Vec<String>, String> {
    let rd = std::fs::read_dir(dir.join("voices")).map_err(|e| format!("voices/: {e}"))?;
    let mut v: Vec<String> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
        .collect();
    v.sort();
    Ok(v)
}

fn load_voice(dir: &Path, name: &str) -> Result<(Vec<f32>, usize), String> {
    let w = Weights::open(&dir.join("voices").join(format!("{name}.safetensors")))?;
    let (shape, data) = w.get(w.names().first().ok_or("empty voice file")?)?;
    let t = if shape.len() == 3 { shape[1] } else { shape[0] };
    Ok((data.to_vec(), t))
}

fn open(dir: &Path, unit: u32, lang: &str) -> Result<PttsHandle, String> {
    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("bundle.json")).map_err(|e| format!("bundle.json: {e}"))?,
    )
    .map_err(|e| format!("bundle.json: {e}"))?;
    let int = |k: &str| {
        meta[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("bundle.json has no {k}"))
    };
    let float = |k: &str| {
        meta[k].as_f64().map(|v| v as f32).ok_or_else(|| format!("bundle.json has no {k}"))
    };
    let dims = &meta["dims"];
    let dim = |k: &str| {
        dims[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("bundle.json has no dims.{k}"))
    };
    let cfg = Config {
        dims: ptts_coreml::phonon::flow_lm::Dims {
            d: dim("d")?,
            heads: dim("heads")?,
            layers: dim("layers")?,
            ff: dim("ff")?,
            ldim: dim("ldim")?,
            flow_d: dim("flow_d")?,
            flow_blocks: dim("flow_blocks")?,
        },
        ctx: int("ctx")?,
        prefill_len: int("prefill_len")?,
        mimi_window: int("mimi_window")?,
        max_frames: int("max_frames")?,
        eos_threshold: float("eos_threshold")?,
        temperature: float("temperature")?,
        seed: 0,
        flow_unit: if unit == PTTS_UNIT_ANE {
            Compute::CpuAndNeuralEngine
        } else {
            Compute::CpuOnly
        },
    };
    let voices = voice_names(dir)?;
    let (voice, vlen) = load_voice(dir, voices.first().ok_or("no voices in the bundle")?)?;
    let tokenizer = ptts::tok::Tok::open(&dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let normalize = Normalize::parse(lang).map_err(|e| e.to_string())?;
    let mut voice_blob = Vec::new();
    for v in &voices {
        voice_blob.extend_from_slice(v.as_bytes());
        voice_blob.push(0);
    }
    voice_blob.push(0);
    Ok(PttsHandle {
        phonon: Phonon::load(dir, cfg, voice, vlen)?,
        tokenizer,
        normalize,
        dir: dir.to_path_buf(),
        voice_blob,
        audio: Vec::new(),
        last_error: None,
    })
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Mimi's frame rate, for the per-chunk frame budget.
const FRAME_RATE: f64 = 12.5;

/// Split `text` into chunks the prefill graph takes, as `(tokens, frames_after_eos)`.
///
/// The same plan `ptts::synth` makes: normalize the whole input first, split it at sentence
/// ends, and prepare each sentence on its own. A single sentence longer than the graph's rows
/// is split again at its middle word; the tokenizer's ids per word do not depend on the words
/// around them, so that loses nothing but the prosody across the cut.
fn plan(h: &PttsHandle, text: &str) -> Result<Vec<(Vec<u32>, usize)>, String> {
    let max = h.phonon.max_tokens();
    let text = h.normalize.apply(text);
    let sentences = ptts::tts_model::split_into_best_sentences(&h.tokenizer, &text, Some(max))
        .map_err(|e| e.to_string())?;
    let mut chunks = Vec::new();
    let mut todo: Vec<String> = sentences.into_iter().rev().collect();
    while let Some(s) = todo.pop() {
        let (prepared, frames_after_eos) = ptts::tts_model::prepare_text_prompt(&s);
        let tokens = h.tokenizer.encode(&prepared).map_err(|e| e.to_string())?;
        let words: Vec<&str> = s.split_whitespace().collect();
        if tokens.len() > max && words.len() > 1 {
            let (a, b) = words.split_at(words.len() / 2);
            todo.push(b.join(" "));
            todo.push(a.join(" "));
        } else if tokens.len() > max {
            return Err(format!(
                "one word is {} tokens, over the {max} the model takes",
                tokens.len()
            ));
        } else if !tokens.is_empty() {
            chunks.push((tokens, frames_after_eos));
        }
    }
    if chunks.is_empty() {
        return Err("nothing to say: the text is empty".into());
    }
    Ok(chunks)
}

fn speak(
    h: &mut PttsHandle,
    text: &str,
    sink: &mut dyn FnMut(&[f32]) -> bool,
) -> Result<PttsResult, String> {
    let start = std::time::Instant::now();
    let chunks = plan(h, text)?;
    let mut audio = Vec::new();
    let (mut ms, mut ttfa) = (Vec::new(), None);
    for (tokens, frames_after_eos) in chunks {
        let budget = ptts::plan::frame_budget(tokens.len(), FRAME_RATE);
        let t = h.phonon.generate(&tokens, frames_after_eos, budget, &mut |pcm: &[f32]| {
            // CoreML has been seen to hand back a stray non-finite sample; silence it rather
            // than send it to the speaker.
            let from = audio.len();
            audio.extend(pcm.iter().map(|v| if v.is_finite() { *v } else { 0.0 }));
            sink(&audio[from..])
        })?;
        ttfa.get_or_insert(start.elapsed() - t.total + t.ttfa);
        ms.extend(t.frames.iter().map(|d| d.as_secs_f64() * 1e3));
        if t.stopped {
            break;
        }
    }
    h.audio = audio;
    let total_ms = start.elapsed().as_secs_f64() * 1e3;
    Ok(PttsResult {
        pcm: h.audio.as_ptr(),
        pcm_len: h.audio.len(),
        sample_rate: 24_000,
        frames: ms.len() as u32,
        total_ms,
        ttfa_ms: ttfa.unwrap_or_default().as_secs_f64() * 1e3,
        mean_frame_ms: ms.iter().sum::<f64>() / ms.len().max(1) as f64,
        max_frame_ms: ms.iter().copied().fold(0.0, f64::max),
        per_frame_ms: median(ms),
        rtf: h.audio.len() as f64 / 24.0 / total_ms,
    })
}

/// Compile the bundle's graphs if they are not compiled yet: about 10 s on an iPhone 16 Pro, once per
/// install, and kept beside the packages. `ptts_new` does this itself when needed; calling it
/// first lets an app show that it is happening. Returns 1 if it compiled anything, 0 if nothing
/// needed doing, -1 on error.
///
/// # Safety
/// `dir` must be a NUL-terminated UTF-8 path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_prepare(dir: *const c_char) -> i32 {
    let Ok(dir) = unsafe { CStr::from_ptr(dir) }.to_str().map(PathBuf::from) else { return -1 };
    let run = || -> Result<bool, String> {
        let mut did = false;
        for e in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
            let p = e.map_err(|e| e.to_string())?.path();
            if p.extension().is_some_and(|x| x == "mlpackage") {
                did |= Model::precompile(&p, &p.with_extension("mlmodelc"))?;
            }
        }
        Ok(did)
    };
    match run() {
        Ok(did) => i32::from(did),
        Err(e) => {
            set_global_error(&e);
            -1
        }
    }
}

/// Load a bundle for `unit` (`PTTS_UNIT_ANE` or `PTTS_UNIT_CPU`), normalizing text as `lang`
/// (`en`, `fr`, `de`, `es`, `pt`, or `none`). Null on failure; see `ptts_last_error(NULL)`.
///
/// # Safety
/// `dir` and `lang` must be NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_new(
    dir: *const c_char,
    unit: u32,
    lang: *const c_char,
) -> *mut PttsHandle {
    let dir = unsafe { CStr::from_ptr(dir) }.to_string_lossy().to_string();
    let lang = unsafe { CStr::from_ptr(lang) }.to_string_lossy().to_string();
    match open(Path::new(&dir), unit, &lang) {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(e) => {
            set_global_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Speak `text`, calling `cb(pcm, n, user)` with each frame's audio as soon as it is decoded.
/// `out` then describes the whole utterance. Returns false on failure; see `ptts_last_error`.
///
/// # Safety
/// `h` must come from `ptts_new`; `text` must be NUL-terminated UTF-8; `cb` must be valid for
/// the duration of the call; `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_speak(
    h: *mut PttsHandle,
    text: *const c_char,
    cb: PttsFrameFn,
    user: *mut c_void,
    out: *mut PttsResult,
) -> bool {
    let h = unsafe { &mut *h };
    let text = unsafe { CStr::from_ptr(text) }.to_string_lossy().to_string();
    match speak(h, &text, &mut |pcm: &[f32]| cb(pcm.as_ptr(), pcm.len(), user)) {
        Ok(r) => {
            unsafe { *out = r };
            true
        }
        Err(e) => {
            h.last_error = CString::new(e).ok();
            false
        }
    }
}

/// Voice names, sorted, as one NUL-separated block ending in a second NUL. Borrowed.
///
/// # Safety
/// `h` must come from `ptts_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_voices(h: *const PttsHandle) -> *const c_char {
    unsafe { (*h).voice_blob.as_ptr() as *const c_char }
}

/// Switch speaker. Returns false on failure; see `ptts_last_error`.
///
/// # Safety
/// `h` must come from `ptts_new`; `name` must be NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_set_voice(h: *mut PttsHandle, name: *const c_char) -> bool {
    let h = unsafe { &mut *h };
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy().to_string();
    match load_voice(&h.dir, &name).and_then(|(v, n)| h.phonon.set_voice(v, n)) {
        Ok(()) => true,
        Err(e) => {
            h.last_error = CString::new(e).ok();
            false
        }
    }
}

/// The last error on `h`, or the last construction error when `h` is null. Borrowed until the
/// next failing call.
///
/// # Safety
/// `h` must be null or come from `ptts_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_last_error(h: *const PttsHandle) -> *const c_char {
    if h.is_null() {
        // The CString is kept in the static, so the pointer outlives the guard.
        return GLOBAL_ERROR.lock().unwrap().as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
    }
    unsafe { (*h).last_error.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()) }
}

/// # Safety
/// `h` must come from `ptts_new` and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_free(h: *mut PttsHandle) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}
