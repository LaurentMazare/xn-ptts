//! C interface over the Core ML Phonon driver, which the PhononTTS Swift package wraps.
//! Empty on non-Apple targets, so the workspace still builds there.
//!
//! `ptts_new` loads a model bundle, compiling it for the device first if it has not been; it is
//! the expensive call. `ptts_speak` may then be called repeatedly, streaming each frame's audio
//! to a callback as it is decoded.
//!
//! Only the platform glue lives here: text preparation, normalization and tokenization are
//! `ptts`'s, and generation is `ptts_coreml`'s. `include/ptts.h` declares this interface by
//! hand, so any change to a signature or to `PttsResult` has to be made there too.
#![cfg(target_vendor = "apple")]

use ptts::plan::Chunk;
use ptts::preprocess::Normalize;
use ptts_coreml::Weights;
use ptts_coreml::phonon::driver::{Config, Phonon, Voice};
use ptts_coreml::run::Compute;
use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

/// Where the flow LM runs.
pub const PTTS_UNIT_CPU: u32 = 0;
pub const PTTS_UNIT_ANE: u32 = 2;

pub struct PttsHandle {
    phonon: Phonon,
    tokenizer: ptts::tok::Tok,
    normalize: Normalize,
    dir: PathBuf,
    voices: Vec<String>,
    /// The voice names as NUL-separated bytes, so `ptts_voices` can hand out a borrowed pointer.
    voice_blob: Vec<u8>,
    last_error: Option<CString>,
}

/// One utterance's timings. The audio itself went to the callback.
#[repr(C)]
pub struct PttsResult {
    pub frames: u32,
    /// 24 kHz samples delivered.
    pub samples: usize,
    /// From the call to the first audio being delivered.
    pub ttfa_ms: f64,
    pub total_ms: f64,
}

/// Called once per decoded frame with its PCM, on the generating thread. Return false to stop
/// speaking: generation ends there and `ptts_speak` returns normally with what was produced.
pub type PttsFrameFn = extern "C" fn(*const f32, usize, *mut c_void) -> bool;

thread_local! {
    /// The last `ptts_new` failure on this thread. Thread-local, so a failure elsewhere cannot
    /// free the string a caller is still reading.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn c_string(e: &str) -> CString {
    CString::new(e.replace('\0', " ")).expect("NULs were replaced")
}

/// Run `f`, turning a panic into an error: unwinding across the C interface aborts the app.
fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| {
        let msg = p
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| p.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown".into());
        Err(format!("internal error: {msg}"))
    })
}

fn voice_names(dir: &Path) -> Vec<String> {
    ptts::loader::voices_in(&dir.join("voices")).into_iter().map(|(name, _)| name).collect()
}

/// A voice as the exporter writes it: an `emb` tensor of `[1, T, D]` or `[T, D]`, and for a
/// voice baked into the checkpoint, the `conditions` it is spoken with.
fn load_voice(dir: &Path, name: &str) -> Result<Voice, String> {
    let w = Weights::open(&dir.join("voices").join(format!("{name}.safetensors")))?;
    let (shape, data) = w.get("emb").map_err(|_| format!("voice {name} has no `emb` tensor"))?;
    let len = if shape.len() == 3 { shape[1] } else { shape[0] };
    let conditions = w.data("conditions").ok().map(<[f32]>::to_vec);
    Ok(Voice { emb: data.to_vec(), len, conditions })
}

fn open(dir: &Path, unit: u32, lang: &str) -> Result<PttsHandle, String> {
    let flow_unit = match unit {
        PTTS_UNIT_ANE => Compute::CpuAndNeuralEngine,
        PTTS_UNIT_CPU => Compute::CpuOnly,
        u => return Err(format!("unknown compute unit {u}")),
    };
    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("bundle.json")).map_err(|e| format!("bundle.json: {e}"))?,
    )
    .map_err(|e| format!("bundle.json: {e}"))?;
    let get =
        |v: &serde_json::Value, k: &str| v[k].as_f64().ok_or(format!("bundle.json has no {k}"));
    let int = |k: &str| get(&meta, k).map(|v| v as usize);
    let dim = |k: &str| get(&meta["dims"], k).map(|v| v as usize);
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
        eos_threshold: get(&meta, "eos_threshold")? as f32,
        temperature: get(&meta, "temperature")? as f32,
        seed: 0,
        flow_unit,
    };
    let voices = voice_names(dir);
    let voice = load_voice(dir, voices.first().ok_or("no voices in the bundle")?)?;
    let tokenizer = ptts::tok::Tok::open(&dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let normalize = lang.parse::<Normalize>().map_err(|e| e.to_string())?;
    let mut voice_blob = Vec::new();
    for v in &voices {
        voice_blob.extend_from_slice(v.as_bytes());
        voice_blob.push(0);
    }
    voice_blob.push(0);
    Ok(PttsHandle {
        phonon: Phonon::load(dir, cfg, voice)?,
        tokenizer,
        normalize,
        dir: dir.to_path_buf(),
        voices,
        voice_blob,
        last_error: None,
    })
}

/// Mimi's frame rate, for the per-chunk frame budget.
const FRAME_RATE: f64 = 12.5;

/// Split `text` into chunks the prefill graph takes.
///
/// The chunks every frontend makes, from [`ptts::plan::chunks`], with one addition the graph
/// forces: it has a fixed number of rows, so a single sentence longer than that is split again
/// at its middle word. The tokenizer's ids per word do not depend on the words around them, so
/// that loses nothing but the prosody across the cut.
fn plan(h: &PttsHandle, text: &str) -> Result<Vec<Chunk>, String> {
    let max = h.phonon.max_tokens();
    let planned = ptts::plan::chunks(&h.tokenizer, text, h.normalize, max, FRAME_RATE)
        .map_err(|e| e.to_string())?;
    let mut chunks = Vec::new();
    let mut todo: Vec<Chunk> = planned.into_iter().rev().collect();
    while let Some(chunk) = todo.pop() {
        if chunk.tokens.is_empty() {
            continue;
        }
        if chunk.tokens.len() <= max {
            chunks.push(chunk);
            continue;
        }
        let words: Vec<&str> = chunk.text.split_whitespace().collect();
        if words.len() < 2 {
            return Err(format!(
                "one word is {} tokens, over the {max} the model takes",
                chunk.tokens.len()
            ));
        }
        let (a, b) = words.split_at(words.len() / 2);
        for half in [b, a] {
            let half = Chunk::new(half.join(" "), &h.tokenizer, FRAME_RATE);
            todo.push(half.map_err(|e| e.to_string())?);
        }
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
    let (mut frames, mut samples, mut ttfa) = (0, 0, None);
    let mut pcm_buf = Vec::new();
    for chunk in chunks {
        let (tokens, budget) = (&chunk.tokens, chunk.frame_budget);
        let t =
            h.phonon.generate(tokens, chunk.frames_after_eos, budget, &mut |pcm: &[f32]| {
                // Core ML has been seen to hand back a stray non-finite sample; silence it rather
                // than send it to the speaker.
                pcm_buf.clear();
                pcm_buf.extend(pcm.iter().map(|v| if v.is_finite() { *v } else { 0.0 }));
                sink(&pcm_buf)
            })?;
        ttfa.get_or_insert(start.elapsed() - t.total + t.ttfa);
        frames += t.frames;
        samples += t.samples;
        if t.stopped {
            break;
        }
    }
    Ok(PttsResult {
        frames: frames as u32,
        samples,
        ttfa_ms: ttfa.unwrap_or_default().as_secs_f64() * 1e3,
        total_ms: start.elapsed().as_secs_f64() * 1e3,
    })
}

/// Load a bundle for `unit` (`PTTS_UNIT_ANE` or `PTTS_UNIT_CPU`), normalizing text as `lang`
/// (`en`, `fr`, `de`, `es`, `pt`, or `none`), and compiling the models for this device first if
/// they have not been (about 10 s on an iPhone 16 Pro, once per install). Null on failure; then
/// `ptts_last_error(NULL)`, on the same thread, says why.
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
    match guarded(|| open(Path::new(&dir), unit, &lang)) {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(e) => {
            LAST_ERROR.with(|l| *l.borrow_mut() = Some(c_string(&e)));
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
    let r = guarded(|| speak(h, &text, &mut |pcm: &[f32]| cb(pcm.as_ptr(), pcm.len(), user)));
    match r {
        Ok(r) => {
            unsafe { *out = r };
            true
        }
        Err(e) => {
            h.last_error = Some(c_string(&e));
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

/// Switch to one of `ptts_voices`. Returns false on failure; see `ptts_last_error`.
///
/// # Safety
/// `h` must come from `ptts_new`; `name` must be NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_set_voice(h: *mut PttsHandle, name: *const c_char) -> bool {
    let h = unsafe { &mut *h };
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy().to_string();
    let r = guarded(|| {
        if !h.voices.contains(&name) {
            return Err(format!("no voice named {name:?}"));
        }
        h.phonon.set_voice(load_voice(&h.dir, &name)?)
    });
    match r {
        Ok(()) => true,
        Err(e) => {
            h.last_error = Some(c_string(&e));
            false
        }
    }
}

/// The last error on `h`, or, when `h` is null, the last `ptts_new` failure on this thread.
/// Borrowed until the next failing call.
///
/// # Safety
/// `h` must be null or come from `ptts_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ptts_last_error(h: *const PttsHandle) -> *const c_char {
    if h.is_null() {
        // The string lives in this thread's slot until the next failure on this thread.
        return LAST_ERROR.with(|l| l.borrow().as_ref().map_or(std::ptr::null(), |c| c.as_ptr()));
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
