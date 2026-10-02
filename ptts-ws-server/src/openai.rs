//! The OpenAI-compatible speech API: `POST /v1/audio/speech`, and the `GET /v1/models`,
//! `GET /v1/audio/voices` and `GET /health` that clients probe. Anything with an "OpenAI TTS
//! with a custom base URL" setting can use this server unchanged.

use crate::encoder::{Encoder, Format};
use crate::model::AppState;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use ptts::synth::{SpeechOptions, SpeechStream};

/// OpenAI's own voice names. Clients default to them, and Pipecat sends nothing else, so each
/// one speaks in the checkpoint's default voice unless a voice of that name is registered.
const OPENAI_VOICES: [&str; 13] = [
    "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin", "nova", "onyx", "sage",
    "shimmer", "verse",
];

/// OpenAI's limit on `input`.
const MAX_INPUT_CHARS: usize = 4096;

/// The request body OpenAI specifies. `model`, `instructions` and `stream_format` are accepted
/// and ignored: there is one model, it takes no style prompt, and the reply is always raw audio,
/// which LiveKit, the one client that asks for SSE, also reads.
#[derive(serde::Deserialize)]
struct SpeechRequest {
    input: String,
    #[serde(default)]
    voice: Option<Voice>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    speed: Option<f64>,
}

/// A voice name, or OpenAI's `{"id": ...}` form for a custom voice.
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum Voice {
    Name(String),
    Custom { id: String },
}

/// OpenAI's `pcm`: headerless 16-bit mono at this rate, which clients such as Pipecat assume.
const PCM_RATE: usize = 24000;

/// A request that passed every check, ready to synthesize.
#[derive(Debug, PartialEq)]
struct Checked {
    input: String,
    /// `None` for the checkpoint's default voice.
    voice: Option<String>,
    format: Format,
    content_type: &'static str,
}

/// Why a request is a 400: the field at fault, if one is, and what to say.
type Rejection = (Option<&'static str>, String);

/// Check a request body against what this server can do, `voices` being the registered ones.
fn check(body: &[u8], voices: &[String]) -> Result<Checked, Rejection> {
    let req: SpeechRequest =
        serde_json::from_slice(body).map_err(|e| (None, format!("invalid request: {e}")))?;
    if req.input.trim().is_empty() {
        return Err((Some("input"), "input is empty".into()));
    }
    let chars = req.input.chars().count();
    if chars > MAX_INPUT_CHARS {
        let message = format!("input is {chars} characters; the limit is {MAX_INPUT_CHARS}");
        return Err((Some("input"), message));
    }
    if let Some(speed) = req.speed
        && speed != 1.0
    {
        let message = format!(
            "speed {speed} is not supported: this model has no rate control yet; omit speed or \
             send 1.0"
        );
        return Err((Some("speed"), message));
    }
    let (format, content_type) = match req.response_format.as_deref().unwrap_or("mp3") {
        "mp3" => (Format::Mp3, "audio/mpeg"),
        "opus" => (Format::OggOpus, "audio/opus"),
        "wav" => (Format::Wav, "audio/wav"),
        // Pinned rather than the checkpoint's own: free when that is 24 kHz, resampled if not.
        "pcm" => (Format::pcm(PCM_RATE), "audio/pcm"),
        other => {
            let message =
                format!("response_format '{other}' is not supported; use mp3, opus, wav or pcm");
            return Err((Some("response_format"), message));
        }
    };
    let voice = match req.voice {
        None => None,
        Some(Voice::Name(name) | Voice::Custom { id: name }) => {
            // In any case, as OpenAI's own names are matched.
            if let Some(voice) = voices.iter().find(|v| v.eq_ignore_ascii_case(&name)) {
                Some(voice.clone())
            } else if name.eq_ignore_ascii_case("default")
                || OPENAI_VOICES.contains(&name.to_lowercase().as_str())
            {
                // `default` means the default voice, as on the WebSocket route.
                None
            } else {
                let message = format!(
                    "unknown voice '{name}'; this server has {}, and takes any OpenAI voice name \
                     as its default voice",
                    voices.join(", ")
                );
                return Err((Some("voice"), message));
            }
        }
    };
    Ok(Checked { input: req.input, voice, format, content_type })
}

pub async fn speech(State(app): State<AppState>, body: Bytes) -> Response {
    let Checked { input, voice, format, content_type } = match check(&body, &app.voices) {
        Ok(checked) => checked,
        Err((param, message)) => return error(StatusCode::BAD_REQUEST, param, message),
    };
    let mut opts = SpeechOptions::default().seed(next_seed(app.seed_base));
    if let Some(voice) = voice {
        opts = opts.voice(voice);
    }

    // Synthesis and encoding share one blocking thread. The response starts only once the
    // stream has, so a request the model refuses gets an error, not a 200 with no audio.
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<std::io::Result<Vec<u8>>>(16);
    let frame = app.frame_size as usize;
    let sample_rate = app.sample_rate as usize;
    tokio::task::spawn_blocking(move || {
        let stream = match app.synth.stream_with(&input, &opts) {
            Ok(stream) => stream,
            Err(e) => {
                let _ = started_tx.send(Err(e));
                return;
            }
        };
        let _ = started_tx.send(Ok(()));
        if let Err(e) = encode(stream, format, frame, sample_rate, &tx) {
            tracing::warn!(error = %e, "speech request failed after it started");
            let _ = tx.blocking_send(Err(std::io::Error::other(e.to_string())));
        }
    });
    match started_rx.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            let status = match e {
                ptts::Error::InvalidArgument(_) | ptts::Error::UnknownVoice { .. } => {
                    StatusCode::BAD_REQUEST
                }
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            return error(status, None, e.to_string());
        }
        Err(_) => {
            let message = "the generation task ended before it started";
            return error(StatusCode::INTERNAL_SERVER_ERROR, None, message);
        }
    }
    let body = futures_util::stream::poll_fn(move |cx| rx.poll_recv(cx));
    ([(header::CONTENT_TYPE, content_type)], Body::from_stream(body)).into_response()
}

/// Encode `stream` frame by frame onto `tx`. Returns early, without error, once the client has
/// gone: dropping `stream` stops the generation workers.
fn encode(
    stream: SpeechStream,
    format: Format,
    frame: usize,
    sample_rate: usize,
    tx: &tokio::sync::mpsc::Sender<std::io::Result<Vec<u8>>>,
) -> anyhow::Result<()> {
    let send = |bytes: Vec<u8>| bytes.is_empty() || tx.blocking_send(Ok(bytes)).is_ok();
    let mut encoder = Encoder::new(format, frame, sample_rate)?;
    if let Some(header) = encoder.header()
        && !send(header.to_vec())
    {
        return Ok(());
    }
    // A chunk can carry several frames, and the resampling and Opus encoders want one at a
    // time, as the WebSocket handler feeds them.
    for chunk in stream {
        for pcm in chunk?.chunks(frame) {
            if !send(encoder.encode(pcm)?.data) {
                return Ok(());
            }
        }
    }
    send(encoder.finish()?);
    Ok(())
}

/// A different seed per request, as the WebSocket handler gives each stream.
fn next_seed(base: u64) -> u64 {
    static REQUESTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = REQUESTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    base ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

pub async fn models(State(app): State<AppState>) -> Response {
    let model = serde_json::json!({
        "id": app.model_name, "object": "model", "created": 0, "owned_by": "gradium",
    });
    json(StatusCode::OK, serde_json::json!({ "object": "list", "data": [model] }))
}

/// The registered voices, in the shape Kokoro-FastAPI uses and Home Assistant's OpenAI TTS
/// integration reads.
pub async fn voices(State(app): State<AppState>) -> Response {
    json(StatusCode::OK, serde_json::json!({ "voices": app.voices }))
}

pub async fn health() -> Response {
    json(StatusCode::OK, serde_json::json!({ "status": "ok" }))
}

/// An error in OpenAI's shape, which client SDKs parse into their own error types.
fn error(status: StatusCode, param: Option<&str>, message: impl Into<String>) -> Response {
    let kind = if status.is_client_error() { "invalid_request_error" } else { "server_error" };
    let error = serde_json::json!({
        "message": message.into(), "type": kind, "param": param, "code": null,
    });
    json(status, serde_json::json!({ "error": error }))
}

fn json(status: StatusCode, value: serde_json::Value) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], value.to_string()).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voices(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    fn checked(body: &str) -> Checked {
        check(body.as_bytes(), &voices(&["Freya", "Toby"])).unwrap()
    }

    fn rejected(body: &str) -> Option<&'static str> {
        check(body.as_bytes(), &voices(&["Freya", "Toby"])).unwrap_err().0
    }

    #[test]
    fn a_bare_request_is_mp3_in_the_default_voice() {
        let got = checked(r#"{"model": "tts-1", "input": "Hello."}"#);
        assert_eq!(got.format, Format::Mp3);
        assert_eq!(got.content_type, "audio/mpeg");
        assert_eq!(got.voice, None);
    }

    #[test]
    fn each_format_has_its_encoder_and_type() {
        for (name, format, content_type) in [
            ("mp3", Format::Mp3, "audio/mpeg"),
            ("opus", Format::OggOpus, "audio/opus"),
            ("wav", Format::Wav, "audio/wav"),
            ("pcm", Format::pcm(24000), "audio/pcm"),
        ] {
            let got = checked(&format!(r#"{{"input": "Hi.", "response_format": "{name}"}}"#));
            assert_eq!((got.format, got.content_type), (format, content_type), "{name}");
        }
        assert_eq!(
            rejected(r#"{"input": "Hi.", "response_format": "aac"}"#),
            Some("response_format")
        );
    }

    #[test]
    fn voices_match_in_any_case_and_openai_names_are_the_default() {
        assert_eq!(
            checked(r#"{"input": "Hi.", "voice": "freya"}"#).voice.as_deref(),
            Some("Freya")
        );
        assert_eq!(
            checked(r#"{"input": "Hi.", "voice": {"id": "Toby"}}"#).voice.as_deref(),
            Some("Toby")
        );
        assert_eq!(checked(r#"{"input": "Hi.", "voice": "Alloy"}"#).voice, None);
        assert_eq!(checked(r#"{"input": "Hi.", "voice": "default"}"#).voice, None);
        assert_eq!(rejected(r#"{"input": "Hi.", "voice": "nobody"}"#), Some("voice"));
        // A voice registered as `default` is that voice, not an alias.
        let with_default = check(br#"{"input": "Hi.", "voice": "DEFAULT"}"#, &voices(&["default"]));
        assert_eq!(with_default.unwrap().voice.as_deref(), Some("default"));
    }

    #[test]
    fn requests_the_model_cannot_serve_are_rejected() {
        assert_eq!(checked(r#"{"input": "Hi.", "speed": 1.0}"#).input, "Hi.");
        assert_eq!(rejected(r#"{"input": "Hi.", "speed": 1.5}"#), Some("speed"));
        assert_eq!(rejected(r#"{"input": "  "}"#), Some("input"));
        let long = format!(r#"{{"input": "{}"}}"#, "a".repeat(MAX_INPUT_CHARS + 1));
        assert_eq!(rejected(&long), Some("input"));
        assert_eq!(rejected(r#"{"voice": "alloy"}"#), None, "no input at all");
    }
}
