mod encoder;
mod handler;
mod model;
mod protocol;
mod utils;

use anyhow::Result;
use axum::Router;
use axum::routing::any;
use clap::Parser;
use ptts::preprocess::{Normalize, Rules};
use ptts::synth::{DeviceKind, Quant};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "ptts-ws-server")]
#[command(about = "WebSocket server for Phonon")]
struct Args {
    #[arg(long, default_value = "0.0.0.0:8080")]
    addr: String,

    #[arg(long)]
    config: Option<std::path::PathBuf>,

    /// Optional directory of additional voice safetensors to load. Each
    /// `*.safetensors` file is loaded as a voice keyed by its file stem; load
    /// errors are logged and skipped rather than fatal.
    #[arg(long)]
    voice_dir: Option<std::path::PathBuf>,

    #[arg(long, default_value_t = 0.3)]
    temperature: f32,

    #[arg(long, default_value_t = 4242424242424242)]
    seed: u64,

    #[arg(long, default_value_t = 4096)]
    max_seq_len: usize,

    /// Device to run on: auto, cpu, cuda, vulkan or metal. `auto` picks the GPU backend this
    /// build was compiled with, if any, and the CPU otherwise.
    #[arg(long, default_value = "auto")]
    device: String,

    /// Quantization for the flow_lm transformer linear weights.
    /// One of: q8|q8_0, q8_1, q8k, q6k, q5|q5_0, q5_1, q5k, q4|q4_0, q4_1, q4k.
    /// CPU only.
    #[arg(long)]
    quant: Option<String>,

    /// Language incoming text is normalized as before tokenizing: `en`, `fr`, `de`, `es` or
    /// `pt`. Required: the spoken forms differ per language, so there is nothing safe to guess.
    /// `none` serves the text as written, which the model reads less well.
    #[arg(long)]
    lang: String,

    /// Which word rewrites run on the normalized text: `default` (numbers, currency,
    /// dashed-digits, emails, urls), `all` (those and phones, times, dates), `none`, or a
    /// comma-separated list of rule names. Has no effect with `--lang none`.
    #[arg(long, default_value = "default")]
    rewrites: String,
}

fn init_tracing() {
    // `info` for everything but the Hub download stack: `hf_hub` transfers through the Xet
    // backend, which reports every retry policy and range probe at `info`. Keep in sync with
    // `LOG_DIRECTIVES` in `ptts/examples/model_helpers.rs`.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(
            "info,xet=warn,xet_client=warn,xet_data=warn,xet_runtime=warn,xet_core_structures=warn",
        )
    });
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::Layer::new().with_target(false))
        .with(filter)
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();

    // The checkpoint is downloaded with hf-hub's async client on this runtime.
    // The weight loading that follows is CPU-bound and blocks the runtime
    // thread, which is fine here: nothing is served until it is done.
    let app_state = build_app_state(&args).await?;

    let app = Router::new()
        .route("/speech/tts", any(handler::ws_handler))
        .with_state(app_state)
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    tracing::info!(addr = %args.addr, "listening on /speech/tts");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutdown requested");
        })
        .await?;
    Ok(())
}

async fn build_app_state(args: &Args) -> Result<model::AppState> {
    let device = args.device.parse::<DeviceKind>()?;
    let quant = match args.quant.as_deref() {
        None => Quant::F32,
        Some(name) => name.parse::<Quant>()?,
    };
    // Both checks happen before `load_ptts` downloads anything: `SynthBuilder`
    // would catch them, but only after the checkpoint is on disk.
    quant.check_device(device)?;
    let normalize = args.lang.parse::<Normalize>()?.with_rules(args.rewrites.parse::<Rules>()?);
    let unavailable = match device {
        DeviceKind::Cuda if !cfg!(feature = "cuda") => Some("cuda"),
        DeviceKind::Vulkan if !cfg!(feature = "vulkan") => Some("vulkan"),
        DeviceKind::Metal if !cfg!(feature = "metal") => Some("metal"),
        _ => None,
    };
    if let Some(name) = unavailable {
        anyhow::bail!(
            "--device {name} requested, but this binary was built without --features {name}"
        );
    }
    model::load_ptts(
        args.config.as_ref(),
        args.voice_dir.as_ref(),
        device,
        quant,
        args.temperature,
        args.seed,
        args.max_seq_len,
        normalize,
    )
    .await
}
