#[path = "audio_helpers.rs"]
mod audio_helpers;

use anyhow::{Context, Result};
use clap::Parser;
use ptts::tts_model::MimiEnc;
use xn::Tensor;
use xn::nn::VB;

#[derive(Parser, Debug)]
#[command(name = "create-voice")]
#[command(about = "Generate some embedding files for Pocket TTS")]
struct Args {
    #[arg(long)]
    config: String,

    #[arg(long)]
    weights: Option<String>,

    #[arg(long)]
    output: std::path::PathBuf,

    /// Voice to use
    #[arg(long)]
    input: String,
}

fn remap_key(name: &str) -> Option<String> {
    // Skip keys we don't need
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
    {
        return None;
    }

    let mut name = name.to_string();

    // Order matters: more specific replacements first
    name = name.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    );
    name = name.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    );
    name = name.replace("flow_lm.backbone.", "flow_lm.transformer.");
    name = name.replace("flow_lm.flow.", "flow_lm.flow_net.");
    name = name.replace("mimi.model.", "mimi.");

    Some(name)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    run(args)?;
    Ok(())
}

fn run(args: Args) -> Result<()> {
    use std::str::FromStr;

    let dev = xn::CpuDevice;
    let cfg: ptts::tts_model::TTSConfig =
        serde_json::from_str(&std::fs::read_to_string(&args.config)?)?;
    tracing::info!("loading voice from audio file {}", args.input);

    let (pcm, sample_rate) = audio_helpers::pcm_decode(&args.input)?;
    let sample_rate = sample_rate as usize;
    let pcm = if sample_rate != cfg.mimi.sample_rate {
        audio_helpers::resample(&pcm, sample_rate, cfg.mimi.sample_rate)?
    } else {
        pcm
    };
    tracing::info!("loaded audio with {} samples", pcm.len());
    // Trim it to 10s max.
    let pcm = if pcm.len() > cfg.mimi.sample_rate * 10 {
        tracing::info!("trimming audio to 10 seconds");
        pcm[..cfg.mimi.sample_rate * 10].to_vec()
    } else {
        pcm
    };
    let pcm_tensor = Tensor::from_vec(pcm, (1, 1, ()), &dev)?.to::<f32>()?;

    let config = std::fs::canonicalize(args.config)?;
    let parent = config.parent().context("config path has no parent")?;
    let model_path = match args.weights.as_ref() {
        None => parent.join("model.safetensors"),
        Some(p) => std::path::PathBuf::from_str(p)?,
    };
    tracing::info!(?model_path, "loading model");
    let vb = if model_path.extension().and_then(|v| v.to_str()) == Some("gguf") {
        let reader = std::fs::File::open(&model_path)?;
        let reader = std::io::BufReader::new(reader);
        VB::load_gguf_with_key_map(reader, dev, remap_key)?
    } else {
        VB::load_with_key_map(&[&model_path], dev, remap_key)?
    };
    let vb = vb.root();
    let mimi_enc: MimiEnc<xn::Unquantized<f32, xn::CpuDevice>> = MimiEnc::load(&vb, &cfg)?;
    tracing::info!("encoding audio to latent");
    let emb = mimi_enc.encode_audio(&pcm_tensor)?;
    tracing::info!(?emb, "encoded audio to latent");
    let tensors = std::collections::HashMap::from([("emb".to_string(), xn::TypedTensor::F32(emb))]);
    xn::safetensors::save(&tensors, &args.output)?;
    Ok(())
}
