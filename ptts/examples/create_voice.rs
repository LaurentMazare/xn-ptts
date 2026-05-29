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
    tracing::info!("loading config from {}", args.config);
    let (cfg, model_path) = if args.config.ends_with("json") {
        let cfg: ptts::tts_model::TTSConfig =
            serde_json::from_str(&std::fs::read_to_string(&args.config)?)?;
        let config = std::fs::canonicalize(args.config)?;
        let parent = config.parent().context("config path has no parent")?;
        let model_path = match args.weights.as_ref() {
            None => parent.join("model.safetensors"),
            Some(p) => std::path::PathBuf::from_str(p)?,
        };
        (cfg, model_path)
    } else {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(args.config);
        let cfg = repo.get("config.json")?;
        let cfg: ptts::tts_model::TTSConfig = serde_json::from_str(&std::fs::read_to_string(cfg)?)?;
        let model_path = match args.weights.as_ref() {
            None => repo.get("model.safetensors")?,
            Some(p) => std::path::PathBuf::from_str(p)?,
        };
        (cfg, model_path)
    };
    let model_ext = cfg.model_ext();
    let mimi_sample_rate = cfg.mimi.sample_rate;
    tracing::info!(?model_ext, "model extension");
    tracing::info!("loading voice from audio file {}", args.input);

    let (mut pcm, sample_rate) = audio_helpers::pcm_decode(&args.input)?;
    ptts::utils::normalize_loudness(&mut pcm, sample_rate)?;
    let sample_rate = sample_rate as usize;
    let pcm = if sample_rate != mimi_sample_rate {
        audio_helpers::resample(&pcm, sample_rate, mimi_sample_rate)?
    } else {
        pcm
    };
    tracing::info!("loaded audio with {} samples", pcm.len());
    // Trim it to 10s max.
    let pcm = if pcm.len() > mimi_sample_rate * 10 {
        tracing::info!("trimming audio to 10 seconds");
        pcm[..mimi_sample_rate * 10].to_vec()
    } else {
        pcm
    };
    let pcm_tensor = Tensor::from_vec(pcm, (1, 1, ()), &dev)?.to::<f32>()?;

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
    let data_info = std::collections::HashMap::from([(
        "model_ext".to_string(),
        model_ext.unwrap_or("unknown".to_string()),
    )]);
    xn::safetensors::save_with_data_info(&tensors, Some(data_info), &args.output)?;
    Ok(())
}
