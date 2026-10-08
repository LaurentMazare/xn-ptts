//! The shortest thing that makes a sound.
//!
//! ```text
//! cargo run --release --example say --features hf -- "hello world"
//! ```

#[path = "model_helpers.rs"]
mod model_helpers;

use ptts::preprocess::Lang;

fn main() -> anyhow::Result<()> {
    let text = std::env::args().nth(1).unwrap_or_else(|| "Hello from Phonon.".to_string());

    let checkpoint =
        model_helpers::from_hub(model_helpers::REPO_ID, None, ptts::synth::Quant::F32)?;
    // Which language to normalize as has no default: see `SynthBuilder::new`.
    let mut tts = checkpoint.builder(Lang::En).build()?;
    checkpoint.register_voices(&mut tts);

    let pcm = tts.say_with(&text, &ptts::synth::SpeechOptions::default().voice("alba"))?;
    ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate())?;

    println!("wrote out.wav ({:.2}s)", pcm.len() as f32 / tts.sample_rate() as f32);
    Ok(())
}
