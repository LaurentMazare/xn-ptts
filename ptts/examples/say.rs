//! The shortest thing that makes a sound from an explicitly supplied model folder.
//!
//! ```text
//! cargo run --release --example say --features hf -- /path/to/model "hello world"
//! ```

use anyhow::Context as _;
use ptts::checkpoint::{Checkpoint, ResolveOptions};
use ptts::preprocess::Lang;
use ptts::synth::Quant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().context("usage: say <model directory> [text]")?;
    let text = args.next().unwrap_or_else(|| "Hello from Phonon.".to_string());
    let checkpoint = Checkpoint::resolve(dir, ResolveOptions { quant: Quant::Q80, weights: None })?;
    let mut tts = checkpoint.builder(Lang::En).build()?;
    checkpoint.register_voices(&mut tts);
    let pcm = tts.say(&text)?;
    ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate())?;
    println!("wrote out.wav ({:.2}s)", pcm.len() as f32 / tts.sample_rate() as f32);
    Ok(())
}
