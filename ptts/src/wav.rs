//! Writing PCM as a 16-bit WAV file.

use std::io::prelude::*;

pub trait Sample {
    fn to_i16(&self) -> i16;
}

impl Sample for f32 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
}

impl Sample for f64 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
}

impl Sample for i16 {
    fn to_i16(&self) -> i16 {
        *self
    }
}

/// The 44-byte header of a 16-bit PCM WAV holding `data_len` bytes of samples. `None` writes
/// both sizes as `0xFFFF_FFFF`, as a stream does whose length is not known yet.
pub fn write_header<W: Write>(
    w: &mut W,
    sample_rate: u32,
    n_channels: u32,
    data_len: Option<u32>,
) -> std::io::Result<()> {
    let (riff_len, data_len) = match data_len {
        Some(n) => (36 + n, n),
        None => (u32::MAX, u32::MAX),
    };
    w.write_all(b"RIFF")?;
    w.write_all(&riff_len.to_le_bytes())?; // everything after these first 8 bytes
    w.write_all(b"WAVE")?;

    // Format block
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // size of this block
    w.write_all(&1u16.to_le_bytes())?; // 1: PCM int, 2: PCM float
    w.write_all(&(n_channels as u16).to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&(sample_rate * 2 * n_channels).to_le_bytes())?; // bytes per second
    w.write_all(&(2 * n_channels as u16).to_le_bytes())?; // bytes per frame
    w.write_all(&16u16.to_le_bytes())?; // bits per sample

    // Data block
    w.write_all(b"data")?;
    w.write_all(&data_len.to_le_bytes())
}

/// Append samples after a [`write_header`].
pub fn write_samples<W: Write, S: Sample>(w: &mut W, samples: &[S]) -> std::io::Result<()> {
    for sample in samples {
        w.write_all(&sample.to_i16().to_le_bytes())?
    }
    Ok(())
}

pub fn write_pcm_as_wav<W: Write, S: Sample>(
    w: &mut W,
    samples: &[S],
    sample_rate: u32,
    n_channels: u32,
) -> std::io::Result<()> {
    write_header(w, sample_rate, n_channels, Some(samples.len() as u32 * 2))?;
    write_samples(w, samples)
}

/// Write mono PCM to a WAV file at `path`.
///
/// A convenience over [`write_pcm_as_wav`] for the common case of saving a
/// finished generation: open the file, wrap it in a `BufWriter`, write, flush.
///
/// An I/O failure is [`crate::Error::Io`], so a missing directory reaches Python as `OSError`.
pub fn write_wav_file<P: AsRef<std::path::Path>>(
    path: P,
    pcm: &[f32],
    sample_rate: u32,
) -> crate::Result<()> {
    let path = path.as_ref();
    let named = |e: std::io::Error| {
        std::io::Error::new(e.kind(), format!("cannot write {}: {e}", path.display()))
    };
    let mut writer = std::io::BufWriter::new(std::fs::File::create(path).map_err(named)?);
    write_pcm_as_wav(&mut writer, pcm, sample_rate, 1).map_err(named)?;
    writer.flush().map_err(named)?;
    Ok(())
}
