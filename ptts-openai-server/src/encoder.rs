//! The response formats, each encoding the model's audio as it arrives.

use crate::mp3::Mp3Encoder;
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Mp3,
    Opus,
    Wav,
    /// Headerless 16-bit little-endian mono.
    Pcm,
}

pub enum Encoder {
    Mp3(Mp3Encoder),
    Opus(kaudio::ogg_opus::Encoder),
    Wav { sample_rate: u32 },
    Pcm,
}

impl Encoder {
    pub fn new(format: Format, sample_rate: u32) -> Result<Self> {
        Ok(match format {
            Format::Mp3 => Self::Mp3(Mp3Encoder::new(sample_rate)?),
            Format::Opus => Self::Opus(kaudio::ogg_opus::Encoder::new(sample_rate as usize)?),
            Format::Wav => Self::Wav { sample_rate },
            Format::Pcm => Self::Pcm,
        })
    }

    /// What goes before the audio: Ogg's header pages, or a WAV header without a length, which
    /// is not known while streaming.
    pub fn header(&self) -> Result<Vec<u8>> {
        let mut header = vec![];
        match self {
            Self::Opus(opus) => header.extend_from_slice(opus.header_data()),
            Self::Wav { sample_rate } => {
                ptts::wav::write_header(&mut header, *sample_rate, 1, None)?
            }
            Self::Mp3(_) | Self::Pcm => {}
        }
        Ok(header)
    }

    pub fn encode(&mut self, pcm: &[f32]) -> Result<Vec<u8>> {
        match self {
            Self::Mp3(mp3) => mp3.encode(pcm),
            Self::Opus(opus) => Ok(opus.encode_page(pcm)?),
            Self::Wav { .. } | Self::Pcm => {
                let mut bytes = Vec::with_capacity(pcm.len() * 2);
                ptts::wav::write_samples(&mut bytes, pcm)?;
                Ok(bytes)
            }
        }
    }

    /// What the encoder still holds once the audio has ended: MP3's last frames, and nothing
    /// for the other formats.
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        match self {
            Self::Mp3(mp3) => mp3.finish(),
            _ => Ok(vec![]),
        }
    }
}
