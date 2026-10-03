//! MP3 encoding through the system's LAME (`libmp3lame`). It is linked dynamically: LAME is
//! LGPL, and a shared library keeps it replaceable without relinking this binary.

use anyhow::{Result, bail};
use std::ffi::c_int;

/// LAME's `lame_global_flags`, only ever handled through a pointer.
#[repr(C)]
struct Lame {
    _private: [u8; 0],
}

#[link(name = "mp3lame")]
unsafe extern "C" {
    fn lame_init() -> *mut Lame;
    fn lame_set_in_samplerate(gfp: *mut Lame, rate: c_int) -> c_int;
    fn lame_set_num_channels(gfp: *mut Lame, channels: c_int) -> c_int;
    fn lame_set_mode(gfp: *mut Lame, mode: c_int) -> c_int;
    fn lame_set_brate(gfp: *mut Lame, kbps: c_int) -> c_int;
    fn lame_set_quality(gfp: *mut Lame, quality: c_int) -> c_int;
    fn lame_set_bWriteVbrTag(gfp: *mut Lame, on: c_int) -> c_int;
    fn lame_set_write_id3tag_automatic(gfp: *mut Lame, on: c_int);
    fn lame_init_params(gfp: *mut Lame) -> c_int;
    fn lame_encode_buffer_ieee_float(
        gfp: *mut Lame,
        pcm_l: *const f32,
        pcm_r: *const f32,
        samples: c_int,
        mp3: *mut u8,
        mp3_len: c_int,
    ) -> c_int;
    fn lame_encode_flush(gfp: *mut Lame, mp3: *mut u8, mp3_len: c_int) -> c_int;
    fn lame_close(gfp: *mut Lame) -> c_int;
}

/// `MPEG_mode::MONO` in `lame.h`.
const MONO: c_int = 3;

/// The most LAME can emit for one call, as `lame.h` documents it.
fn worst_case(samples: usize) -> usize {
    samples * 5 / 4 + 7200
}

/// Mono MP3 at a constant 64 kbps, which is plenty for speech.
pub struct Mp3Encoder(*mut Lame);

// LAME keeps no thread-local state, and this handle is only used through `&mut self`.
unsafe impl Send for Mp3Encoder {}

impl Mp3Encoder {
    pub fn new(sample_rate: u32) -> Result<Self> {
        let gfp = unsafe { lame_init() };
        if gfp.is_null() {
            bail!("lame_init failed");
        }
        // Owned from here, so an error below still closes it.
        let encoder = Self(gfp);
        let ok = unsafe {
            lame_set_in_samplerate(gfp, sample_rate as c_int) == 0
                && lame_set_num_channels(gfp, 1) == 0
                && lame_set_mode(gfp, MONO) == 0
                && lame_set_brate(gfp, 64) == 0
                && lame_set_quality(gfp, 2) == 0
                // A stream has no length for a Xing header, and nothing to tag.
                && lame_set_bWriteVbrTag(gfp, 0) == 0
        };
        unsafe { lame_set_write_id3tag_automatic(gfp, 0) };
        if !ok || unsafe { lame_init_params(gfp) } < 0 {
            bail!("LAME rejected mono {sample_rate} Hz at 64 kbps");
        }
        Ok(encoder)
    }

    /// MP3 frames for `pcm`, which is in [-1, 1]. Often empty: LAME buffers until a frame fills.
    pub fn encode(&mut self, pcm: &[f32]) -> Result<Vec<u8>> {
        let mut mp3 = vec![0u8; worst_case(pcm.len())];
        // The right channel is ignored for mono, but must still be a valid pointer.
        let n = unsafe {
            lame_encode_buffer_ieee_float(
                self.0,
                pcm.as_ptr(),
                pcm.as_ptr(),
                pcm.len() as c_int,
                mp3.as_mut_ptr(),
                mp3.len() as c_int,
            )
        };
        if n < 0 {
            bail!("LAME failed to encode ({n})");
        }
        mp3.truncate(n as usize);
        Ok(mp3)
    }

    /// The frames LAME still holds. Nothing can be encoded after this.
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        let mut mp3 = vec![0u8; worst_case(0)];
        let n = unsafe { lame_encode_flush(self.0, mp3.as_mut_ptr(), mp3.len() as c_int) };
        if n < 0 {
            bail!("LAME failed to flush ({n})");
        }
        mp3.truncate(n as usize);
        Ok(mp3)
    }
}

impl Drop for Mp3Encoder {
    fn drop(&mut self) {
        unsafe { lame_close(self.0) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_of_tone_becomes_mp3_frames() {
        let rate = 24000;
        let pcm: Vec<f32> = (0..rate)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5)
            .collect();
        let mut encoder = Mp3Encoder::new(rate as u32).unwrap();
        let mut mp3 = vec![];
        for chunk in pcm.chunks(1920) {
            mp3.extend(encoder.encode(chunk).unwrap());
        }
        mp3.extend(encoder.finish().unwrap());
        // Every MPEG audio frame starts with an 11-bit sync word.
        assert!(mp3[0] == 0xFF && mp3[1] & 0xE0 == 0xE0, "no frame sync: {:02x?}", &mp3[..4]);
        // One second at 64 kbps is about 8 KB.
        assert!((6000..10000).contains(&mp3.len()), "{} bytes", mp3.len());
    }
}
