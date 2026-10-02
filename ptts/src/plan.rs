//! Generation policy: how many frames to run, how much KV budget to reserve, and
//! when to stop.
//!
//! These rules were inlined — as literals — in every frontend: the `ptts`
//! and `bench` examples, `ptts-pyo3` (twice), `ptts-wasm` and `ptts-ws-server`
//! each carried their own copy of `((n / 3.0 + 2.0) * 12.5).ceil()` and their own
//! EOS countdown loop. They are pure functions of the token count and the config,
//! so they belong here, where they can be unit tested once.
//!
//! Every frontend is converted now. The `ptts` and `say` examples, `ptts-pyo3` and
//! `ptts-ws-server` reach these through [`crate::synth`], which applies them on
//! the caller's behalf; `bench` and `ptts-wasm` call them directly, because both
//! drive [`crate::tts_model::TTSModel`] themselves — `bench` to attribute
//! sampling and decoding time to the frame that caused them, `ptts-wasm` because
//! `Synth` generates on background threads and the browser has none.

use crate::Tokenizer;
use crate::preprocess::Normalize;
use crate::tts_model::{prepare_text_prompt, split_into_best_sentences};
use crate::{Error, Result};

/// Extra KV-cache entries reserved on top of the text tokens and the generated
/// frames, covering the voice-prompt frames (~125 at 12.5Hz for a 10s prompt)
/// plus slack.
pub const PROMPT_SEQ_HEADROOM: usize = 512;

/// Frames of audio to generate for a text prompt of `num_tokens` tokens.
///
/// Roughly three tokens per second of speech, plus two seconds of slack,
/// converted to frames at the codec frame rate. This is an upper bound: normal
/// generation stops earlier, when the model signals EOS (see [`EosPolicy`]).
///
/// `frame_rate` comes from `TTSConfig::mimi.frame_rate`. The frontends all
/// hardcoded `12.5`, which silently assumed the shipped config.
pub fn frame_budget(num_tokens: usize, frame_rate: f64) -> usize {
    ((num_tokens as f64 / 3.0 + 2.0) * frame_rate).ceil() as usize
}

/// KV-cache length to allocate for a single text chunk: its text tokens, the
/// frames it may generate, and [`PROMPT_SEQ_HEADROOM`] for the voice prompt.
///
/// A state is allocated once and reused across chunks, so callers with several
/// chunks should allocate the max over all of them.
pub fn seq_budget(num_tokens: usize, frame_budget: usize) -> usize {
    num_tokens + PROMPT_SEQ_HEADROOM + frame_budget
}

/// Tracks the tail of a generation: once the model reports EOS, a few more
/// frames are still emitted so the codec can close out the utterance cleanly.
///
/// `frames_after_eos` comes from [`crate::tts_model::prepare_text_prompt`] — 3
/// for very short prompts, 1 otherwise.
#[derive(Clone, Copy, Debug)]
pub struct EosPolicy {
    frames_after_eos: usize,
    countdown: Option<usize>,
}

impl EosPolicy {
    pub fn new(frames_after_eos: usize) -> Self {
        Self { frames_after_eos, countdown: None }
    }

    /// Records the EOS flag of the frame that was just generated and returns
    /// whether generation should stop now.
    ///
    /// Call this once per frame, *after* the frame has been handed to the
    /// decoder — the EOS frame itself is part of the output.
    pub fn should_stop(&mut self, is_eos: bool) -> bool {
        if is_eos && self.countdown.is_none() {
            self.countdown = Some(self.frames_after_eos);
        }
        match self.countdown.as_mut() {
            None => false,
            Some(0) => true,
            Some(countdown) => {
                *countdown -= 1;
                false
            }
        }
    }

    /// True once the model has signalled EOS, whether or not the tail has run out.
    pub fn saw_eos(&self) -> bool {
        self.countdown.is_some()
    }
}

/// One piece of the text, ready to prompt the model with.
///
/// Built by [`chunks`] or [`Chunk::new`], so the budgets always match the tokens.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Chunk {
    /// The piece of normalized text, before [`crate::tts_model::prepare_text_prompt`]; empty
    /// for a chunk built from tokens alone.
    pub text: String,
    pub tokens: Vec<u32>,
    /// The tail [`EosPolicy`] runs after the model signals the end.
    pub frames_after_eos: usize,
    /// The most frames this chunk may generate, from [`frame_budget`].
    pub frame_budget: usize,
}

impl Chunk {
    /// Prepare and tokenize one piece of already normalized text.
    pub fn new(text: String, tokenizer: &dyn Tokenizer, frame_rate: f64) -> Result<Self> {
        let (prepared, frames_after_eos) = prepare_text_prompt(&text);
        let tokens = tokenizer.encode(&prepared)?;
        let frame_budget = frame_budget(tokens.len(), frame_rate);
        Ok(Self { text, tokens, frames_after_eos, frame_budget })
    }

    /// The KV-cache length this chunk needs, from [`seq_budget`].
    pub fn seq_budget(&self) -> usize {
        seq_budget(self.tokens.len(), self.frame_budget)
    }
}

/// Split `text` into chunks the way every frontend does.
///
/// The whole input is normalized first, since that rewrites the characters the sentence
/// splitter looks for. Then whole sentences are grouped into chunks of up to `max_tokens`
/// tokens ([`split_into_best_sentences`]); a single sentence longer than that stays one chunk.
/// Each chunk is then prepared and tokenized on its own.
pub fn chunks(
    tokenizer: &dyn Tokenizer,
    text: &str,
    normalize: Normalize,
    max_tokens: usize,
    frame_rate: f64,
) -> Result<Vec<Chunk>> {
    let text = normalize.apply(text);
    let chunks = split_into_best_sentences(tokenizer, &text, Some(max_tokens))?
        .into_iter()
        .map(|text| Chunk::new(text, tokenizer, frame_rate))
        .collect::<Result<Vec<_>>>()?;
    if chunks.is_empty() {
        return Err(Error::invalid_argument("nothing to synthesize: the text is empty"));
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per word and per `.`, `!` or `?`, so sentence ends get their own ids as
    /// they do in the real tokenizer.
    #[derive(Default)]
    struct Words(std::cell::RefCell<Vec<String>>);

    impl Tokenizer for Words {
        fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
            let mut vocab = self.0.borrow_mut();
            let mut id = |piece: String| match vocab.iter().position(|v| *v == piece) {
                Some(i) => i as u32,
                None => {
                    vocab.push(piece);
                    vocab.len() as u32 - 1
                }
            };
            let mut ids = vec![];
            for word in text.split_whitespace() {
                let stem = word.trim_end_matches(['.', '!', '?']);
                if !stem.is_empty() {
                    ids.push(id(stem.to_string()));
                }
                ids.extend(word[stem.len()..].chars().map(|c| id(c.to_string())));
            }
            Ok(ids)
        }

        fn decode(&self, tokens: &[u32]) -> xn::Result<String> {
            let vocab = self.0.borrow();
            let mut text = String::new();
            for &t in tokens {
                let piece = &vocab[t as usize];
                if !text.is_empty() && !matches!(piece.as_str(), "." | "!" | "?") {
                    text.push(' ');
                }
                text.push_str(piece);
            }
            Ok(text)
        }
    }

    fn texts(chunks: &[Chunk]) -> Vec<&str> {
        chunks.iter().map(|c| c.text.as_str()).collect()
    }

    #[test]
    fn chunks_group_whole_sentences_up_to_the_limit() {
        let tok = Words::default();
        let text = "one two. three four. five six.";
        // Each sentence is three tokens: two of them fit in nine, all three do not fit in eight.
        let two = chunks(&tok, text, Normalize::OFF, 8, 12.5).unwrap();
        assert_eq!(texts(&two), ["One two. three four.", "five six."]);
        let one = chunks(&tok, text, Normalize::OFF, 9, 12.5).unwrap();
        assert_eq!(texts(&one), ["One two. three four. five six."]);
    }

    #[test]
    fn a_sentence_over_the_limit_stays_one_chunk() {
        // Never cut inside a sentence: that is what the original splitter does.
        let tok = Words::default();
        let chunks = chunks(&tok, "a b c d e f g h", Normalize::OFF, 3, 12.5).unwrap();
        assert_eq!(texts(&chunks), ["A b c d e f g h."]);
        assert!(chunks[0].tokens.len() > 3);
    }

    #[test]
    fn each_chunk_is_prepared_and_budgeted_on_its_own() {
        let tok = Words::default();
        let chunks =
            chunks(&tok, "short one. and now a longer second sentence", Normalize::OFF, 4, 12.5)
                .unwrap();
        assert_eq!(chunks.len(), 2);
        for chunk in &chunks {
            let (prepared, frames_after_eos) = prepare_text_prompt(&chunk.text);
            assert_eq!(chunk.tokens, tok.encode(&prepared).unwrap());
            assert_eq!(chunk.frames_after_eos, frames_after_eos);
            assert_eq!(chunk.frame_budget, frame_budget(chunk.tokens.len(), 12.5));
            assert_eq!(chunk.seq_budget(), seq_budget(chunk.tokens.len(), chunk.frame_budget));
        }
    }

    #[test]
    fn chunks_normalize_the_whole_text_first() {
        let tok = Words::default();
        let en = Normalize::for_lang(crate::preprocess::Lang::En);
        let chunks = chunks(&tok, "mail a@b now", en, 50, 12.5).unwrap();
        assert_eq!(texts(&chunks), ["Mail a at b now."]);
    }

    #[test]
    fn empty_text_is_an_error() {
        let err = chunks(&Words::default(), "   ", Normalize::OFF, 50, 12.5).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn frame_budget_matches_the_frontends() {
        // The literal every frontend carried: ((n / 3 + 2) * 12.5).ceil().
        for n in [0usize, 1, 7, 12, 50, 137] {
            let expected = ((n as f64 / 3.0 + 2.0) * 12.5).ceil() as usize;
            assert_eq!(frame_budget(n, 12.5), expected, "n = {n}");
        }
        assert_eq!(frame_budget(0, 12.5), 25);
        assert_eq!(frame_budget(50, 12.5), 234);
    }

    #[test]
    fn frame_budget_follows_the_frame_rate() {
        assert_eq!(frame_budget(30, 12.5), frame_budget(30, 25.0) / 2);
    }

    #[test]
    fn seq_budget_covers_tokens_frames_and_headroom() {
        assert_eq!(seq_budget(40, 234), 40 + 512 + 234);
    }

    /// Reference implementation, transcribed from `ptts.rs` before the
    /// refactor. `EosPolicy` must agree with it frame for frame.
    fn reference(eos_at: Option<usize>, frames_after_eos: usize, max_frames: usize) -> usize {
        let mut eos_countdown: Option<usize> = None;
        let mut emitted = 0;
        for step in 0..max_frames {
            emitted += 1;
            let is_eos = eos_at == Some(step);
            if is_eos && eos_countdown.is_none() {
                eos_countdown = Some(frames_after_eos);
            }
            if let Some(ref mut countdown) = eos_countdown {
                if *countdown == 0 {
                    break;
                }
                *countdown -= 1;
            }
        }
        emitted
    }

    fn under_test(eos_at: Option<usize>, frames_after_eos: usize, max_frames: usize) -> usize {
        let mut policy = EosPolicy::new(frames_after_eos);
        let mut emitted = 0;
        for step in 0..max_frames {
            emitted += 1;
            if policy.should_stop(eos_at == Some(step)) {
                break;
            }
        }
        emitted
    }

    #[test]
    fn eos_policy_matches_the_reference_loop() {
        for frames_after_eos in [0usize, 1, 3] {
            for eos_at in [None, Some(0), Some(1), Some(5), Some(19)] {
                for max_frames in [1usize, 6, 20] {
                    assert_eq!(
                        under_test(eos_at, frames_after_eos, max_frames),
                        reference(eos_at, frames_after_eos, max_frames),
                        "frames_after_eos = {frames_after_eos}, eos_at = {eos_at:?}, max = {max_frames}"
                    );
                }
            }
        }
    }

    #[test]
    fn eos_policy_emits_the_eos_frame_plus_the_tail() {
        // EOS on frame 5 with a 1-frame tail: frames 0..=6 are emitted.
        assert_eq!(under_test(Some(5), 1, 100), 7);
        // A 3-frame tail emits three more.
        assert_eq!(under_test(Some(5), 3, 100), 9);
        // No EOS: capped by max_frames.
        assert_eq!(under_test(None, 1, 100), 100);
    }

    #[test]
    fn eos_policy_reports_eos() {
        let mut policy = EosPolicy::new(1);
        assert!(!policy.saw_eos());
        policy.should_stop(false);
        assert!(!policy.saw_eos());
        policy.should_stop(true);
        assert!(policy.saw_eos());
    }
}
