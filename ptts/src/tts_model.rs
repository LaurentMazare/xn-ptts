//! [`TTSModel`], the flow LM and the Mimi decoder together, and [`TTSConfig`], a checkpoint's
//! `config.json`.
//!
//! It exposes generation one step at a time: prompt the state with a voice and text, step out
//! latents, decode them. [`crate::synth`] drives these steps on two threads. A caller with its
//! own event loop, such as the browser build, drives them directly.

use crate::conditioners::LUTConditioner;
use crate::flow_lm::{FlowLM, FlowLMConfig, FlowLMState};
use crate::mimi::{MimiConfig, MimiDecoder, MimiDecoderState, MimiEncoder};
use xn::nn::{Linear, var_builder::Path};
use xn::{BackendQ, Result, Tensor, Unquantized};

/// How training's fuser combined each conditioning. Only `sum` is read: it names the summed
/// conditionings, see [`SumLut`] and [`SumContinuous`].
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FuserConfig {
    pub sum: Vec<String>,
    pub streaming_sum: Vec<String>,
    pub prepend: Vec<String>,
    pub cross: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LutConditioner {
    pub n_bins: usize,
    pub dim: usize,
    pub possible_values: Vec<String>,
    pub tokenizer: String,
    /// What an unknown value maps to in training (`''` is the padding slot). Informational.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_value: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConditionerInnerConfig {
    Lut { lut: LutConditioner },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConditionerConfig {
    pub name: String,
    #[serde(flatten)]
    pub inner: ConditionerInnerConfig,
}

fn default_audio_prompt_min_duration() -> f32 {
    10.0
}

fn default_audio_prompt_max_duration() -> f32 {
    10.0
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelId {
    pub sig: String,
    pub epoch: usize,
}

/// Optional separate Mimi codec used only for speaker (voice-prompt) encoding.
/// When set, `MimiEnc` loads its encoder weights from `prefix` and uses
/// `mimi` as the codec config, instead of the main `TTSConfig.mimi`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SpeakerMimiConfig {
    /// Weight-name prefix for the speaker mimi (e.g. `"speaker_mimi"`).
    pub prefix: String,
    pub mimi: MimiConfig,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TTSConfig {
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
    pub lsd_decode_steps: usize,
    pub eos_threshold: f32,
    /// Read for its summed conditionings (`sum`), see [`SumLut`] and [`SumContinuous`].
    #[serde(default)]
    pub fuser: FuserConfig,
    /// The conditioners training had beyond the transcript and the voice prompt.
    #[serde(default)]
    pub conditioners: Vec<ConditionerConfig>,
    pub model_id: Option<ModelId>,
    /// Minimum allowed duration in seconds for an audio prompt passed to
    /// `get_state_for_audio`. If zero, an empty audio prompt is allowed, in
    /// which case the conditioned state skips the `prompt_audio` call entirely.
    #[serde(default = "default_audio_prompt_min_duration")]
    pub audio_prompt_min_duration: f32,
    /// Maximum allowed duration in seconds for an audio prompt. Frontends that
    /// trim long audio (e.g. the `ptts` example) should trim to this
    /// value rather than a hardcoded 10s.
    #[serde(default = "default_audio_prompt_max_duration")]
    pub audio_prompt_max_duration: f32,
    /// If true, the CFG null state is built without any audio prompting (the
    /// `prompt_audio` step is skipped on the null state). If false, the null
    /// state is prompted with the encoding of a zero waveform matching the
    /// real audio prompt's length, which preserves the historical behavior.
    #[serde(default)]
    pub cfg_null_audio_empty: bool,
    /// Optional, when set the speaker encoder loads from this prefix using
    /// this dedicated `MimiConfig` rather than the main `mimi` codec.
    #[serde(default)]
    pub speaker_mimi: Option<SpeakerMimiConfig>,
    /// Voices shipped with the checkpoint, registered by [`crate::synth::SynthBuilder`] in this
    /// order, the first being the default. Empty for a checkpoint that ships none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub voices: Vec<BundledVoice>,
}

/// A voice shipped with a checkpoint: values for its summed LUT conditionings, a prefix tensor
/// in the weights file, or both.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BundledVoice {
    pub name: String,
    /// Summed LUT values this voice selects, by conditioning name, e.g. `voice_name`.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub conditions: std::collections::BTreeMap<String, String>,
    /// Name of the voice's prefix tensor in the weights file, conventionally
    /// `voices.<name>.speaker_wavs` for speaker-Mimi latents (run through the speaker
    /// projection) or `voices.<name>.emb` for a projected embedding. `None` for no prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
}

impl TTSConfig {
    pub fn v202601() -> Self {
        Self {
            flow_lm: FlowLMConfig {
                d_model: 1024,
                num_heads: 16,
                num_layers: 6,
                dim_feedforward: 4096,
                max_period: 10000.0,
                n_bins: 4000,
                lut_dim: 1024,
                flow_dim: 512,
                flow_depth: 6,
                ldim: 32,
            },
            mimi: MimiConfig {
                channels: 1,
                sample_rate: 24000,
                frame_rate: 12.5,
                dimension: 512,
                quantizer_dimension: 32,
                quantizer_output_dimension: 512,
                n_filters: 64,
                n_residual_layers: 1,
                ratios: vec![6, 5, 4],
                kernel_size: 7,
                last_kernel_size: 3,
                residual_kernel_size: 3,
                dilation_base: 2,
                compress: 2,
                transformer_d_model: 512,
                transformer_num_heads: 8,
                transformer_num_layers: 2,
                transformer_layer_scale: 0.01,
                transformer_context: 250,
                transformer_max_period: 10000.0,
                transformer_dim_feedforward: 2048,
                downsample_channel_wise: false,
            },
            lsd_decode_steps: 1,
            eos_threshold: -4.0,
            fuser: FuserConfig::default(),
            conditioners: vec![],
            model_id: None,
            audio_prompt_min_duration: 10.0,
            audio_prompt_max_duration: 10.0,
            cfg_null_audio_empty: false,
            speaker_mimi: None,
            voices: vec![],
        }
    }

    pub fn model_ext(&self) -> Option<String> {
        self.model_id.as_ref().map(|id| format!("{}@{}", id.sig, id.epoch))
    }

    /// Returns the `MimiConfig` used to encode the voice prompt — the
    /// dedicated `speaker_mimi.mimi` if set, otherwise the main `mimi`.
    pub fn speaker_mimi_cfg(&self) -> &MimiConfig {
        match self.speaker_mimi.as_ref() {
            Some(s) => &s.mimi,
            None => &self.mimi,
        }
    }

    /// Returns the weight-name prefix used to load the speaker encoder.
    pub fn speaker_mimi_prefix(&self) -> &str {
        match self.speaker_mimi.as_ref() {
            Some(s) => s.prefix.as_str(),
            None => "mimi",
        }
    }
}

pub struct TTSModel<Q: BackendQ> {
    pub flow_lm: FlowLM<Q>,
    pub mimi: MimiDecoder<Unquantized<f32, Q::B>>,
    speaker_proj: Option<Linear<f32, Q::B>>,
    sum_luts: Vec<SumLut<Q>>,
    lsd_decode_steps: usize,
    eos_threshold: f32,
}

/// A LUT conditioning summed into every audio frame whose value is chosen per state, e.g. a
/// fixed voice (audium's `config/conditioner/tts_voice_lut.yaml`). These are the `fuser.sum`
/// entries of `conditioners` other than `num_speakers`, which keeps its fixed-value path.
pub struct SumLut<Q: BackendQ> {
    pub name: String,
    pub values: Vec<String>,
    cond: LUTConditioner<Q::T, Q::B>,
}

/// Refuse a summed LUT whose ids this crate would get wrong. Training's `noop` and `whitespace`
/// tokenizers give a known value its position in `possible_values` and put padding at `n_bins`
/// (audiocraft `conditioners/text.py`, `_WordToToken`); `whitespace` also splits a value on
/// spaces into several summed ids, which [`lut_id`] does not do.
fn check_sum_lut(name: &str, lut: &LutConditioner) -> Result<()> {
    match lut.tokenizer.as_str() {
        "noop" => {}
        "whitespace" => {
            if let Some(v) = lut.possible_values.iter().find(|v| v.split_whitespace().count() != 1)
            {
                xn::bail!("summed LUT '{name}': value '{v}' is not a single whitespace token")
            }
        }
        other => xn::bail!("summed LUT '{name}': unsupported tokenizer '{other}'"),
    }
    if lut.possible_values.len() > lut.n_bins {
        xn::bail!(
            "summed LUT '{name}' lists {} values but has only {} bins",
            lut.possible_values.len(),
            lut.n_bins
        )
    }
    Ok(())
}

/// The embedding row for `value` of a summed LUT: its position in `possible_values`.
fn lut_id(name: &str, values: &[String], value: &str) -> Result<u32> {
    match values.iter().position(|v| v == value) {
        Some(index) => Ok(index as u32),
        None => xn::bail!("unknown value '{value}' for '{name}', expected one of {values:?}"),
    }
}

#[derive(Clone, Debug)]
pub struct TTSState<Q: BackendQ> {
    pub flow_lm_state: FlowLMState<Q>,
}

impl<Q: BackendQ> TTSModel<Q> {
    pub fn load(
        vb: &Path<Q::B>,
        tokenizer: Box<dyn crate::Tokenizer + Send + Sync>,
        cfg: &TTSConfig,
    ) -> Result<Self> {
        let flow_lm = FlowLM::load(&vb.pp("flow_lm"), tokenizer, &cfg.flow_lm)?;
        let mimi = MimiDecoder::load(&vb.pp("mimi"), &cfg.mimi)?;
        let speaker_proj = crate::loader::load_speaker_proj(vb, cfg)?;
        let mut sum_luts = vec![];
        for cond in cfg.conditioners.iter() {
            if cond.name == "num_speakers" || !cfg.fuser.sum.contains(&cond.name) {
                continue;
            }
            let ConditionerInnerConfig::Lut { lut } = &cond.inner;
            check_sum_lut(&cond.name, lut)?;
            let vb = vb.pp(format!("flow_lm.condition_provider.conditioners.{}", cond.name));
            let lut_cond =
                LUTConditioner::load(&vb, lut.n_bins, None, lut.dim, cfg.flow_lm.d_model)?;
            sum_luts.push(SumLut {
                name: cond.name.clone(),
                values: lut.possible_values.clone(),
                cond: lut_cond,
            });
        }
        if sum_luts.len() > 1 {
            let names: Vec<_> = sum_luts.iter().map(|lut| lut.name.as_str()).collect();
            tracing::warn!(
                "several summed LUT conditionings {names:?}: `Synth` registers no voices for \
                 them, pick values with `TTSModel::set_sum_conditions`"
            );
        }

        Ok(Self {
            flow_lm,
            mimi,
            speaker_proj,
            sum_luts,
            lsd_decode_steps: cfg.lsd_decode_steps,
            eos_threshold: cfg.eos_threshold,
        })
    }

    pub fn with_eos_threshold(mut self, eos_threshold: f32) -> Self {
        self.eos_threshold = eos_threshold;
        self
    }

    pub fn sample_rate(&self) -> usize {
        self.mimi.sample_rate
    }

    /// The checkpoint's speaker projection, when it has one. Voice files holding stored
    /// speaker latents go through it in [`crate::loader::load_voice_emb`].
    pub fn speaker_proj(&self) -> Option<&Linear<f32, Q::B>> {
        self.speaker_proj.as_ref()
    }

    /// The per-state summed LUT conditionings, see [`SumLut`].
    pub fn sum_luts(&self) -> &[SumLut<Q>] {
        &self.sum_luts
    }

    /// Choose the value of every per-state summed LUT for `state`. A name mapped to
    /// `Some(value)` embeds that value; a name that is absent or mapped to `None` gets what
    /// training feeds for a dropped attribute (the learnt padding, or nothing when the LUT has
    /// none), which is what a CFG null state or a voice with no LUT value needs. Naming a LUT
    /// the model does not have, or a value it does not know, is an error rather than a silent
    /// fall back to padding.
    pub fn set_sum_conditions(
        &self,
        state: &mut TTSState<Q>,
        values: &std::collections::HashMap<String, Option<String>>,
    ) -> Result<()> {
        state.flow_lm_state.extra_sum = self.sum_conditions(values)?;
        Ok(())
    }

    /// The `[1, 1, d_model]` term [`Self::set_sum_conditions`] stores for `values`, or `None`
    /// when it adds nothing; it fails as that does on a name or value the model does not have.
    pub fn sum_conditions(
        &self,
        values: &std::collections::HashMap<String, Option<String>>,
    ) -> Result<Option<Tensor<Q::T, Q::B>>> {
        for name in values.keys() {
            if !self.sum_luts.iter().any(|lut| &lut.name == name) {
                let known: Vec<_> = self.sum_luts.iter().map(|lut| lut.name.as_str()).collect();
                xn::bail!("the model has no summed LUT conditioning '{name}', it has {known:?}")
            }
        }
        let mut total: Option<Tensor<Q::T, Q::B>> = None;
        for lut in self.sum_luts.iter() {
            // A dropped attribute: training multiplies its embedding by the zero mask and adds
            // the learnt padding if there is one, so without one it contributes nothing. The
            // padding is taken as is, never looked up by id (`embed_tokens` refuses its row).
            let emb = match values.get(&lut.name).and_then(|v| v.as_deref()) {
                Some(value) => lut.cond.embed_tokens(&[lut_id(&lut.name, &lut.values, value)?])?,
                None => match lut.cond.learnt_padding() {
                    Some(padding) => padding.clone(),
                    None => continue,
                },
            };
            total = Some(match total {
                Some(total) => total.broadcast_add(&emb)?,
                None => emb,
            });
        }
        Ok(total)
    }

    /// Initialize flow LM state with the given sequence length budget. Every per-state summed
    /// LUT starts as a dropped attribute (see [`Self::set_sum_conditions`], which picks values),
    /// so a speaker-prompted voice on a LUT model still sees what training fed it.
    pub fn init_flow_lm_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<TTSState<Q>> {
        let mut state =
            TTSState { flow_lm_state: self.flow_lm.init_state(batch_size, sequence_length)? };
        self.set_sum_conditions(&mut state, &Default::default())?;
        Ok(state)
    }

    /// Run flow LM step with text tokens. Increments state.
    pub fn prompt_text(&self, state: &mut TTSState<Q>, text_tokens: &[u32]) -> Result<()> {
        let text_embeddings = self.flow_lm.conditioner.embed_tokens(text_tokens)?;
        let dev = text_embeddings.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        self.run_backbone_and_increment(state, &text_embeddings, &empty_latents)?;
        Ok(())
    }

    pub fn prompt_text_null(&self, state: &mut TTSState<Q>) -> Result<()> {
        let empty_text = match self.flow_lm.conditioner.learnt_padding() {
            None => xn::bail!("Model does not support null text prompt"),
            Some(p) => p,
        };
        let dev = empty_text.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        self.run_backbone_and_increment(state, empty_text, &empty_latents)?;
        Ok(())
    }

    /// Run flow LM step with audio conditioning. Increments state.
    pub fn prompt_audio(
        &self,
        state: &mut TTSState<Q>,
        audio_conditioning: &Tensor<Q::T, Q::B>,
    ) -> Result<()> {
        // Nothing to prompt, e.g. a model conditioned on a summed voice with no voice prefix.
        // Running the backbone on zero frames fails on CUDA (CUDA_ERROR_INVALID_VALUE).
        if audio_conditioning.dims3()?.1 == 0 {
            return Ok(());
        }
        let dev = audio_conditioning.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), dev)?;
        let text_embeddings = Tensor::cat(&[&self.empty_text()?, audio_conditioning], 1)?;
        self.run_backbone_and_increment(state, &text_embeddings, &empty_latents)?;
        Ok(())
    }

    /// One autoregressive step, returning the latent and the *raw* eos logit.
    /// Reads nothing back, so this is the entry point a browser can drive.
    #[allow(clippy::type_complexity)]
    pub fn generate_step_parts(
        &self,
        state: &mut TTSState<Q>,
        input: crate::flow_lm::StepInput<'_, Q>,
        rng: &mut impl crate::flow_lm::Rng,
    ) -> Result<(Tensor<Q::T, Q::B>, Tensor<Q::T, Q::B>)> {
        self.flow_lm.sample_next_latent_parts(
            input,
            &self.empty_text()?,
            &mut state.flow_lm_state,
            self.lsd_decode_steps,
            rng,
        )
    }

    /// Run one autoregressive generation step.
    /// Returns (next_latent [B, 1, ldim], is_eos).
    #[allow(clippy::type_complexity)]
    pub fn generate_step(
        &self,
        state: &mut TTSState<Q>,
        input: crate::flow_lm::StepInput<'_, Q>,
        rng: &mut impl crate::flow_lm::Rng,
    ) -> Result<(Tensor<Q::T, Q::B>, bool)> {
        let (latent, eos_logit) = self.generate_step_parts(state, input, rng)?;
        Ok((latent, self.eos_from_logit(&eos_logit.to_vec()?)))
    }

    /// Threshold an eos logit the caller has brought back to the host.
    pub fn eos_from_logit(&self, eos_val: &[Q::T]) -> bool {
        crate::flow_lm::FlowLM::<Q>::eos_from_logit(eos_val, self.eos_threshold)
    }

    /// As [`Self::generate_step`], with classifier-free guidance. Reads the eos
    /// logit back every step, so unlike [`Self::generate_step_parts`] it is not
    /// browser-safe.
    #[allow(clippy::type_complexity)]
    pub fn generate_step_cfg(
        &self,
        state: &mut TTSState<Q>,
        null_state: &mut TTSState<Q>,
        cfg_coef: f32,
        input: crate::flow_lm::StepInput<'_, Q>,
        rng: &mut impl crate::flow_lm::Rng,
    ) -> Result<(Tensor<Q::T, Q::B>, bool)> {
        let (latent, is_eos) = self.flow_lm.sample_next_latent_cfg(
            input,
            &self.empty_text()?,
            &mut state.flow_lm_state,
            &mut null_state.flow_lm_state,
            cfg_coef,
            self.lsd_decode_steps,
            rng,
            self.eos_threshold,
        )?;

        Ok((latent, is_eos))
    }

    /// Decode latent to audio using mimi (streaming).
    pub fn decode_latent(
        &self,
        latent: &Tensor<Q::T, Q::B>,
        mimi_state: &mut MimiDecoderState<f32, Q::B>,
    ) -> Result<Tensor<f32, Q::B>> {
        let denorm =
            latent.broadcast_mul(&self.flow_lm.emb_std)?.broadcast_add(&self.flow_lm.emb_mean)?;

        // [B, T, C] -> [B, C, T]
        let transposed = denorm.transpose(1, 2)?.contiguous()?;
        // Convert from Q::T to f32 for mimi
        let f32_transposed = transposed.to()?;
        let quantized = self.mimi.quantizer.forward(&f32_transposed)?;
        self.mimi.decode_from_latent_step(&quantized, mimi_state)
    }

    /// Initialize mimi streaming state.
    ///
    /// The decoder transformer's context window is fixed at load time from
    /// `MimiConfig::transformer_context`; nothing about this state is sized per call, so unlike
    /// [`Self::init_flow_lm_state`] there is no budget to pass.
    pub fn init_mimi_state(&self, batch_size: usize) -> Result<MimiDecoderState<f32, Q::B>> {
        // `sequence_length` reaches only the flow-LM attention kind, which a Mimi decoder has
        // none of, so any value here is discarded.
        self.mimi.init_state(batch_size, 0)
    }

    fn run_backbone_and_increment(
        &self,
        state: &mut TTSState<Q>,
        text_embeddings: &Tensor<Q::T, Q::B>,
        backbone_input_latents: &Tensor<Q::T, Q::B>,
    ) -> Result<()> {
        let input = self.flow_lm.input_linear.forward(backbone_input_latents)?;
        let input = Tensor::cat(&[text_embeddings, &input], 1)?;
        let _out =
            self.flow_lm.transformer.forward(&input, &mut state.flow_lm_state.transformer_state)?;
        Ok(())
    }

    pub fn device(&self) -> &Q::B {
        self.flow_lm.input_linear.device()
    }

    /// The empty text prefix a step with no new tokens is conditioned on.
    fn empty_text(&self) -> Result<Tensor<Q::T, Q::B>> {
        Tensor::zeros((1, 0, self.flow_lm.conditioner.dim), self.device())
    }
}

/// The speaker encoder: speaker-Mimi latents from audio, projected to the flow LM's width.
pub struct MimiEnc<Q: BackendQ> {
    speaker_proj: Linear<Q::T, Q::B>,
    mimi: MimiEncoder<Unquantized<f32, Q::B>>,
}

impl<Q: BackendQ> MimiEnc<Q> {
    /// Fails when the checkpoint has no speaker projection: the encoder's latents are not a
    /// voice embedding on their own, and conditioning on them makes the model babble or stop
    /// at once with no other symptom, so a missing weight is better caught here.
    pub fn load(vb: &Path<Q::B>, cfg: &TTSConfig) -> Result<Self> {
        let mimi_cfg = cfg.speaker_mimi_cfg();
        let mimi = MimiEncoder::load(&vb.pp(cfg.speaker_mimi_prefix()), mimi_cfg)?;
        if !vb.contains(crate::loader::SPEAKER_PROJ_WEIGHT) {
            xn::bail!(
                "checkpoint has a speaker encoder under `{}` but no speaker projection \
                 (`{}`), which voice cloning needs. A GGUF written by an older `quantize \
                 --no-mimi-encoder` dropped it: regenerate the GGUF from the safetensors \
                 checkpoint.",
                cfg.speaker_mimi_prefix(),
                crate::loader::SPEAKER_PROJ_WEIGHT
            )
        }
        let weights = vb.tensor(
            crate::loader::SPEAKER_PROJ_WEIGHT,
            (cfg.flow_lm.d_model, mimi_cfg.dimension),
        )?;
        Ok(Self { speaker_proj: Linear::new(weights), mimi })
    }

    /// Encode audio for voice conditioning. Returns [1, T', dim].
    pub fn encode_audio(&self, audio: &Tensor<Q::T, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        let f32_audio = audio.to::<f32>()?;
        let encoded = self.mimi.encode_to_latent(&f32_audio)?;
        // [B, C, T] -> [B, T, C]
        let latents = encoded.transpose(1, 2)?.contiguous()?.to::<Q::T>()?;
        self.speaker_proj.forward(&latents)
    }
}

pub const MAX_TOKENS_PER_CHUNK: usize = 50;

/// Split text into sentence-aligned chunks that fit within a token budget.
///
/// This mirrors the Python `split_into_best_sentences` function: it prepares the text,
/// tokenizes it, finds sentence boundaries (after `.`, `!`, `...`, `?` tokens), then
/// greedily groups sentences into chunks of at most `max_tokens` tokens each.
pub fn split_into_best_sentences(
    tokenizer: &dyn crate::Tokenizer,
    text: &str,
    max_tokens: Option<usize>,
) -> Result<Vec<String>> {
    let max_tokens = max_tokens.unwrap_or(MAX_TOKENS_PER_CHUNK);
    let (prepared, _) = prepare_text_prompt(text);
    let prepared = prepared.trim().to_string();
    let tokens = tokenizer.encode(&prepared)?;

    // Get end-of-sentence token ids by tokenizing ".!...?" and skipping the first token
    // (the first token includes the leading space marker from sentencepiece).
    let eos_marker_tokens = tokenizer.encode(".!...?")?;
    let eos_tokens =
        if eos_marker_tokens.len() > 1 { &eos_marker_tokens[1..] } else { &eos_marker_tokens[..] };

    // Find sentence boundary indices: positions where a non-EOS token follows one or more EOS tokens.
    let mut sentence_boundaries = vec![0usize];
    let mut prev_was_eos = false;

    for (idx, &token) in tokens.iter().enumerate() {
        if eos_tokens.contains(&token) {
            prev_was_eos = true;
        } else {
            if prev_was_eos {
                sentence_boundaries.push(idx);
            }
            prev_was_eos = false;
        }
    }
    sentence_boundaries.push(tokens.len());

    // Build (token_count, sentence_text) pairs by decoding each token sub-range.
    let mut sentences = Vec::new();
    for window in sentence_boundaries.windows(2) {
        let (start, end) = (window[0], window[1]);
        let text = tokenizer.decode(&tokens[start..end])?;
        sentences.push((end - start, text));
    }

    // Greedily group sentences into chunks that stay under max_tokens.
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_token_count = 0;

    for (nb_tokens, sentence) in sentences {
        if current_chunk.is_empty() {
            current_chunk = sentence;
            current_token_count = nb_tokens;
            continue;
        }

        if current_token_count + nb_tokens > max_tokens {
            chunks.push(current_chunk.trim().to_string());
            current_chunk = sentence;
            current_token_count = nb_tokens;
        } else {
            current_chunk.push(' ');
            current_chunk.push_str(&sentence);
            current_token_count += nb_tokens;
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    Ok(chunks)
}

/// Prepare text for generation: capitalize, add punctuation, pad short text.
pub fn prepare_text_prompt(text: &str) -> (String, usize) {
    let text = text.trim().to_string();
    if text.is_empty() {
        return (text, 3);
    }
    let text = text.replace(['\n', '\r'], " ");
    let mut text: String = text.split_whitespace().collect::<Vec<_>>().join(" ");

    let number_of_words = text.split_whitespace().count();
    let frames_after_eos = if number_of_words <= 4 { 3 } else { 1 };
    let mut chars = text.chars();
    if let Some(first) = chars.next() {
        text = first.to_uppercase().to_string() + chars.as_str();
    }
    if text.chars().last().is_some_and(|c| c.is_alphanumeric()) {
        text.push('.');
    }
    (text, frames_after_eos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_not_padded() {
        // pocket-tts prepended 8 spaces to texts of fewer than 5 words; Phonon models
        // never see those spaces, and short texts go wrong with them.
        assert_eq!(prepare_text_prompt("not a thing"), ("Not a thing.".to_string(), 3));
        assert_eq!(
            prepare_text_prompt("one two three four five"),
            ("One two three four five.".to_string(), 1)
        );
    }

    #[test]
    fn prepare_text_prompt_edge_cases() {
        let cases: &[(&str, &str, usize)] = &[
            ("", "", 3),
            ("  \n ", "", 3),
            // Only a trailing letter or digit gets a full stop; other punctuation is kept.
            ("is it?", "Is it?", 3),
            ("hello world!", "Hello world!", 3),
            ("call me at 5", "Call me at 5.", 3),
            // Line breaks, CRLF included, and runs of spaces become one space.
            ("one\r\ntwo   three\nfour five", "One two three four five.", 1),
            ("éclair au chocolat", "Éclair au chocolat.", 3),
        ];
        for &(input, text, frames) in cases {
            assert_eq!(prepare_text_prompt(input), (text.to_string(), frames), "{input:?}");
        }
    }

    #[test]
    fn config_round_trips_and_fills_defaults() {
        let cfg = TTSConfig::v202601();
        let json = serde_json::to_value(&cfg).unwrap();
        let back: TTSConfig = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(&back).unwrap(), json);

        // A config written before these fields existed still loads, with their defaults.
        let mut old = json;
        let fields = old.as_object_mut().unwrap();
        for key in [
            "audio_prompt_min_duration",
            "audio_prompt_max_duration",
            "cfg_null_audio_empty",
            "speaker_mimi",
        ] {
            assert!(fields.remove(key).is_some(), "{key} is no longer in the config");
        }
        let old: TTSConfig = serde_json::from_value(old).unwrap();
        assert_eq!(old.audio_prompt_min_duration, default_audio_prompt_min_duration());
        assert_eq!(old.audio_prompt_max_duration, default_audio_prompt_max_duration());
        assert!(!old.cfg_null_audio_empty);
        assert!(old.speaker_mimi.is_none());
    }

    #[test]
    fn config_ignores_keys_it_no_longer_reads() {
        // Published config.json files still carry these, and have to keep loading.
        let mut json = serde_json::to_value(TTSConfig::v202601()).unwrap();
        let fields = json.as_object_mut().unwrap();
        fields.insert("temp".into(), serde_json::json!(0.7));
        let id = serde_json::json!({"sig": "abc", "epoch": 1, "mimi_sig": "def", "mimi_epoch": 2});
        fields.insert("model_id".into(), id);
        let cfg: TTSConfig = serde_json::from_value(json).unwrap();
        assert_eq!(cfg.model_ext().as_deref(), Some("abc@1"));
    }

    fn lut(tokenizer: &str, n_bins: usize, values: &[&str]) -> LutConditioner {
        LutConditioner {
            n_bins,
            dim: 4,
            possible_values: values.iter().map(|v| v.to_string()).collect(),
            tokenizer: tokenizer.to_string(),
            default_value: Some(String::new()),
        }
    }

    #[test]
    fn a_value_is_its_position() {
        let values = ["a".to_string(), "b".to_string()];
        assert_eq!(lut_id("v", &values, "a").unwrap(), 0);
        assert_eq!(lut_id("v", &values, "b").unwrap(), 1);
    }

    #[test]
    fn an_unknown_value_is_an_error_not_padding() {
        let values = ["a".to_string()];
        let err = lut_id("v", &values, "z").unwrap_err().to_string();
        assert!(err.contains("unknown value 'z'"), "{err}");
    }

    #[test]
    fn bundled_voices_are_read_from_the_config() {
        let mut cfg = serde_json::to_value(TTSConfig::v202601()).unwrap();
        assert!(cfg.get("voices").is_none(), "an empty list is not written");
        cfg["voices"] = serde_json::json!([
            {"name": "alba", "conditions": {"voice_name": "a@300"},
             "prefix": "voices.alba.speaker_wavs"},
            {"name": "lut-only", "conditions": {"voice_name": "b@300"}},
        ]);
        let cfg: TTSConfig = serde_json::from_value(cfg).unwrap();
        assert_eq!(cfg.voices.len(), 2);
        assert_eq!(cfg.voices[0].prefix.as_deref(), Some("voices.alba.speaker_wavs"));
        assert_eq!(cfg.voices[1].conditions["voice_name"], "b@300");
        assert_eq!(cfg.voices[1].prefix, None);
    }

    #[test]
    fn a_lut_is_checked_at_load() {
        check_sum_lut("v", &lut("noop", 2, &["a b", "c"])).unwrap();
        check_sum_lut("v", &lut("whitespace", 2, &["a", "c"])).unwrap();
        // More values than bins would reach the padding row or past the table.
        assert!(check_sum_lut("v", &lut("noop", 1, &["a", "b"])).is_err());
        assert!(check_sum_lut("v", &lut("whitespace", 2, &["a b"])).is_err());
        assert!(check_sum_lut("v", &lut("sentencepiece", 2, &["a"])).is_err());
    }
}
