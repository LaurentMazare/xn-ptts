//! The Graphon ASR language model: a causal transformer that consumes one Mimi
//! frame (32 audio codes) plus the previously emitted text token per step and
//! predicts the next text token.

use crate::conditioners::LUTConditioner;
use crate::rms_norm::RmsNorm;
use crate::rope::RotaryEmbedding;
use crate::transformer::{CachedMultiheadAttention, KvCache};
use xn::nn::{Embedding, Linear, var_builder::Path};
use xn::{Backend, BackendQ, Result, Tensor, WithDTypeF};

/// The norm's `alpha` gain is trained against this epsilon.
const NORM_EPS: f32 = 1e-8;

// ---- Config ----

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelId {
    pub sig: String,
    pub epoch: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LutConfig {
    pub n_bins: usize,
    pub dim: usize,
    #[serde(default)]
    pub possible_values: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConditionerConfig {
    Lut {
        lut: LutConfig,
    },
    /// Conditioner kinds this model does not implement. The ASR checkpoint only
    /// ships a LUT, but keeping the catch-all means a sibling checkpoint still
    /// parses instead of failing at `serde_json::from_str`.
    #[serde(other)]
    Unsupported,
}

/// The subset of the checkpoint's `config.json` this model needs. Unknown
/// fields (the depformer block, delays, ...) are ignored.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub card: usize,
    pub n_q: usize,
    pub dim: usize,
    pub text_card: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub hidden_scale: f64,
    pub context: usize,
    pub max_period: f64,
    pub asr_delay_in_tokens: usize,
    #[serde(default)]
    pub extra_heads_num_heads: Option<usize>,
    #[serde(default)]
    pub extra_heads_dim: Option<usize>,
    #[serde(default)]
    pub conditioners: std::collections::HashMap<String, ConditionerConfig>,
    #[serde(default)]
    pub vad_horizons: Vec<f64>,
    #[serde(default)]
    pub model_id: Option<ModelId>,
}

impl Config {
    fn dim_feedforward(&self) -> usize {
        (self.dim as f64 * self.hidden_scale) as usize
    }

    /// Width of each half of the gated MLP.
    fn gating_hidden(&self) -> usize {
        let ff = self.dim_feedforward();
        if ff == 4 * self.dim { 11 * self.dim / 4 } else { 2 * ff / 3 }
    }

    fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }

    pub fn audio_vocab_size(&self) -> usize {
        self.card + 1
    }

    pub fn text_in_vocab_size(&self) -> usize {
        self.text_card + 1
    }

    fn extra_heads(&self) -> Option<(usize, usize)> {
        match (self.extra_heads_num_heads, self.extra_heads_dim) {
            (Some(n), Some(dim)) if n > 0 => Some((n, dim)),
            _ => None,
        }
    }

    pub fn model_ext(&self) -> Option<String> {
        self.model_id.as_ref().map(|id| format!("{}@{}", id.sig, id.epoch))
    }
}

// ---- Layers ----

/// SwiGLU feed-forward: `linear_in` emits both halves, gate first.
struct Mlp<Q: BackendQ> {
    linear_in: Q::LinearQ,
    linear_out: Q::LinearQ,
    hidden: usize,
}

impl<Q: BackendQ> Mlp<Q> {
    fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        let hidden = cfg.gating_hidden();
        let vb = vb.pp("gating");
        let linear_in = Q::linear_load(vb.pp("linear_in"), cfg.dim, 2 * hidden)?;
        let linear_out = Q::linear_load(vb.pp("linear_out"), hidden, cfg.dim)?;
        Ok(Self { linear_in, linear_out, hidden })
    }

    fn forward(&self, xs: &Tensor<Q::T, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        use xn::ModuleT;
        let xs = self.linear_in.forward(xs)?;
        let gate = xs.narrow(2, 0..self.hidden)?.contiguous()?;
        let up = xs.narrow(2, self.hidden..2 * self.hidden)?.contiguous()?;
        self.linear_out.forward(&gate.silu()?.mul(&up)?)
    }
}

struct Layer<Q: BackendQ> {
    norm1: RmsNorm<Q::T, Q::B>,
    self_attn: CachedMultiheadAttention<Q>,
    norm2: RmsNorm<Q::T, Q::B>,
    mlp: Mlp<Q>,
}

impl<Q: BackendQ> Layer<Q> {
    fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        Ok(Self {
            norm1: RmsNorm::load(&vb.pp("norm1"), cfg.dim, NORM_EPS)?,
            self_attn: CachedMultiheadAttention::load(
                &vb.pp("self_attn"),
                cfg.dim,
                cfg.num_heads,
                cfg.context,
            )?,
            norm2: RmsNorm::load(&vb.pp("norm2"), cfg.dim, NORM_EPS)?,
            mlp: Mlp::load(vb, cfg)?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor<Q::T, Q::B>,
        rope: &RotaryEmbedding<Q::T, Q::B>,
        cache: &mut KvCache<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        // One query token per step and every cached key precedes it, so the
        // causal mask is all-zero and can be skipped entirely.
        let residual = self.self_attn.forward(&self.norm1.forward(xs)?, rope, cache, None)?;
        let xs = xs.add(&residual)?;
        let residual = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        xs.add(&residual)
    }
}

// ---- Model ----

pub struct LmState<T: WithDTypeF, B: Backend> {
    caches: Vec<KvCache<T, B>>,
    offset: usize,
}

pub struct LmModel<Q: BackendQ> {
    layers: Vec<Layer<Q>>,
    text_emb: Embedding<Q::T, Q::B>,
    audio_embs: Vec<Embedding<Q::T, Q::B>>,
    out_norm: RmsNorm<Q::T, Q::B>,
    text_linear: Q::LinearQ,
    extra_heads: Vec<Linear<Q::T, Q::B>>,
    /// Keyed by conditioner name, e.g. `languages_in_segment`.
    pub conditioners: std::collections::HashMap<String, LUTConditioner<Q::T, Q::B>>,
    head_dim: usize,
    max_period: f32,
    audio_vocab_size: usize,
    text_in_vocab_size: usize,
}

impl<Q: BackendQ> LmModel<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        let vb_l = vb.pp("transformer").pp("layers");
        let layers = (0..cfg.num_layers)
            .map(|i| Layer::load(&vb_l.pp(i), cfg))
            .collect::<Result<Vec<_>>>()?;

        let text_emb = Embedding::load(vb.pp("text_emb"), cfg.text_in_vocab_size(), cfg.dim)?;
        let vb_e = vb.pp("emb");
        let audio_embs = (0..cfg.n_q)
            .map(|i| Embedding::load(vb_e.pp(i), cfg.audio_vocab_size(), cfg.dim))
            .collect::<Result<Vec<_>>>()?;

        let out_norm = RmsNorm::load(&vb.pp("out_norm"), cfg.dim, NORM_EPS)?;
        let text_linear = Q::linear_load(vb.pp("text_linear"), cfg.dim, cfg.text_card)?;

        let mut extra_heads = vec![];
        if let Some((num_heads, dim)) = cfg.extra_heads() {
            let vb_h = vb.pp("extra_heads");
            for i in 0..num_heads {
                extra_heads.push(Linear::load(vb_h.pp(i), cfg.dim, dim)?);
            }
        }

        let vb_c = vb.pp("condition_provider").pp("conditioners");
        let mut conditioners = std::collections::HashMap::new();
        for (name, cond) in cfg.conditioners.iter() {
            match cond {
                ConditionerConfig::Lut { lut } => {
                    let conditioner =
                        LUTConditioner::load(&vb_c.pp(name), lut.n_bins, None, lut.dim, cfg.dim)?
                            .with_possible_values(lut.possible_values.clone());
                    conditioners.insert(name.clone(), conditioner);
                }
                ConditionerConfig::Unsupported => {
                    tracing::warn!(name, "ignoring unsupported conditioner")
                }
            }
        }

        Ok(Self {
            layers,
            text_emb,
            audio_embs,
            out_norm,
            text_linear,
            extra_heads,
            conditioners,
            head_dim: cfg.head_dim(),
            max_period: cfg.max_period as f32,
            audio_vocab_size: cfg.audio_vocab_size(),
            text_in_vocab_size: cfg.text_in_vocab_size(),
        })
    }

    pub fn init_state(&self) -> Result<LmState<Q::T, Q::B>> {
        let caches =
            self.layers.iter().map(|l| l.self_attn.init_state()).collect::<Result<Vec<_>>>()?;
        Ok(LmState { caches, offset: 0 })
    }

    /// Codebook entry that stands in for "no audio yet".
    pub fn audio_pad_token(&self) -> u32 {
        self.audio_vocab_size as u32 - 1
    }

    /// Text token fed at the very first step.
    pub fn text_start_token(&self) -> u32 {
        self.text_in_vocab_size as u32 - 1
    }

    pub fn device(&self) -> &Q::B {
        self.text_emb.device()
    }

    /// One decoding step. Returns the text logits `(1, 1, text_card)` and the
    /// transformer output `(1, 1, dim)` that the extra heads read.
    #[allow(clippy::type_complexity)]
    pub fn step(
        &self,
        state: &mut LmState<Q::T, Q::B>,
        text_token: u32,
        audio_tokens: &[u32],
        condition: Option<&Tensor<Q::T, Q::B>>,
    ) -> Result<(Tensor<Q::T, Q::B>, Tensor<Q::T, Q::B>)> {
        use xn::ModuleT;
        if audio_tokens.len() != self.audio_embs.len() {
            xn::bail!("expected {} audio tokens, got {}", self.audio_embs.len(), audio_tokens.len())
        }
        let dev = self.device();
        let ids = Tensor::from_vec(vec![text_token as i64], 1, dev)?;
        let mut xs = self.text_emb.forward(&ids)?.unsqueeze(1)?;
        for (emb, &token) in self.audio_embs.iter().zip(audio_tokens.iter()) {
            let ids = Tensor::from_vec(vec![token as i64], 1, dev)?;
            xs = xs.add(&emb.forward(&ids)?.unsqueeze(1)?)?;
        }
        if let Some(condition) = condition {
            xs = xs.add(condition)?;
        }

        let rope = RotaryEmbedding::new(self.head_dim, state.offset, 1, self.max_period, dev)?;
        for (layer, cache) in self.layers.iter().zip(state.caches.iter_mut()) {
            xs = layer.forward(&xs, &rope, cache)?;
        }
        state.offset += 1;

        let xs = self.out_norm.forward(&xs)?;
        let logits = self.text_linear.forward(&xs)?;
        Ok((logits, xs))
    }

    /// One probability per `vad_horizons` entry: how likely the speaker has
    /// stopped within that horizon. Near 0 mid-utterance, 1 once it ends.
    pub fn extra_heads(&self, xs: &Tensor<Q::T, Q::B>) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(self.extra_heads.len());
        for head in self.extra_heads.iter() {
            let pr = head.forward(xs)?.softmax()?.to_vec()?;
            out.push(<Q::T as WithDTypeF>::to_f32(pr[0]));
        }
        Ok(out)
    }
}
