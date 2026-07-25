use crate::conditioners::LUTConditioner;
use crate::mlp::SimpleMLPAdaLN;
use crate::transformer::{StreamingTransformer, StreamingTransformerState};
use xn::nn::{Linear, var_builder::Path};
use xn::{Backend, BackendQ, Result, Tensor, WithDTypeF};

pub trait Rng {
    fn sample(&mut self) -> f32;
}

/// Lagrangian Self Distillation decode.
/// Rebuilds the data sample from starting point x_0.
fn lsd_decode<T: WithDTypeF, B: Backend>(
    flow_net: &SimpleMLPAdaLN<T, B>,
    transformer_out: &Tensor<T, B>,
    x_0: &Tensor<T, B>,
    num_steps: usize,
) -> Result<Tensor<T, B>> {
    let mut current = x_0.clone();
    let dev = x_0.device();

    for i in 0..num_steps {
        let s_val = i as f32 / num_steps as f32;
        let t_val = (i + 1) as f32 / num_steps as f32;

        // Create s and t tensors matching x_0 shape but with last dim = 1
        let shape: Vec<usize> =
            x_0.dims().iter().copied().take(x_0.rank() - 1).chain([1]).collect();
        let s = Tensor::full(T::from_f32(s_val), shape.clone(), dev)?;
        let t = Tensor::full(T::from_f32(t_val), shape, dev)?;

        let flow_dir = flow_net.forward(transformer_out, &[&s, &t], &current)?;
        let step_scale = T::from_f32(1.0 / num_steps as f32);
        current = current.add(&flow_dir.scale(step_scale)?)?;
    }
    Ok(current)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FlowLMConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dim_feedforward: usize,
    pub max_period: f32,
    pub n_bins: usize,
    pub lut_dim: usize,
    pub flow_dim: usize,
    pub flow_depth: usize,
    pub ldim: usize,
}

/// Transformer-based flow language model.
pub struct FlowLM<Q: BackendQ> {
    pub conditioner: LUTConditioner<Q::T, Q::B>,
    pub num_speakers: Option<Tensor<Q::T, Q::B>>,
    flow_net: SimpleMLPAdaLN<Q::T, Q::B>,
    pub transformer: StreamingTransformer<Q>,
    pub emb_std: Tensor<Q::T, Q::B>,
    pub emb_mean: Tensor<Q::T, Q::B>,
    bos_emb: Tensor<Q::T, Q::B>,
    pub input_linear: Linear<Q::T, Q::B>,
    out_norm_weight: Tensor<Q::T, Q::B>,
    out_norm_bias: Tensor<Q::T, Q::B>,
    out_eos: Linear<Q::T, Q::B>,
    pub dim: usize,
    pub ldim: usize,
}

#[derive(Clone, Debug)]
pub struct FlowLMState<Q: BackendQ> {
    pub transformer_state: StreamingTransformerState<Q::T, Q::B>,
    /// Per-item key-padding flags covering every position processed so far,
    /// `padding[b][p]` being true when position `p` of batch item `b` is a padding
    /// position that must not be attended to. `None` when the stream contains no
    /// padding at all (the common case), which skips the mask computation entirely.
    pub padding: Option<Vec<Vec<bool>>>,
}

impl<Q: BackendQ> FlowLMState<Q> {
    /// Number of positions processed so far.
    pub fn current_end(&self) -> usize {
        match self.transformer_state.layer_states.first() {
            Some(crate::transformer::LayerAttentionState::FlowLm(s)) => s.current_end,
            Some(crate::transformer::LayerAttentionState::Mimi(s)) => s.absolute_offset(),
            None => 0,
        }
    }

    /// Record the padding flags for `new_padding.len()` new positions of each batch
    /// item before they get processed. Once padding tracking is enabled (some
    /// position of some item was padded), it stays on for the whole stream.
    pub fn extend_padding(&mut self, new_padding: &[Vec<bool>]) -> xn::Result<()> {
        let any_padding = new_padding.iter().any(|row| row.iter().any(|&p| p));
        if self.padding.is_none() && !any_padding {
            return Ok(());
        }
        let current_end = self.current_end();
        let padding =
            self.padding.get_or_insert_with(|| vec![vec![false; current_end]; new_padding.len()]);
        if padding.len() != new_padding.len() {
            xn::bail!(
                "batch size mismatch in padding tracking: {} vs {}",
                padding.len(),
                new_padding.len()
            )
        }
        for (row, new_row) in padding.iter_mut().zip(new_padding.iter()) {
            row.extend_from_slice(new_row);
        }
        Ok(())
    }

    /// Record `len` new non-padding positions for each batch item.
    pub fn extend_valid(&mut self, len: usize) {
        if let Some(padding) = self.padding.as_mut() {
            for row in padding.iter_mut() {
                row.extend(std::iter::repeat_n(false, len));
            }
        }
    }
}

impl<Q: BackendQ> FlowLM<Q> {
    pub fn load(
        vb: &Path<Q::B>,
        tokenizer: Box<dyn crate::Tokenizer + Send + Sync>,
        cfg: &FlowLMConfig,
    ) -> Result<Self> {
        let conditioner = LUTConditioner::load(
            &vb.pp("conditioner"),
            cfg.n_bins,
            Some(tokenizer),
            cfg.lut_dim,
            cfg.d_model,
        )?;
        let num_speakers = {
            let vb = vb.pp("condition_provider.conditioners.num_speakers");
            if vb.contains("embed.weight") {
                let conditioner = LUTConditioner::load(&vb, 31, None, 16, cfg.d_model)?;
                let condition_tensor = conditioner.embed_tokens(&[1])?;
                Some(condition_tensor)
            } else {
                None
            }
        };

        let flow_net = SimpleMLPAdaLN::load(
            &vb.pp("flow_net"),
            cfg.ldim,       // in_channels
            cfg.flow_dim,   // model_channels
            cfg.ldim,       // out_channels
            cfg.d_model,    // cond_channels
            cfg.flow_depth, // num_res_blocks
            2,              // num_time_conds
        )?;

        let transformer = StreamingTransformer::load(
            &vb.pp("transformer"),
            cfg.d_model,
            cfg.num_heads,
            cfg.num_layers,
            None,
            cfg.dim_feedforward,
            None,
            cfg.max_period,
            crate::transformer::Kind::FlowLm,
        )?;

        let emb_std = vb.tensor("emb_std", (cfg.ldim,))?;
        let emb_mean = vb.tensor("emb_mean", (cfg.ldim,))?;
        let bos_emb = vb.tensor("bos_emb", (cfg.ldim,))?;
        let input_linear = Linear::load(vb.pp("input_linear"), cfg.ldim, cfg.d_model)?;
        let out_norm_weight = vb.pp("out_norm").tensor("weight", (cfg.d_model,))?;
        let out_norm_bias = vb.pp("out_norm").tensor("bias", (cfg.d_model,))?;
        let out_eos = Linear::load_b(vb.pp("out_eos"), cfg.d_model, 1)?;

        Ok(Self {
            conditioner,
            num_speakers,
            flow_net,
            transformer,
            emb_std,
            emb_mean,
            bos_emb,
            input_linear,
            out_norm_weight,
            out_norm_bias,
            out_eos,
            dim: cfg.d_model,
            ldim: cfg.ldim,
        })
    }

    pub fn init_state(&self, batch_size: usize, sequence_length: usize) -> Result<FlowLMState<Q>> {
        let transformer_state = self.transformer.init_state(batch_size, sequence_length)?;
        Ok(FlowLMState { transformer_state, padding: None })
    }

    /// Run the backbone: concat text_embeddings + input, run transformer, strip prefix.
    fn backbone(
        &self,
        input: &Tensor<Q::T, Q::B>,
        text_embeddings: &Tensor<Q::T, Q::B>,
        seq_len: usize,
        state: &mut FlowLMState<Q>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        let input = match self.num_speakers.as_ref() {
            Some(ns) => input.broadcast_add(ns)?,
            None => input.clone(),
        };
        let input = Tensor::cat(&[text_embeddings, &input], 1)?;
        state.extend_valid(input.dim(1usize)?);
        let out = self.transformer.forward_masked(
            &input,
            &mut state.transformer_state,
            state.padding.as_deref(),
        )?;
        let out = out.layer_norm(&self.out_norm_weight, &self.out_norm_bias, 1e-5)?;
        // Remove prefix, keep only last seq_len positions
        let total = out.dim(1usize)?;
        let start = total - seq_len;
        out.narrow(1, start..total)?.contiguous()
    }

    /// Sample next latent using flow matching.
    /// Returns (next_latent [B, 1, ldim], is_eos).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn sample_next_latent(
        &self,
        sequence: &Tensor<Q::T, Q::B>,
        text_embeddings: &Tensor<Q::T, Q::B>,
        state: &mut FlowLMState<Q>,
        lsd_decode_steps: usize,
        rng: &mut impl Rng,
        eos_threshold: f32,
    ) -> Result<(Tensor<Q::T, Q::B>, bool)> {
        let (latent, is_eos) = self.sample_next_latents(
            sequence,
            text_embeddings,
            state,
            lsd_decode_steps,
            rng,
            eos_threshold,
        )?;
        Ok((latent, is_eos[0]))
    }

    /// Batched variant of `sample_next_latent`.
    /// Returns (next_latent [B, 1, ldim], per-item is_eos flags of length B).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn sample_next_latents(
        &self,
        sequence: &Tensor<Q::T, Q::B>,
        text_embeddings: &Tensor<Q::T, Q::B>,
        state: &mut FlowLMState<Q>,
        lsd_decode_steps: usize,
        rng: &mut impl Rng,
        eos_threshold: f32,
    ) -> Result<(Tensor<Q::T, Q::B>, Vec<bool>)> {
        let (b, s, _) = sequence.dims3()?;
        let dev = sequence.device();

        let sequence = self.replace_nan_with_bos(sequence)?;
        let input = self.input_linear.forward(&sequence)?;
        let transformer_out = self.backbone(&input, text_embeddings, s, state)?;
        let t_len = transformer_out.dim(1usize)?;
        let transformer_out = transformer_out.narrow(1, t_len - 1..t_len)?.contiguous()?;
        let transformer_out = transformer_out.reshape((b, self.dim))?;

        let eos_logit = self.out_eos.forward(&transformer_out)?;
        let eos_val = eos_logit.to_vec()?;
        let is_eos = eos_val.iter().map(|v| v.to_f32() > eos_threshold).collect::<Vec<_>>();
        let noise_data: Vec<Q::T> =
            (0..b * self.ldim).map(|_| Q::T::from_f32(rng.sample())).collect();
        let noise = Tensor::from_vec(noise_data, (b, self.ldim), dev)?;
        let latent = lsd_decode(&self.flow_net, &transformer_out, &noise, lsd_decode_steps)?;
        let latent = latent.reshape((b, 1, self.ldim))?;
        Ok((latent, is_eos))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn sample_next_latent_cfg(
        &self,
        sequence: &Tensor<Q::T, Q::B>,
        text_embeddings: &Tensor<Q::T, Q::B>,
        state: &mut FlowLMState<Q>,
        null_state: &mut FlowLMState<Q>,
        cfg_coef: f32,
        lsd_decode_steps: usize,
        rng: &mut impl Rng,
        eos_threshold: f32,
    ) -> Result<(Tensor<Q::T, Q::B>, bool)> {
        let (b, s, _) = sequence.dims3()?;
        let dev = sequence.device();

        let sequence = self.replace_nan_with_bos(sequence)?;
        let input = self.input_linear.forward(&sequence)?;
        let t_out = self.backbone(&input, text_embeddings, s, state)?;
        let t_len = t_out.dim(1usize)?;
        let t_out = t_out.narrow(1, t_len - 1..t_len)?.contiguous()?;
        let t_out = t_out.reshape((b, self.dim))?;
        let null_out = self.backbone(&input, text_embeddings, s, null_state)?;
        let null_out =
            null_out.narrow(1, t_len - 1..t_len)?.contiguous()?.reshape((b, self.dim))?;
        let s = Q::T::from_f32(cfg_coef);
        let t_out = t_out.sub(&null_out)?.scale(s)?.add(&null_out)?;
        let eos_logit = self.out_eos.forward(&t_out)?;
        let eos_val = eos_logit.to_vec()?;
        let is_eos = eos_val[0].to_f32() > eos_threshold;
        let noise_data: Vec<Q::T> =
            (0..b * self.ldim).map(|_| Q::T::from_f32(rng.sample())).collect();
        let noise = Tensor::from_vec(noise_data, (b, self.ldim), dev)?;
        let latent = lsd_decode(&self.flow_net, &t_out, &noise, lsd_decode_steps)?;
        let latent = latent.reshape((b, 1, self.ldim))?;
        Ok((latent, is_eos))
    }

    /// Replace NaN values in sequence with bos_emb.
    fn replace_nan_with_bos(&self, sequence: &Tensor<Q::T, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        let data = sequence.to_vec()?;
        // TODO(laurent): avoid the `to_vec` below. For this, we could introduce
        // something like torch.where.
        let bos_data = self.bos_emb.to_vec()?;
        let mut out_data = data.clone();
        let ldim = self.ldim;

        for i in 0..out_data.len() {
            if out_data[i].to_f32().is_nan() {
                out_data[i] = bos_data[i % ldim];
            }
        }
        Tensor::from_vec(out_data, sequence.shape().clone(), sequence.device())
    }
}
