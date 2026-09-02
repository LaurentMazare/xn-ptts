use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

/// RMS norm: a single `alpha` gain, no mean removal.
///
/// Backed by the `rms_norm` tensor primitive, which uses a biased variance
/// estimator. The flow LM's timestep embedder needs the unbiased one and so
/// builds its norm out of `LayerNorm` flags instead — see `mlp.rs`.
pub struct RmsNorm<T: WithDTypeF, B: Backend> {
    alpha: Tensor<T, B>,
    eps: f32,
}

impl<T: WithDTypeF, B: Backend> RmsNorm<T, B> {
    /// The checkpoint stores `alpha` as `(1, 1, dim)`.
    pub fn load(vb: &Path<B>, dim: usize, eps: f32) -> Result<Self> {
        let alpha = vb.tensor::<T>("alpha", (1, 1, dim))?.reshape((dim,))?;
        Ok(Self { alpha, eps })
    }

    pub fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        xs.rms_norm(&self.alpha, self.eps)
    }
}
