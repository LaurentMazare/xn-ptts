use xn::{Backend, Result, Tensor, WithDTypeF};

/// Rotary position embedding (RoPE).
/// Precomputes cos/sin tables and wraps the tensor `rope_i` primitive.
pub struct RotaryEmbedding<T: WithDTypeF, B: Backend> {
    cos: Tensor<T, B>,
    sin: Tensor<T, B>,
}

impl<T: WithDTypeF, B: Backend> RotaryEmbedding<T, B> {
    pub fn new(
        head_dim: usize,
        offset: usize,
        max_seq_len: usize,
        max_period: f32,
        device: &B,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            inv_freq.push(1.0f32 / max_period.powf(i as f32 / half_dim as f32));
        }

        let mut cos_data = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin_data = Vec::with_capacity(max_seq_len * half_dim);
        for pos in 0..max_seq_len {
            let pos = pos + offset;
            for &freq in &inv_freq {
                let angle = pos as f32 * freq;
                cos_data.push(T::from_f32(angle.cos()));
                sin_data.push(T::from_f32(angle.sin()));
            }
        }

        let cos = Tensor::from_vec(cos_data, (max_seq_len, half_dim), device)?;
        let sin = Tensor::from_vec(sin_data, (max_seq_len, half_dim), device)?;
        Ok(Self { cos, sin })
    }

    /// Build per-batch-item cos/sin tables: `positions[b][i]` is the rope position of
    /// the i-th new step for batch item `b`. This lets items of a batch advance their
    /// positions independently, e.g. when some of them carry padding that should not
    /// consume a position.
    pub fn new_per_item(
        head_dim: usize,
        positions: &[Vec<usize>],
        max_period: f32,
        device: &B,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            inv_freq.push(1.0f32 / max_period.powf(i as f32 / half_dim as f32));
        }

        let batch_size = positions.len();
        let seq_len = positions.first().map_or(0, |p| p.len());
        let mut cos_data = Vec::with_capacity(batch_size * seq_len * half_dim);
        let mut sin_data = Vec::with_capacity(batch_size * seq_len * half_dim);
        for row in positions.iter() {
            if row.len() != seq_len {
                xn::bail!("inconsistent per-item position lengths: {} vs {seq_len}", row.len())
            }
            for &pos in row.iter() {
                for &freq in &inv_freq {
                    let angle = pos as f32 * freq;
                    cos_data.push(T::from_f32(angle.cos()));
                    sin_data.push(T::from_f32(angle.sin()));
                }
            }
        }

        let cos = Tensor::from_vec(cos_data, (batch_size, seq_len, half_dim), device)?;
        let sin = Tensor::from_vec(sin_data, (batch_size, seq_len, half_dim), device)?;
        Ok(Self { cos, sin })
    }

    /// Apply RoPE to q and k tensors in [B, T, H, D] layout.
    /// Internally transposes to [B, H, T, D] for the `rope_i` primitive.
    pub fn forward(
        &self,
        q: &Tensor<T, B>,
        k: &Tensor<T, B>,
    ) -> Result<(Tensor<T, B>, Tensor<T, B>)> {
        // [B, T, H, D] -> [B, H, T, D]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let q = q.rope_i(&self.cos, &self.sin, 0)?;
        let k = k.rope_i(&self.cos, &self.sin, 0)?;
        // [B, H, T, D] -> [B, T, H, D]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        Ok((q, k))
    }
}
